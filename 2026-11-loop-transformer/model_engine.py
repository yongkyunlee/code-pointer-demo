"""Frozen, no-cache forward passes for the four-stage checkpoint experiments.

Sites are (one-based stage, one-based block offset). Captures and edits apply
after the physical decoder block and before any stage-end RMSNorm. Ouro feeds
its normalized state into the next pass. Qwen feeds its residual stream into
the next layer and normalizes only for an output-head readout.
"""
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINTS = {
    "ouro": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
    "qwen": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
}


@dataclass
class Run:
    logits: torch.Tensor
    states: dict
    engine: object

    def stage_logits(self, stage, positions=None):
        """Project a captured stage boundary through the frozen output head."""
        hidden = self.states[(stage, self.engine.width)]
        if positions is not None:
            hidden = hidden[:, positions, :]
        return self.engine.readout(hidden)


class Engine:
    def __init__(self, name, model, tokenizer):
        self.name = name
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        layers = model.model.layers
        if name == "ouro":
            if model.model.total_ut_steps != 4:
                raise ValueError("Expected four native Ouro passes")
            self.width = len(layers)
        elif name == "qwen" and len(layers) % 4 == 0:
            self.width = len(layers) // 4
        else:
            raise ValueError("Expected Ouro or a Qwen stack divisible into four stages")
        self.sdpa = model.config._attn_implementation == "sdpa"
        if model.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("This adapter requires eager or math SDPA attention")
        if any(layer.attention_type != "full_attention" for layer in layers):
            raise ValueError("This experiment assumes full causal attention throughout")

    @classmethod
    def load(cls, name):
        if not torch.cuda.is_available():
            raise RuntimeError("Select a Kaggle GPU accelerator before loading weights")
        repo, revision = CHECKPOINTS[name]
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            code_revision=revision,
            trust_remote_code=(name == "ouro"),
            use_safetensors=True,
            torch_dtype=torch.float16,
            attn_implementation="eager" if name == "ouro" else "sdpa",
            device_map={"": "cuda:0"},
        )
        return cls(name, model, tokenizer)

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def readout(self, hidden):
        return self.model.lm_head(self.model.model.norm(hidden))

    def attention_context(self):
        # T4 math SDPA accumulates FP16 QK, softmax and AV in FP32 by default.
        # Force the same kernel for native causal and experimental 4-D masks.
        return sdpa_kernel(SDPBackend.MATH) if self.sdpa else nullcontext()

    def run(self, ids, *, mask_fn=None, edit_fn=None, capture=(), grad=False,
            position_offset=0, logits_positions=None):
        """Run four full stages on one sequence.

        mask_fn receives an independently writable additive causal mask
        [1,1,query,key]. edit_fn may return a replacement state or None.
        For sparse output, pass a slice/list as logits_positions.
        """
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1, -1)
        length = ids.shape[1]
        if length == 0 or position_offset < 0:
            raise ValueError("Expected a nonempty sequence and nonnegative RoPE offset")
        sites = set(capture)
        if any(not (1 <= stage <= 4 and 1 <= offset <= self.width)
               for stage, offset in sites):
            raise ValueError(f"Capture site outside this checkpoint: {sites}")
        backbone = self.model.model
        cache_position = torch.arange(length, device=self.device)
        positions = (cache_position + position_offset).unsqueeze(0)
        states = {}
        with torch.set_grad_enabled(grad), self.attention_context():
            hidden = backbone.embed_tokens(ids)
            # Frozen parameters need a differentiable input to expose gradients.
            if grad:
                hidden.requires_grad_(True)
            rotary = backbone.rotary_emb(hidden, positions)
            blocked = float("-inf") if self.sdpa else torch.finfo(self.dtype).min
            causal = torch.full((length, length), blocked,
                                dtype=self.dtype, device=self.device)
            causal = causal.triu(1)[None, None, :, :]
            for stage in range(1, 5):
                start = 0 if self.name == "ouro" else (stage - 1) * self.width
                extra = {"current_ut": stage - 1} if self.name == "ouro" else {}
                for index in range(self.width):
                    site = (stage, index + 1)
                    mask = causal if mask_fn is None else mask_fn(*site, causal.clone())
                    if self.sdpa and mask_fn is not None:
                        mask = mask.masked_fill(mask == torch.finfo(mask.dtype).min, float("-inf"))
                    hidden = backbone.layers[start + index](
                        hidden,
                        attention_mask=mask,
                        position_ids=positions,
                        cache_position=cache_position,
                        position_embeddings=rotary,
                        use_cache=False,
                        **extra,
                    )
                    if edit_fn is not None:
                        replacement = edit_fn(*site, hidden)
                        if replacement is not None:
                            hidden = replacement
                    if site in sites:
                        states[site] = hidden
                if self.name == "ouro" or stage == 4:
                    hidden = backbone.norm(hidden)
            if logits_positions is not None:
                hidden = hidden[:, logits_positions, :]
            logits = self.model.lm_head(hidden)
        return Run(logits, states, self)

    @torch.no_grad()
    def validate(self, ids=None):
        """Fail before measurement if native, manual, or sham executions diverge."""
        if ids is None:
            ids = self.ids("The blue box contains a pear. It contains a")
        if self.sdpa and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        x = torch.tensor(ids, device=self.device).reshape(1, -1)
        boundaries = [(stage, self.width) for stage in range(1, 5)]

        def finite(tensor, label):
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).nonzero()[0].tolist()
                raise FloatingPointError(f"{self.name} {self.dtype} {label} nonfinite at {bad}")

        def check_block(stage, offset, hidden):
            finite(hidden, f"manual stage {stage}, block {offset}")

        manual = self.run(ids, capture=boundaries, edit_fn=check_block)
        finite(manual.logits, "manual output head")
        native_states = {}
        handles = []
        for index, layer in enumerate(self.model.model.layers):
            def record(module, args, kwargs, value, physical=index):
                if self.name == "ouro":
                    site = (kwargs["current_ut"] + 1, physical + 1)
                else:
                    group, within = divmod(physical, self.width)
                    site = (group + 1, within + 1)
                finite(value, f"native stage {site[0]}, block {site[1]}")
                if site in boundaries:
                    native_states[site] = value
            handles.append(layer.register_forward_hook(record, with_kwargs=True))
        try:
            extra = {"exit_at_step": 3} if self.name == "ouro" else {}
            with self.attention_context():
                native = self.model(input_ids=x, use_cache=False, **extra).logits
        finally:
            for handle in handles:
                handle.remove()
        sham = self.run(ids, mask_fn=lambda stage, offset, mask: mask,
                        edit_fn=lambda stage, offset, hidden: hidden).logits

        def max_error(actual, expected):
            finite(actual, "comparison output")
            finite(expected, "native output")
            error = (actual.float() - expected.float()).abs().max().item()
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.001)
            return error

        return {
            "checkpoint": CHECKPOINTS[self.name],
            "blocks_per_stage": self.width,
            "executed_depth": self.width * 4,
            "final_native_max_abs_error": max_error(manual.logits, native),
            "stage_native_max_abs_errors": [
                max_error(manual.states[k], native_states[k]) for k in boundaries
            ],
            "stage_four_head_max_abs_error": max_error(manual.stage_logits(4), native),
            "sham_final_max_abs_error": max_error(sham, native),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.dtype),
            "attention": "sdpa_math" if self.sdpa else "eager",
            "use_cache": False,
            "token_ids": x.flatten().tolist(),
        }
