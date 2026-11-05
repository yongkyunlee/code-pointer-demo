"""Supply fixtures to Backend.AI without starting its external services.

The scheduler, quota validator, FIFO sequencer, selectors, and phase recorder
are upstream code. Only persistence and queue storage use in-memory adapters.
Nothing in this module launches containers or observes workload execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import NAMESPACE_URL, uuid5

from ai.backend.common.data.entity.domain import DomainID
from ai.backend.common.data.entity.project import ProjectID
from ai.backend.common.data.entity.resource_group import ResourceGroupID, ResourceGroupName
from ai.backend.common.data.entity.resource_slot import ResourceSlotName
from ai.backend.common.data.entity.user import UserID
from ai.backend.common.types import (
    AccessKey,
    AgentId,
    AgentSelectionStrategy,
    ArchName,
    ClusterMode,
    KernelId,
    PreemptionOrder,
    SessionId,
    SessionTypes,
)
from ai.backend.manager.data.session.options import AgentSelectionPolicy
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.scheduler.provisioner.provisioner import (
    SessionProvisioner,
    SessionProvisionerArgs,
)
from ai.backend.manager.sokovan.scheduler.provisioner.selectors.pool import create_agent_selector
from ai.backend.manager.sokovan.scheduler.provisioner.sequencers.fifo import FIFOSequencer
from ai.backend.manager.sokovan.scheduler.provisioner.validators.dependencies import (
    DependenciesValidator,
)
from ai.backend.manager.sokovan.scheduler.provisioner.validators.reserved_batch import (
    ReservedBatchSessionValidator,
)
from ai.backend.manager.sokovan.scheduler.provisioner.validators.resource_policy import (
    ResourcePolicyValidator,
)
from ai.backend.manager.sokovan.scheduler.provisioner.validators.validator import (
    SchedulingValidator,
)
from ai.backend.manager.views.sokovan.agent import (
    AgentLimit,
    AgentMeta,
    AgentResource,
    ResourceGroupResource,
    SlotResource,
)
from ai.backend.manager.views.sokovan.allocation import SessionAllocation
from ai.backend.manager.views.sokovan.resource_group import ResourceGroupMeta
from ai.backend.manager.views.sokovan.scheduling import SchedulingData
from ai.backend.manager.views.sokovan.snapshot import (
    GlobalScopeSnapshot,
    ResourceGroupSchedulingPolicy,
    ResourceGroupScopeSnapshot,
    ResourceOccupancySnapshot,
    ResourcePolicySnapshot,
    SessionDependencySnapshot,
    SystemSnapshot,
    UserResourceLimit,
)
from ai.backend.manager.views.sokovan.workload import (
    KernelWorkload,
    ResourceRequest,
    SessionPlacement,
    SessionWorkload,
    WorkloadMeta,
    WorkloadOwner,
)

if TYPE_CHECKING:
    from ai.backend.common.clients.valkey_client.valkey_schedule.client import ValkeyScheduleClient
    from ai.backend.manager.config.provider import ManagerConfigProvider
    from ai.backend.manager.repositories.fair_share import FairShareRepository
    from ai.backend.manager.repositories.scheduler import SchedulerRepository


CPU = ResourceSlotName("cpu")
MEM = ResourceSlotName("mem")
GIB = 1024**3
CAPACITY = 4
MEMORY_CAPACITY = 16 * GIB
AGENT_ID = AgentId("worker-1")
GROUP_NAME = ResourceGroupName("experiment")
GROUP_ID = ResourceGroupID(uuid5(NAMESPACE_URL, "code-pointer-demo/resource-group"))
DOMAIN_ID = DomainID(uuid5(NAMESPACE_URL, "code-pointer-demo/domain"))
ARCHITECTURE = ArchName("x86_64")  # Synthetic metadata, not the host architecture.


class ExperimentError(RuntimeError):
    """The experiment's inputs or observed results violate its contract."""


class Decision(TypedDict):
    session_id: str
    outcome: str
    reason: str


class PassRecording(TypedDict):
    decisions: list[Decision]
    snapshot: dict[str, object]
    upstream_records: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SessionSpec:
    """One deterministic fixture identity and its requested CPU slots."""

    name: str
    label: str
    cpus: int

    @property
    def session_id(self) -> SessionId:
        return SessionId(uuid5(NAMESPACE_URL, f"code-pointer-demo/session/{self.name}"))

    def workload(self) -> SessionWorkload:
        return SessionWorkload(
            meta=WorkloadMeta(
                session_id=self.session_id,
                resource_group_id=GROUP_ID,
                owner=WorkloadOwner(
                    access_key=AccessKey(f"FIXTURE-{self.name.upper()}"),
                    user_uuid=UserID(uuid5(NAMESPACE_URL, f"code-pointer-demo/user/{self.name}")),
                    project_id=ProjectID(
                        uuid5(NAMESPACE_URL, f"code-pointer-demo/project/{self.name}")
                    ),
                    domain_id=DOMAIN_ID,
                ),
            ),
            placement=SessionPlacement(
                cluster_mode=ClusterMode.SINGLE_NODE,
                kernels=[
                    KernelWorkload(
                        kernel_id=KernelId(
                            uuid5(NAMESPACE_URL, f"code-pointer-demo/kernel/{self.name}")
                        ),
                        architecture=ARCHITECTURE,
                        requested_slots=ResourceRequest(
                            slots={CPU: Decimal(self.cpus), MEM: Decimal(GIB)}
                        ),
                    )
                ],
                agent_selection_policy=AgentSelectionPolicy.STRICT,
                designated_agent_ids=None,
                session_group=None,
            ),
            priority=0,
            job_priority=0,
            session_type=SessionTypes.INTERACTIVE,
            requested_starts_at=None,
            is_preemptible=False,
        )


SESSION_SPECS = (
    SessionSpec("current", "Current", 2),
    SessionSpec("alice", "Alice", 4),
    SessionSpec("bob", "Bob", 1),
)


@dataclass
class FixtureState:
    """Input state maintained by this experiment, not a Backend.AI database."""

    specs: tuple[SessionSpec, ...] = SESSION_SPECS
    allocations: dict[str, int] = field(default_factory=lambda: {"current": 2})
    pending: list[str] = field(default_factory=lambda: ["alice", "bob"])
    released: set[str] = field(default_factory=set)
    last_decisions: dict[str, dict[str, str]] = field(default_factory=dict)

    def validate(self) -> None:
        specs = {spec.name: spec for spec in self.specs}
        if len(specs) != len(self.specs) or any(spec.cpus < 1 for spec in self.specs):
            raise ExperimentError("Session fixtures need unique names and positive CPU requests")
        allocated, pending = set(self.allocations), set(self.pending)
        if len(pending) != len(self.pending):
            raise ExperimentError("A session appears twice in the pending queue")
        if allocated & pending or allocated & self.released or pending & self.released:
            raise ExperimentError("Allocated, pending, and released sessions must be disjoint")
        if allocated | pending | self.released != set(specs):
            raise ExperimentError("Every fixture session must have exactly one state")
        if any(slots != specs[name].cpus for name, slots in self.allocations.items()):
            raise ExperimentError("An allocation differs from the session's resource request")
        if not 0 <= sum(self.allocations.values()) <= CAPACITY:
            raise ExperimentError("Fixture CPU allocations exceed worker capacity")

    def release(self, name: str) -> None:
        """Supply a release input; do not claim to terminate a real workload."""
        self.validate()
        if name not in self.allocations:
            raise ExperimentError(f"Cannot release unallocated session {name!r}")
        del self.allocations[name]
        self.released.add(name)
        self.last_decisions.pop(name, None)
        self.validate()

    def snapshot(self) -> dict[str, object]:
        self.validate()
        allocated = sum(self.allocations.values())
        return {
            "node": {
                "id": str(AGENT_ID),
                "capacity": CAPACITY,
                "allocated": allocated,
                "free": CAPACITY - allocated,
                "allocations": [
                    {"session_id": spec.name, "slots": self.allocations[spec.name]}
                    for spec in self.specs
                    if spec.name in self.allocations
                ],
            },
            "sessions": [
                {
                    "id": spec.name,
                    "label": spec.label,
                    "requested": spec.cpus,
                    "status": (
                        "allocated"
                        if spec.name in self.allocations
                        else "pending"
                        if spec.name in self.pending
                        else "released"
                    ),
                    "last_decision": (
                        dict(self.last_decisions[spec.name])
                        if spec.name in self.last_decisions
                        else None
                    ),
                }
                for spec in self.specs
            ],
            "pending_order": list(self.pending),
        }


class _AllocationSink:
    """Capture upstream write-boundary values without claiming SQL persistence."""

    def __init__(self) -> None:
        self.allocations: list[SessionAllocation] = []

    async def allocate_sessions(self, allocations: Sequence[SessionAllocation]) -> list[SessionId]:
        self.allocations.extend(allocations)
        return [allocation.session_id for allocation in allocations]

    async def reserve_sessions(self, allocations: Sequence[SessionAllocation]) -> list[SessionId]:
        if allocations:
            raise ExperimentError("Preemption reservations are outside this experiment")
        return []


class _PendingQueueSink:
    """Capture the queue that upstream would persist in Valkey."""

    def __init__(self) -> None:
        self.pending: list[SessionId] | None = None

    async def set_pending_queue(
        self, resource_group: ResourceGroupName, session_ids: Sequence[SessionId]
    ) -> None:
        if resource_group != GROUP_NAME or self.pending is not None:
            raise ExperimentError("Unexpected pending-queue write")
        self.pending = list(session_ids)


class _UnusedDependency:
    def __getattr__(self, name: str) -> object:
        raise ExperimentError(f"Unexpected external dependency access: {name}")


def _scheduling_data(state: FixtureState, user_cpu_limits: dict[str, int] | None) -> SchedulingData:
    state.validate()
    workloads = {spec.name: spec.workload() for spec in state.specs}
    occupancy = ResourceOccupancySnapshot(by_user={}, by_project={}, by_domain={})
    for name in state.allocations:
        workload = workloads[name]
        owner = workload.meta.owner
        occupancy.add_occupancy(
            owner.user_uuid, owner.project_id, owner.domain_id, workload.requested_slots
        )
    limits = user_cpu_limits or {}
    if set(limits) - set(workloads) or any(value < 0 for value in limits.values()):
        raise ExperimentError("User CPU limits must name fixture sessions and be nonnegative")
    resource_policy = ResourcePolicySnapshot(
        by_user={
            workload.meta.owner.user_uuid: UserResourceLimit(
                slots={CPU: Decimal(limits.get(name, CAPACITY)), MEM: Decimal(MEMORY_CAPACITY)},
                max_session_count=None,
                max_sftp_session_count=None,
            )
            for name, workload in workloads.items()
        },
        by_project={},
        by_domain={},
    )
    return SchedulingData(
        resource_group=ResourceGroupMeta(id=GROUP_ID, name=GROUP_NAME),
        workloads=[workloads[name] for name in state.pending],
        system_snapshot=SystemSnapshot(
            resource_group=ResourceGroupScopeSnapshot(
                resources=ResourceGroupResource(
                    agents=[
                        AgentMeta(
                            id=AGENT_ID,
                            addr="tcp://fixture-worker.invalid:6001",
                            architecture=ARCHITECTURE,
                            resources=AgentResource(
                                slots={
                                    CPU: SlotResource(
                                        capacity=Decimal(CAPACITY),
                                        reserved=Decimal(sum(state.allocations.values())),
                                        used=Decimal(0),
                                    ),
                                    MEM: SlotResource(
                                        capacity=Decimal(MEMORY_CAPACITY),
                                        reserved=Decimal(len(state.allocations) * GIB),
                                        used=Decimal(0),
                                    ),
                                }
                            ),
                            container_count=0,
                        )
                    ]
                ),
                session_dependencies=SessionDependencySnapshot(by_session={}),
                policy=ResourceGroupSchedulingPolicy(
                    scheduler="fifo",
                    agent_selection_strategy=AgentSelectionStrategy.CONCENTRATED,
                    preemption_order=PreemptionOrder.OLDEST,
                ),
            ),
            global_scope=GlobalScopeSnapshot(
                occupancy=occupancy,
                resource_policy=resource_policy,
                agent_limit=AgentLimit(max_container_count=None),
            ),
            observed_at=datetime.now(UTC),
        ),
    )


async def schedule_pass(
    state: FixtureState, *, user_cpu_limits: dict[str, int] | None = None
) -> PassRecording:
    """Run upstream once and apply only its accepted allocation decisions."""
    data = _scheduling_data(state, user_cpu_limits)
    allocations, queue = _AllocationSink(), _PendingQueueSink()
    # Upstream annotations name concrete service classes. These explicit casts
    # adapt their small write interfaces without constructing external clients.
    provisioner = SessionProvisioner(
        SessionProvisionerArgs(
            validator=SchedulingValidator(
                [
                    DependenciesValidator(),
                    ReservedBatchSessionValidator(),
                    ResourcePolicyValidator(),
                ]
            ),
            default_sequencer=FIFOSequencer(),
            agent_selector=create_agent_selector(["cpu", "mem"]),
            repository=cast("SchedulerRepository", allocations),
            fair_share_repository=cast("FairShareRepository", _UnusedDependency()),
            config_provider=cast("ManagerConfigProvider", _UnusedDependency()),
            valkey_schedule=cast("ValkeyScheduleClient", queue),
        )
    )
    specs_by_id = {spec.session_id: spec for spec in state.specs}
    input_ids = [workload.meta.session_id for workload in data.workloads]
    with RecorderContext[SessionId].scope("schedule", entity_ids=input_ids) as pool:
        result = await provisioner.schedule_resource_group(data)
        traces = {
            specs_by_id[session_id].name: record.model_dump(mode="json")
            for session_id, record in pool.build_all_records().items()
        }

    scheduled = result.scheduled_session_ids
    failed = {failure.session_id: failure.msg for failure in result.scheduling_failures}
    skipped = {skip.session_id: skip.msg for skip in result.scheduling_skips}
    classified = list(scheduled) + list(failed) + list(skipped)
    if (
        len(classified) != len(set(classified))
        or set(classified) != set(input_ids)
        or result.reserved_session_ids
        or result.preemption_plan
    ):
        raise ExperimentError("Upstream did not classify each pending session exactly once")
    if queue.pending is None or set(queue.pending) != set(failed) | set(skipped):
        raise ExperimentError("Upstream pending-queue output disagrees with its decisions")
    if [allocation.session_id for allocation in allocations.allocations] != scheduled:
        raise ExperimentError("Upstream allocations disagree with scheduled session IDs")

    for allocation in allocations.allocations:
        spec = specs_by_id[allocation.session_id]
        expected_kernel = spec.workload().placement.kernels[0].kernel_id
        if (
            len(allocation.kernel_allocations) != 1
            or allocation.kernel_allocations[0].kernel_id != expected_kernel
            or allocation.kernel_allocations[0].agent_id != AGENT_ID
        ):
            raise ExperimentError("Upstream allocated an unexpected kernel or worker")
        state.allocations[spec.name] = spec.cpus

    decisions: list[Decision] = []
    for session_id in input_ids:
        name = specs_by_id[session_id].name
        if session_id in scheduled:
            outcome, reason = (
                "scheduled",
                "Allocation accepted by the in-memory persistence adapter",
            )
        elif session_id in failed:
            outcome, reason = "failed", failed[session_id]
        else:
            outcome, reason = "skipped", skipped[session_id]
        decision = {"outcome": outcome, "reason": reason}
        state.last_decisions[name] = decision
        decisions.append({"session_id": name, **decision})
    state.pending = [specs_by_id[session_id].name for session_id in queue.pending]
    state.validate()
    return {"decisions": decisions, "snapshot": state.snapshot(), "upstream_records": traces}
