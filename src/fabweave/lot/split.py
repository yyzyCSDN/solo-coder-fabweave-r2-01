from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.lot.merge import Lot


@dataclass(frozen=True, slots=True)
class MachineCapacity:
    tool_id: str
    chamber_id: str
    available: int


@dataclass(frozen=True, slots=True)
class Allocation:
    tool_id: str
    chamber_id: str
    quantity: int


@dataclass(frozen=True, slots=True)
class Sublot:
    lot_id: str
    parent_id: str
    product: str
    route: str
    step: str
    recipe_revision: int
    quantity: int
    tool_id: str
    chamber_id: str
    sequence: int


@dataclass(frozen=True, slots=True)
class SplitResult:
    parent_id: str
    remaining_quantity: int
    sublots: tuple[Sublot, ...]


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validated_capacities(capacities: list[MachineCapacity]) -> list[MachineCapacity]:
    keys = [(capacity.tool_id, capacity.chamber_id) for capacity in capacities]
    if len(set(keys)) != len(keys):
        raise ValueError("the same machine cannot offer capacity twice")
    if any(not _positive_int(capacity.available) for capacity in capacities):
        raise ValueError("available capacity must be positive")
    return sorted(capacities, key=lambda capacity: (capacity.tool_id, capacity.chamber_id))


def plan_capacity_allocations(
    parent_quantity: int,
    capacities: list[MachineCapacity],
    minimum_quantity: int,
) -> tuple[Allocation, ...]:
    """Greedy, deterministic fill over machine-id-sorted idle capacity.

    Machines are filled to their actual available quantity; the minimum only
    governs the size of a split-out sublot, so a machine that would form a
    sublot below the floor is skipped. The loop stops once the parent
    remainder itself drops below the floor: that residual stays with the
    parent lot rather than becoming an undersized child.
    """

    if not _positive_int(parent_quantity):
        raise ValueError("parent quantity must be positive")
    if not _positive_int(minimum_quantity):
        raise ValueError("minimum sublot quantity must be positive")
    ordered = _validated_capacities(capacities)
    remaining = parent_quantity
    allocations: list[Allocation] = []
    for capacity in ordered:
        if remaining < minimum_quantity:
            break
        quantity = min(capacity.available, remaining)
        if quantity < minimum_quantity:
            if capacity.available < minimum_quantity:
                continue  # idle capacity itself is below the floor, try a larger machine
            break  # the final parent remainder would be undersized; keep it
        allocations.append(Allocation(capacity.tool_id, capacity.chamber_id, quantity))
        remaining -= quantity
    return tuple(allocations)


def split_lot(
    request_id: str,
    lot: Lot,
    capacities: list[MachineCapacity],
    minimum_quantity: int,
    allocations: tuple[Allocation, ...] | None = None,
) -> SplitResult:
    """Split one parent lot into machine-bound sublots.

    Child identifiers, quantities and parent links are derived only from the
    request id and the ordered allocations, so the same request replays to the
    same result. Quantity is conserved against the parent, including a
    possible zero remainder when the lot is fully consumed.
    """

    if lot.active_holds:
        raise ValueError("held lots cannot split")
    if not _positive_int(minimum_quantity):
        raise ValueError("minimum sublot quantity must be positive")
    ordered_capacities = _validated_capacities(capacities)
    availability = {(capacity.tool_id, capacity.chamber_id): capacity.available for capacity in ordered_capacities}

    planned = tuple(allocations) if allocations is not None else plan_capacity_allocations(lot.quantity, capacities, minimum_quantity)
    if not planned:
        raise ValueError("no sublot can be formed below the minimum quantity")
    ordered_allocations = sorted(planned, key=lambda allocation: (allocation.tool_id, allocation.chamber_id))
    seen: set[tuple[str, str]] = set()
    for allocation in ordered_allocations:
        key = (allocation.tool_id, allocation.chamber_id)
        if key in seen:
            raise ValueError("the same machine cannot host two sublots of one request")
        seen.add(key)
        if not _positive_int(allocation.quantity) or allocation.quantity < minimum_quantity:
            raise ValueError("sublot quantity below the minimum")
        if key not in availability:
            raise ValueError("allocation references a machine without offered capacity")
        if allocation.quantity > availability[key]:
            raise ValueError("allocation exceeds available capacity")

    total_allocated = sum(allocation.quantity for allocation in ordered_allocations)
    if total_allocated > lot.quantity:
        raise ValueError("allocation exceeds parent quantity")
    remaining = lot.quantity - total_allocated

    sublots: list[Sublot] = []
    after = {lot.lot_id: remaining}
    for sequence, allocation in enumerate(ordered_allocations, start=1):
        child_id = deterministic_id("LOT", request_id, lot.lot_id, allocation.tool_id, allocation.chamber_id)
        sublots.append(
            Sublot(
                child_id,
                lot.lot_id,
                lot.product,
                lot.route,
                lot.step,
                lot.recipe_revision,
                allocation.quantity,
                allocation.tool_id,
                allocation.chamber_id,
                sequence,
            )
        )
        after[child_id] = allocation.quantity
    require_conservation({lot.lot_id: lot.quantity}, after)
    return SplitResult(lot.lot_id, remaining, tuple(sublots))
