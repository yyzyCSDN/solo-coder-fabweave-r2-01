from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.lot.merge import Lot


@dataclass(frozen=True, slots=True)
class MachineCapacity:
    """Idle capacity offered by one parallel machine for a split child lot."""

    tool_id: str
    chamber_id: str
    available_units: int

    def __post_init__(self) -> None:
        if self.available_units < 0:
            raise ValueError("available units cannot be negative")


@dataclass(frozen=True, slots=True)
class ChildLot:
    lot_id: str
    parent_id: str
    tool_id: str
    chamber_id: str
    quantity: int


@dataclass(frozen=True, slots=True)
class SplitPlan:
    request_id: str
    parent: Lot
    children: tuple[ChildLot, ...]
    residual_quantity: int

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("a split plan requires at least one child lot")
        if self.residual_quantity < 0:
            raise ValueError("residual quantity cannot be negative")
        if any(child.quantity <= 0 for child in self.children):
            raise ValueError("child quantity must be positive")
        if len({child.lot_id for child in self.children}) != len(self.children):
            raise ValueError("child lot ids must be unique")
        if any(child.parent_id != self.parent.lot_id for child in self.children):
            raise ValueError("child parent does not match the split lot")
        require_conservation({self.parent.lot_id: self.parent.quantity}, self.after_quantities)

    @property
    def parent_of(self) -> dict[str, str]:
        return {child.lot_id: child.parent_id for child in self.children}

    @property
    def after_quantities(self) -> dict[str, int]:
        quantities = {child.lot_id: child.quantity for child in self.children}
        quantities[self.parent.lot_id] = self.residual_quantity
        return quantities

    def wip_lots(self) -> tuple[Lot, ...]:
        return tuple(
            Lot(child.lot_id, self.parent.product, self.parent.route, self.parent.step, self.parent.recipe_revision, child.quantity)
            for child in self.children
        )


def plan_split(request_id: str, lot: Lot, min_child_quantity: int, capacities: list[MachineCapacity]) -> SplitPlan:
    """Size child lots from actual idle capacity; deterministic and all-or-nothing."""

    if min_child_quantity < 1:
        raise ValueError("minimum child quantity must be positive")
    if lot.quantity <= 0:
        raise ValueError("lot quantity must be positive")
    if lot.active_holds:
        raise ValueError("held lots cannot split")
    ordered = sorted(capacities, key=lambda capacity: (capacity.tool_id, capacity.chamber_id))
    if len({(capacity.tool_id, capacity.chamber_id) for capacity in ordered}) != len(ordered):
        raise ValueError("duplicate machine capacity entry")
    eligible = [capacity for capacity in ordered if capacity.available_units >= min_child_quantity]
    takes: list[tuple[MachineCapacity, int]] = []
    remaining = lot.quantity
    for capacity in eligible:
        if remaining < min_child_quantity:
            break
        take = min(capacity.available_units, remaining)
        takes.append((capacity, take))
        remaining -= take
    while 0 < remaining < min_child_quantity and takes:
        capacity, take = takes[-1]
        need = min_child_quantity - remaining
        if take - need >= min_child_quantity:
            takes[-1] = (capacity, take - need)
            remaining += need
        else:
            remaining += take
            takes.pop()
    if not takes:
        raise ValueError("available capacity cannot yield a valid child lot")
    children = tuple(
        ChildLot(
            deterministic_id("LOT", request_id, lot.lot_id, capacity.tool_id, capacity.chamber_id, index),
            lot.lot_id,
            capacity.tool_id,
            capacity.chamber_id,
            take,
        )
        for index, (capacity, take) in enumerate(takes)
    )
    return SplitPlan(request_id, lot, children, remaining)


class SplitLedger:
    """Append-only registry of committed split plans with replay protection."""

    def __init__(self) -> None:
        self._plans: dict[str, SplitPlan] = {}
        self._children: dict[str, ChildLot] = {}

    def register(self, plan: SplitPlan) -> SplitPlan:
        existing = self._plans.get(plan.request_id)
        if existing is not None:
            if existing != plan:
                raise ValueError("request id reused with a different split plan")
            return existing
        clashes = sorted(child.lot_id for child in plan.children if child.lot_id in self._children)
        if clashes:
            raise ValueError(f"child lot ids already registered: {', '.join(clashes)}")
        self._plans[plan.request_id] = plan
        for child in plan.children:
            self._children[child.lot_id] = child
        return plan

    def plan_for(self, request_id: str) -> SplitPlan | None:
        return self._plans.get(request_id)

    def children_of(self, parent_lot_id: str) -> tuple[ChildLot, ...]:
        ordered = sorted(self._children.values(), key=lambda child: child.lot_id)
        return tuple(child for child in ordered if child.parent_id == parent_lot_id)

    def parent_of(self) -> dict[str, str]:
        ordered = sorted(self._children.values(), key=lambda child: child.lot_id)
        return {child.lot_id: child.parent_id for child in ordered}
