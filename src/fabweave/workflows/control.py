from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fabweave.chamber.matching import Tolerance, mismatch_reasons
from fabweave.chemicals.shelf_life import shelf_life_remaining
from fabweave.cleanroom.excursion import active_excursion
from fabweave.contamination.segregation import incompatible_pairs
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.events import EventLog
from fabweave.core.ids import deterministic_id
from fabweave.core.timebox import TimeBox
from fabweave.dedication.tool import DedicationPolicy, policy_decision
from fabweave.dispatch.priority import DispatchCandidate, rank_candidates
from fabweave.energy.demand import demand_violations
from fabweave.genealogy.carrier import GenealogyGraph
from fabweave.holds.release import HoldState, release_decision
from fabweave.lot.merge import Lot, merge_lots
from fabweave.lot.split import MachineCapacity, SplitResult, Sublot, plan_capacity_allocations, split_lot
from fabweave.maintenance.window import MaintenancePolicy, assignment_decision
from fabweave.metrology.sampling import Wafer, deterministic_sample
from fabweave.queue_time.enforcer import QueueLimit, queue_time_decision
from fabweave.recipe.qualification import Qualification, qualification_decision
from fabweave.reticle.reservation import Reservation, ReservationBook, can_reserve
from fabweave.rework.authorization import ReworkPolicy, authorization_decision
from fabweave.shipment.commit import Allocation, CommitLedger
from fabweave.spc.control import western_electric_rules
from fabweave.takt.monitor import TaktResult, takt_result
from fabweave.yielding.rollup import StepYield, route_yield


def _combine(decisions: list[Decision], **evidence: object) -> Decision:
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    merged = {key: value for decision in decisions for key, value in decision.evidence.items()}
    merged.update(evidence)
    return deny(*reasons, **merged) if reasons else allow(**merged)


def material_readiness(
    queue_limit: QueueLimit,
    entered_at: int,
    evaluated_at: int,
    queue_pauses: list[TimeBox],
    opened_at: int,
    shelf_hours: int,
    frozen_periods: list[TimeBox],
) -> Decision:
    queue = queue_time_decision(queue_limit, entered_at, evaluated_at, queue_pauses)
    remaining = shelf_life_remaining(opened_at, shelf_hours, evaluated_at, frozen_periods)
    chemical = deny("chemical_shelf_life_expired", remaining_shelf_hours=remaining) if remaining < 0 else allow(remaining_shelf_hours=remaining)
    return _combine([queue, chemical])


def rank_releasable(
    candidates: list[DispatchCandidate],
    now: int,
    holds: dict[str, HoldState],
    expected_versions: dict[str, int],
) -> list[DispatchCandidate]:
    evaluated: list[DispatchCandidate] = []
    for candidate in candidates:
        state = holds.get(candidate.lot_id)
        blocked = candidate.active_hold
        if state is not None:
            expected = expected_versions.get(candidate.lot_id, state.version)
            blocked = blocked or not release_decision(state, expected, now).allowed
        evaluated.append(replace(candidate, active_hold=blocked))
    return rank_candidates(evaluated, now)


def reserve_qualified(
    book: ReservationBook,
    request_id: str,
    requested: list[Reservation],
    assignments: dict[str, tuple[str, str]],
    qualifications: list[Qualification],
    at: int,
) -> Decision:
    decisions: list[Decision] = []
    for reservation in requested:
        assignment = assignments.get(reservation.job_id)
        if assignment is None:
            decisions.append(deny(f"assignment_missing:{reservation.job_id}"))
            continue
        tool_id, chamber_id = assignment
        decisions.append(qualification_decision(tool_id, chamber_id, reservation.recipe_revision, at, qualifications))
    combined = _combine(decisions, request_id=request_id)
    if not combined.allowed:
        return combined
    reserved = book.reserve_batch(request_id, requested)
    return allow(request_id=request_id, reservation_count=len(reserved))


def qualified_assignment(
    policy: MaintenancePolicy,
    last_pm_at: float,
    accumulated_run_hours: float,
    now: int,
    job_window: TimeBox,
    planned_maintenance: list[TimeBox],
    tool_id: str,
    chamber_id: str,
    recipe_revision: int,
    qualifications: list[Qualification],
) -> Decision:
    maintenance = assignment_decision(policy, last_pm_at, accumulated_run_hours, now, job_window, planned_maintenance)
    qualification = qualification_decision(tool_id, chamber_id, recipe_revision, job_window.start, qualifications)
    return _combine([maintenance, qualification])


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    wafer_ids: tuple[str, ...]
    rules: tuple[str, ...]


def evaluate_sample(
    wafers: list[Wafer],
    sample_count: int,
    required_zones: set[str],
    measurements: dict[str, float],
    center: float,
    sigma: float,
) -> SampleEvaluation:
    selected = deterministic_sample(wafers, sample_count, required_zones)
    missing = sorted(wafer.wafer_id for wafer in selected if wafer.wafer_id not in measurements)
    if missing:
        raise ValueError(f"sample measurements missing: {', '.join(missing)}")
    values = [measurements[wafer.wafer_id] for wafer in selected]
    return SampleEvaluation(tuple(wafer.wafer_id for wafer in selected), western_electric_rules(values, center, sigma))


def chamber_sample_decision(
    reference: dict[str, float],
    candidate: dict[str, float],
    tolerances: dict[str, Tolerance],
    wafers: list[Wafer],
    sample_count: int,
    required_zones: set[str],
) -> Decision:
    reasons = list(mismatch_reasons(reference, candidate, tolerances))
    try:
        sample = deterministic_sample(wafers, sample_count, required_zones)
    except ValueError as error:
        reasons.append(f"sample:{error}")
        sample = ()
    evidence = {"sample_wafer_ids": tuple(wafer.wafer_id for wafer in sample)}
    return deny(*reasons, **evidence) if reasons else allow(**evidence)


def environmental_containment(
    readings: list[tuple[int, float]],
    open_limit: float,
    close_limit: float,
    required_open: int,
    required_close: int,
    material_classes: set[str],
    contamination_matrix: dict[str, set[str]],
) -> Decision:
    excursion = active_excursion(readings, open_limit, close_limit, required_open, required_close)
    pairs = incompatible_pairs(material_classes, contamination_matrix)
    reasons = [*(("cleanroom_excursion",) if excursion else ()), *(f"incompatible:{left}:{right}" for left, right in pairs)]
    return deny(*reasons, active_excursion=excursion, incompatible_pairs=pairs) if reasons else allow(active_excursion=False, incompatible_pairs=())


def containment_scope(graph: GenealogyGraph, seeds: list[str], max_depth: int) -> tuple[str, ...]:
    impacted = set(seeds)
    for seed in seeds:
        impacted.update(graph.related(seed, "upstream", max_depth))
        impacted.update(graph.related(seed, "downstream", max_depth))
    return tuple(sorted(impacted))


@dataclass(frozen=True, slots=True)
class ContainmentRecord:
    scope: tuple[str, ...]
    event_sequence: int


def record_containment(
    log: EventLog,
    request_id: str,
    graph: GenealogyGraph,
    seeds: list[str],
    max_depth: int,
    at: int,
) -> ContainmentRecord:
    scope = containment_scope(graph, seeds, max_depth)
    event = log.append(
        f"containment:{request_id}",
        0,
        request_id,
        at,
        "containment_planned",
        {"scope": list(scope)},
    )
    return ContainmentRecord(tuple(event.data["scope"]), event.sequence)


def powered_assignment(
    loads: list[tuple[int, float]],
    demand_windows: list[tuple[int, float]],
    maintenance_policy: MaintenancePolicy,
    last_pm_at: float,
    run_hours: float,
    now: int,
    job_window: TimeBox,
    maintenance_windows: list[TimeBox],
) -> Decision:
    widths = demand_violations(loads, demand_windows)
    demand = deny(*(f"demand_limit_exceeded:{width}" for width in widths), violated_windows=widths) if widths else allow(violated_windows=())
    maintenance = assignment_decision(maintenance_policy, last_pm_at, run_hours, now, job_window, maintenance_windows)
    return _combine([demand, maintenance])


def rework_release_decision(
    policy: ReworkPolicy,
    operation: str,
    prior_reworks: int,
    defect_code: str,
    engineer_approval: str | None,
    hold: HoldState,
    expected_hold_version: int,
    at: int,
) -> Decision:
    authorization = authorization_decision(policy, operation, prior_reworks, defect_code, engineer_approval)
    release = release_decision(hold, expected_hold_version, at)
    return _combine([authorization, release])


def merge_and_commit(request_id: str, lots: list[Lot], ledger: CommitLedger, order_id: str) -> tuple[Lot, Allocation]:
    merged = merge_lots(request_id, lots)
    allocation = ledger.allocate(request_id, order_id, merged.quantity)
    return merged, allocation


def dedicated_qualification(
    product: str,
    tool_id: str,
    chamber_id: str,
    recipe_revision: int,
    at: int,
    dedication_policies: list[DedicationPolicy],
    qualifications: list[Qualification],
) -> Decision:
    dedication = policy_decision(product, tool_id, at, dedication_policies)
    qualification = qualification_decision(tool_id, chamber_id, recipe_revision, at, qualifications)
    return _combine([dedication, qualification])


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    takt: TaktResult
    route_yield: float


def performance_summary(
    completions: list[int],
    window_start: int,
    window_end: int,
    steps: list[StepYield],
    expected_steps: list[str],
) -> PerformanceSummary:
    return PerformanceSummary(takt_result(completions, window_start, window_end), route_yield(steps, expected_steps))


@dataclass(frozen=True, slots=True)
class CapacityOffer:
    tool_id: str
    chamber_id: str
    available: int
    reticle_id: str
    window: TimeBox

    def __post_init__(self) -> None:
        if not isinstance(self.available, int) or isinstance(self.available, bool) or self.available <= 0:
            raise ValueError("available capacity must be positive")
        if not self.reticle_id:
            raise ValueError("reticle is required for an offered machine")

    def capacity(self) -> MachineCapacity:
        return MachineCapacity(self.tool_id, self.chamber_id, self.available)


def _offer_eligibility(
    offer: CapacityOffer,
    lot: Lot,
    at: int,
    dedication_policies: list[DedicationPolicy],
    qualifications: list[Qualification],
) -> Decision:
    dedication = policy_decision(lot.product, offer.tool_id, at, dedication_policies)
    qualification = qualification_decision(
        offer.tool_id, offer.chamber_id, lot.recipe_revision, offer.window.start, qualifications
    )
    return _combine(
        [dedication, qualification],
        tool_id=offer.tool_id,
        chamber_id=offer.chamber_id,
        reticle_id=offer.reticle_id,
    )


def _job_id(request_id: str, tool_id: str, chamber_id: str) -> str:
    return deterministic_id("JOB", request_id, tool_id, chamber_id)


def _split_payload(request_id: str, result: SplitResult, offers: tuple[CapacityOffer, ...], minimum_quantity: int) -> dict[str, Any]:
    offer_by_machine = {(offer.tool_id, offer.chamber_id): offer for offer in offers}
    return {
        "request_id": request_id,
        "lot_id": result.parent_id,
        "product": result.sublots[0].product,
        "route": result.sublots[0].route,
        "step": result.sublots[0].step,
        "recipe_revision": result.sublots[0].recipe_revision,
        "remaining_quantity": result.remaining_quantity,
        "minimum_quantity": minimum_quantity,
        "sublots": [
            {
                "lot_id": sublot.lot_id,
                "parent_id": sublot.parent_id,
                "product": sublot.product,
                "route": sublot.route,
                "step": sublot.step,
                "recipe_revision": sublot.recipe_revision,
                "quantity": sublot.quantity,
                "tool_id": sublot.tool_id,
                "chamber_id": sublot.chamber_id,
                "sequence": sublot.sequence,
                "reticle_id": offer_by_machine[(sublot.tool_id, sublot.chamber_id)].reticle_id,
                "window_start": offer_by_machine[(sublot.tool_id, sublot.chamber_id)].window.start,
                "window_end": offer_by_machine[(sublot.tool_id, sublot.chamber_id)].window.end,
            }
            for sublot in result.sublots
        ],
    }


def split_for_idle_capacity(
    log: EventLog,
    book: ReservationBook,
    request_id: str,
    lot: Lot,
    offers: list[CapacityOffer],
    minimum_quantity: int,
    at: int,
    dedication_policies: list[DedicationPolicy],
    qualifications: list[Qualification],
) -> Decision:
    """Split a WIP lot across qualified parallel machines with idle capacity.

    Fail-closed: the lot hold, the reticle reservation and the tool/chamber
    matching gates are all evaluated first, and nothing (no reservation and no
    event) is committed until every sublot is legal. Numbers, ids and
    parent/child links are deterministic, so the same request id can be
    replayed and binds the exact same outcome.
    """

    stream = f"lot-split:{request_id}"
    stored = log.read(stream)
    if stored:
        return _replay_split_decision(log, book, request_id, lot, offers, minimum_quantity, stored)

    if lot.recipe_revision <= 0:
        raise ValueError("invalid recipe revision")
    if not isinstance(minimum_quantity, int) or isinstance(minimum_quantity, bool) or minimum_quantity <= 0:
        raise ValueError("minimum sublot quantity must be positive")

    if lot.active_holds:
        return deny("held_lot_cannot_split", request_id=request_id, lot_id=lot.lot_id)

    ordered = sorted(offers, key=lambda offer: (offer.tool_id, offer.chamber_id))
    if len({(offer.tool_id, offer.chamber_id) for offer in ordered}) != len(ordered):
        raise ValueError("the same machine cannot offer capacity twice")

    reasons: list[str] = []
    eligible: list[CapacityOffer] = []
    for offer in ordered:
        decision = _offer_eligibility(offer, lot, at, dedication_policies, qualifications)
        if not decision.allowed:
            reasons.extend(f"{reason}@{offer.tool_id}:{offer.chamber_id}" for reason in decision.reasons)
        else:
            eligible.append(offer)
    if reasons:
        return deny(*sorted(set(reasons)), request_id=request_id, lot_id=lot.lot_id)

    capacities = [offer.capacity() for offer in eligible]
    allocations = plan_capacity_allocations(lot.quantity, capacities, minimum_quantity)
    if not allocations:
        return deny(
            "no_idle_capacity_meets_minimum_sublot",
            request_id=request_id,
            lot_id=lot.lot_id,
            minimum_quantity=minimum_quantity,
        )
    result = split_lot(request_id, lot, capacities, minimum_quantity, allocations)

    conflict_reasons: list[str] = []
    intra_request: dict[str, list[tuple[int, int]]] = {}
    for sublot in result.sublots:
        offer = next(item for item in eligible if (item.tool_id, item.chamber_id) == (sublot.tool_id, sublot.chamber_id))
        occupied = [
            *(
                (reservation.window.start, reservation.window.end)
                for reservation in book.reservations(offer.reticle_id)
            ),
            *intra_request.get(offer.reticle_id, ()),
        ]
        if not can_reserve(offer.window.start, offer.window.end, occupied):
            conflict_reasons.append(f"reticle_busy:{offer.reticle_id}@{offer.tool_id}:{offer.chamber_id}")
        else:
            intra_request.setdefault(offer.reticle_id, []).append((offer.window.start, offer.window.end))
    if conflict_reasons:
        return deny(*sorted(set(conflict_reasons)), request_id=request_id, lot_id=lot.lot_id)

    offer_by_machine = {
        (offer.tool_id, offer.chamber_id): offer for offer in eligible
    }
    reservations = [
        Reservation(
            offer_by_machine[(sublot.tool_id, sublot.chamber_id)].reticle_id,
            _job_id(request_id, sublot.tool_id, sublot.chamber_id),
            offer_by_machine[(sublot.tool_id, sublot.chamber_id)].window,
            lot.recipe_revision,
        )
        for sublot in result.sublots
    ]
    committed = book.reserve_batch(request_id, reservations)

    payload = _split_payload(request_id, result, tuple(ordered), minimum_quantity)
    event = log.append(
        stream,
        0,
        deterministic_id("EVT", request_id, lot.lot_id, "lot_split"),
        at,
        "lot_split",
        payload,
    )
    return allow(
        request_id=request_id,
        lot_id=lot.lot_id,
        split=result,
        reservations=committed,
        remaining_quantity=result.remaining_quantity,
        event_sequence=event.sequence,
    )


def replay_split(log: EventLog, request_id: str) -> SplitResult:
    """Rebuild the immutable split result from its append-only event."""

    stored = log.read(f"lot-split:{request_id}")
    if not stored:
        raise ValueError("unknown split request")
    return _result_from_payload(stored[-1].data)


def _result_from_payload(payload: dict[str, Any]) -> SplitResult:
    sublots = tuple(
        Sublot(
            lot_id=item["lot_id"],
            parent_id=item["parent_id"],
            product=item["product"],
            route=item["route"],
            step=item["step"],
            recipe_revision=item["recipe_revision"],
            quantity=item["quantity"],
            tool_id=item["tool_id"],
            chamber_id=item["chamber_id"],
            sequence=item["sequence"],
        )
        for item in payload["sublots"]
    )
    return SplitResult(payload["lot_id"], payload["remaining_quantity"], sublots)


def _replay_split_decision(
    log: EventLog,
    book: ReservationBook,
    request_id: str,
    lot: Lot,
    offers: list[CapacityOffer],
    minimum_quantity: int,
    stored: tuple[Any, ...],
) -> Decision:
    event = stored[-1]
    payload = event.data
    result = _result_from_payload(payload)

    identity_keys = ("lot_id", "product", "route", "step", "recipe_revision")
    expected_identity = {
        "lot_id": lot.lot_id,
        "product": lot.product,
        "route": lot.route,
        "step": lot.step,
        "recipe_revision": lot.recipe_revision,
    }
    if any(payload[key] != expected_identity[key] for key in identity_keys):
        raise ValueError("request id reused with a different lot")
    if minimum_quantity != payload["minimum_quantity"]:
        raise ValueError("request id reused with different minimum quantity")
    if payload["remaining_quantity"] + sum(item["quantity"] for item in payload["sublots"]) != lot.quantity:
        raise ValueError("recorded split violates quantity conservation against the lot")

    offer_index = {(offer.tool_id, offer.chamber_id): offer for offer in offers}
    recorded_keys = {(item["tool_id"], item["chamber_id"]) for item in payload["sublots"]}
    if recorded_keys - set(offer_index):
        raise ValueError("request id reused with different machine offers")

    reservations: list[Reservation] = []
    for item in sorted(payload["sublots"], key=lambda sublot: (sublot["tool_id"], sublot["chamber_id"])):
        expected_id = deterministic_id("LOT", request_id, lot.lot_id, item["tool_id"], item["chamber_id"])
        if item["lot_id"] != expected_id:
            raise ValueError("recorded sublot id is not deterministic for the request")
        offer = offer_index[(item["tool_id"], item["chamber_id"])]
        if offer.reticle_id != item["reticle_id"] or (offer.window.start, offer.window.end) != (
            item["window_start"],
            item["window_end"],
        ):
            raise ValueError("request id reused with different reticle or processing window")
        reservations.append(
            Reservation(
                offer.reticle_id,
                _job_id(request_id, offer.tool_id, offer.chamber_id),
                offer.window,
                lot.recipe_revision,
            )
        )

    committed = book.reserve_batch(request_id, reservations)
    return allow(
        request_id=request_id,
        lot_id=lot.lot_id,
        split=result,
        reservations=committed,
        remaining_quantity=result.remaining_quantity,
        event_sequence=event.sequence,
        replay=True,
    )
