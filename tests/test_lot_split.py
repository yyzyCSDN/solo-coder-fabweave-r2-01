from __future__ import annotations

import pytest

from fabweave.core.events import EventLog
from fabweave.core.timebox import TimeBox
from fabweave.dedication.tool import DedicationPolicy
from fabweave.lot.merge import Lot
from fabweave.lot.split import Allocation, MachineCapacity, plan_capacity_allocations, split_lot
from fabweave.recipe.qualification import Qualification
from fabweave.reticle.reservation import Reservation, ReservationBook
from fabweave.workflows.control import CapacityOffer, replay_split, split_for_idle_capacity


POLICIES = [DedicationPolicy(1, 0, {"P": frozenset({"T1", "T2"})})]
QUALIFICATIONS = [
    Qualification("T1", "C1", 2, 0, 1000),
    Qualification("T2", "C2", 2, 0, 1000),
]


def _offer(tool: str, chamber: str, available: int, reticle: str, window: TimeBox) -> CapacityOffer:
    return CapacityOffer(tool, chamber, available, reticle, window)


def test_plan_greedy_skips_undersized_and_keeps_floor_remainder():
    capacities = [MachineCapacity("T2", "C2", 6), MachineCapacity("T1", "C1", 3)]
    allocations = plan_capacity_allocations(10, capacities, minimum_quantity=5)
    # T1 idle capacity is below the floor (skipped); T2 takes its actual 6, residual 4 stays with parent
    assert [(a.tool_id, a.chamber_id, a.quantity) for a in allocations] == [("T2", "C2", 6)]


def test_split_lot_conserves_quantity_and_assigns_deterministic_children():
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    capacities = [MachineCapacity("T1", "C1", 10), MachineCapacity("T2", "C2", 12)]
    first = split_lot("REQ-1", lot, capacities, minimum_quantity=5)
    second = split_lot("REQ-1", lot, capacities, minimum_quantity=5)

    assert first == second
    assert first.parent_id == "LOT-A"
    assert first.remaining_quantity == 3
    assert [sublot.quantity for sublot in first.sublots] == [10, 12]
    assert first.remaining_quantity + sum(s.quantity for s in first.sublots) == lot.quantity
    assert all(sublot.parent_id == "LOT-A" for sublot in first.sublots)
    assert [sublot.sequence for sublot in first.sublots] == [1, 2]
    assert len({sublot.lot_id for sublot in first.sublots}) == 2


def test_split_lot_rejects_allocation_below_floor_and_held_parent():
    lot = Lot("LOT-A", "P", "R", "S", 2, 7)
    capacities = [MachineCapacity("T1", "C1", 7)]
    # the whole lot (and the idle capacity) being below the floor yields no legal sublot
    assert plan_capacity_allocations(7, capacities, minimum_quantity=8) == ()
    with pytest.raises(ValueError):
        split_lot("REQ", lot, capacities, minimum_quantity=8)
    held = Lot("LOT-H", "P", "R", "S", 2, 7, frozenset({"QA"}))
    with pytest.raises(ValueError):
        split_lot("REQ", held, capacities, minimum_quantity=5)


def test_split_lot_rejects_duplicate_unknown_or_oversized_allocations():
    lot = Lot("LOT-A", "P", "R", "S", 2, 20)
    with pytest.raises(ValueError):
        split_lot(
            "REQ",
            lot,
            [MachineCapacity("T1", "C1", 5)],
            minimum_quantity=5,
            allocations=(Allocation("T1", "C1", 5), Allocation("T1", "C1", 5)),
        )
    with pytest.raises(ValueError):
        split_lot(
            "REQ",
            lot,
            [MachineCapacity("T1", "C1", 5)],
            minimum_quantity=5,
            allocations=(Allocation("T9", "C9", 5),),
        )
    with pytest.raises(ValueError):
        split_lot(
            "REQ",
            lot,
            [MachineCapacity("T1", "C1", 5)],
            minimum_quantity=5,
            allocations=(Allocation("T1", "C1", 6),),
        )


def test_workflow_happy_path_is_atomic_and_replayable():
    log, book = EventLog(), ReservationBook()
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    offers = [
        _offer("T2", "C2", 12, "RET-2", TimeBox(100, 110)),
        _offer("T1", "C1", 10, "RET-1", TimeBox(100, 110)),
    ]
    decision = split_for_idle_capacity(log, book, "REQ-1", lot, offers, 5, 100, POLICIES, QUALIFICATIONS)

    assert decision.allowed
    result = decision.evidence["split"]
    assert [s.quantity for s in result.sublots] == [10, 12]
    assert result.remaining_quantity == 3
    assert decision.evidence["event_sequence"] == 1
    assert len(book.reservations("RET-1")) == 1 and len(book.reservations("RET-2")) == 1

    replay = split_for_idle_capacity(log, book, "REQ-1", lot, offers, 5, 200, POLICIES, QUALIFICATIONS)
    assert replay.allowed and replay.evidence["replay"] is True
    assert replay.evidence["split"] == result
    # idempotent reservations: replaying never books a second batch
    assert len(book.reservations("RET-1")) == 1 and len(book.reservations("RET-2")) == 1
    assert replay_split(log, "REQ-1") == result
    assert len(log.read("lot-split:REQ-1")) == 1


def test_workflow_gate_failure_leaves_no_reservation_and_no_event():
    log, book = EventLog(), ReservationBook()
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    offers = [
        _offer("T1", "C1", 10, "RET-1", TimeBox(100, 110)),
        _offer("T2", "C2", 12, "RET-2", TimeBox(100, 110)),
    ]
    # T2 is dropped from the dedication allow list -> the whole split fails closed
    policies = [DedicationPolicy(1, 0, {"P": frozenset({"T1"})})]
    decision = split_for_idle_capacity(log, book, "REQ-1", lot, offers, 5, 100, policies, QUALIFICATIONS)

    assert not decision.allowed
    assert any(reason.startswith("tool_not_in_dedication_allow_list@T2") for reason in decision.reasons)
    assert log.read("lot-split:REQ-1") == ()
    assert book.reservations("RET-1") == () and book.reservations("RET-2") == ()

    # missing qualification on T1 chamber
    decision = split_for_idle_capacity(
        log,
        ReservationBook(),
        "REQ-2",
        lot,
        [_offer("T1", "C9", 10, "RET-1", TimeBox(100, 110))],
        5,
        100,
        POLICIES,
        QUALIFICATIONS,
    )
    assert any(reason.startswith("qualification_missing@T1") for reason in decision.reasons)


def test_workflow_rejects_reticle_conflict_and_holds_without_commit():
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    offers = [_offer("T1", "C1", 10, "RET-1", TimeBox(100, 110))]

    held = Lot("LOT-A", "P", "R", "S", 2, 25, frozenset({"QA"}))
    decision = split_for_idle_capacity(
        EventLog(), ReservationBook(), "REQ-H", held, offers, 5, 100, POLICIES, QUALIFICATIONS
    )
    assert decision.reasons == ("held_lot_cannot_split",)

    book = ReservationBook()
    book.reserve_batch("OTHER", [Reservation("RET-1", "JOB-X", TimeBox(105, 115), 2)])
    log = EventLog()
    decision = split_for_idle_capacity(log, book, "REQ-C", lot, offers, 5, 100, POLICIES, QUALIFICATIONS)
    assert decision.reasons == ("reticle_busy:RET-1@T1:C1",)
    assert log.read("lot-split:REQ-C") == ()

    # one reticle shared by two overlapping sublots inside the same request is rejected
    overlapping = [
        _offer("T1", "C1", 10, "RET-1", TimeBox(100, 110)),
        _offer("T2", "C2", 12, "RET-1", TimeBox(105, 115)),
    ]
    decision = split_for_idle_capacity(
        EventLog(), ReservationBook(), "REQ-I", lot, overlapping, 5, 100, POLICIES, QUALIFICATIONS
    )
    assert any(reason.startswith("reticle_busy:RET-1") for reason in decision.reasons)


def test_workflow_no_capacity_meeting_floor_is_denied():
    log, book = EventLog(), ReservationBook()
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    offers = [_offer("T1", "C1", 4, "RET-1", TimeBox(100, 110))]
    decision = split_for_idle_capacity(log, book, "REQ-N", lot, offers, 5, 100, POLICIES, QUALIFICATIONS)
    assert decision.reasons == ("no_idle_capacity_meets_minimum_sublot",)
    assert log.read("lot-split:REQ-N") == () and book.reservations("RET-1") == ()


def test_replay_rejects_reused_request_with_changed_lot_or_reticle():
    log, book = EventLog(), ReservationBook()
    lot = Lot("LOT-A", "P", "R", "S", 2, 25)
    offers = [_offer("T1", "C1", 10, "RET-1", TimeBox(100, 110))]
    split_for_idle_capacity(log, book, "REQ-1", lot, offers, 5, 100, POLICIES, QUALIFICATIONS)

    different_lot = Lot("LOT-B", "P", "R", "S", 2, 25)
    with pytest.raises(ValueError):
        split_for_idle_capacity(log, book, "REQ-1", different_lot, offers, 5, 100, POLICIES, QUALIFICATIONS)

    changed_reticle = [_offer("T1", "C1", 10, "RET-9", TimeBox(100, 110))]
    with pytest.raises(ValueError):
        split_for_idle_capacity(log, book, "REQ-1", lot, changed_reticle, 5, 100, POLICIES, QUALIFICATIONS)
