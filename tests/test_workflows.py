from __future__ import annotations

import pytest

from fabweave.chamber.matching import Tolerance, mismatch_reasons
from fabweave.chemicals.shelf_life import effective_age, shelf_life_remaining
from fabweave.cleanroom.excursion import active_excursion, excursion_periods
from fabweave.contamination.segregation import incompatible_pairs, sequence_compatible
from fabweave.core.decision import allow, deny
from fabweave.core.events import EventLog
from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.core.timebox import TimeBox, covered_duration, merge_timeboxes
from fabweave.core.units import fraction_to_ppm, ppm_to_fraction
from fabweave.core.versioning import RevisionHistory
from fabweave.dedication.tool import DedicationPolicy, policy_decision
from fabweave.dispatch.priority import DispatchCandidate, rank_candidates
from fabweave.energy.demand import coincident_peak, demand_violations
from fabweave.genealogy.carrier import GenealogyGraph
from fabweave.holds.release import HoldState, release_decision
from fabweave.lot.merge import Lot, merge_lots
from fabweave.lot.split import MachineCapacity, SplitLedger, plan_split
from fabweave.maintenance.window import MaintenancePolicy, assignment_decision
from fabweave.metrology.sampling import Wafer, deterministic_sample
from fabweave.queue_time.enforcer import QueueLimit, queue_time_decision
from fabweave.recipe.qualification import Qualification, qualification_decision
from fabweave.reticle.reservation import Reservation, ReservationBook
from fabweave.rework.authorization import ReworkPolicy, authorization_decision
from fabweave.shipment.commit import CommitLedger
from fabweave.spc.control import western_electric_rules
from fabweave.takt.monitor import takt_result
from fabweave.yielding.rollup import StepYield, route_yield
from fabweave.workflows.control import (
    chamber_sample_decision,
    containment_scope,
    dedicated_qualification,
    environmental_containment,
    evaluate_sample,
    material_readiness,
    merge_and_commit,
    performance_summary,
    powered_assignment,
    qualified_assignment,
    rank_releasable,
    record_containment,
    reserve_qualified,
    rework_release_decision,
    split_for_idle_capacity,
)


def test_timeboxes_are_half_open_and_merge_without_double_counting():
    assert not TimeBox(0, 10).overlaps(TimeBox(10, 20))
    assert merge_timeboxes([TimeBox(5, 15), TimeBox(0, 10), TimeBox(20, 25)]) == [TimeBox(0, 15), TimeBox(20, 25)]
    assert covered_duration([TimeBox(-5, 5), TimeBox(3, 9), TimeBox(20, 30)], TimeBox(0, 10)) == 9


def test_event_log_is_idempotent_and_does_not_leak_mutable_data():
    log = EventLog()
    payload = {"items": [1]}
    first = log.append("LOT-1", 0, "EVT-1", 10, "opened", payload)
    payload["items"].append(2)
    replay = log.append("LOT-1", 0, "EVT-1", 10, "opened", {"items": [1]})
    assert first == replay and log.read("LOT-1")[0].data == {"items": [1]}
    with pytest.raises(ValueError):
        log.append("LOT-1", 0, "EVT-1", 10, "opened", {"items": [9]})


def test_event_log_rejects_stale_version_and_backdated_append():
    log = EventLog()
    log.append("LOT-1", 0, "EVT-1", 10, "opened", {})
    with pytest.raises(ValueError):
        log.append("LOT-1", 0, "EVT-2", 11, "moved", {})
    with pytest.raises(ValueError):
        log.append("LOT-1", 1, "EVT-3", 9, "moved", {})


def test_revision_history_uses_effective_time_and_isolated_values():
    history: RevisionHistory[dict[str, int]] = RevisionHistory()
    source = {"limit": 2}
    history.publish(1, 10, source)
    source["limit"] = 99
    history.publish(2, 20, {"limit": 3})
    assert history.at(15).value == {"limit": 2}
    assert history.latest().version == 2


def test_decision_requires_explanation_only_for_rejections():
    assert allow(version=1).allowed
    assert deny("blocked", version=1).reasons == ("blocked",)
    with pytest.raises(ValueError):
        deny()


def test_quantity_conservation_and_deterministic_ids():
    require_conservation({"A": 10, "B": 15}, {"C": 25})
    assert deterministic_id("LOT", "request", 1) == deterministic_id("LOT", "request", 1)
    with pytest.raises(ValueError):
        require_conservation({"A": 10}, {"B": 9})


def test_unit_conversions_validate_ranges_and_round_trip():
    assert fraction_to_ppm(ppm_to_fraction(12.5)) == pytest.approx(12.5)
    with pytest.raises(ValueError):
        ppm_to_fraction(-1)
    with pytest.raises(ValueError):
        fraction_to_ppm(1.01)


def test_queue_time_clips_and_merges_authorized_pauses():
    decision = queue_time_decision(
        QueueLimit("ETCH", 80, 3),
        100,
        200,
        [TimeBox(80, 110), TimeBox(105, 130), TimeBox(190, 220)],
    )
    assert decision.allowed
    assert decision.evidence["paused_minutes"] == 40
    assert decision.evidence["remaining_minutes"] == 20


def test_dispatch_excludes_held_and_frozen_lots_and_breaks_ties_stably():
    candidates = [
        DispatchCandidate("LOT-B", 1, 2),
        DispatchCandidate("LOT-A", 1, 2),
        DispatchCandidate("LOT-H", 0.1, 1, active_hold=True),
        DispatchCandidate("LOT-F", 0.1, 1, frozen_until=20),
    ]
    assert [item.lot_id for item in rank_candidates(candidates, 10)] == ["LOT-A", "LOT-B"]


def test_reticle_batch_reservation_is_atomic_and_idempotent():
    book = ReservationBook()
    initial = Reservation("RET-1", "JOB-1", TimeBox(0, 10), 1)
    assert book.reserve_batch("REQ-1", [initial]) == (initial,)
    assert book.reserve_batch("REQ-1", [initial]) == (initial,)
    with pytest.raises(ValueError):
        book.reserve_batch(
            "REQ-2",
            [Reservation("RET-2", "JOB-2", TimeBox(0, 10), 1), Reservation("RET-1", "JOB-3", TimeBox(5, 15), 1)],
        )
    assert book.reservations("RET-2") == ()


def test_qualification_selection_honors_chamber_revision_expiry_and_revocation():
    qualifications = [
        Qualification("TOOL-1", "CH-1", 2, 0, 100),
        Qualification("TOOL-1", "CH-2", 2, 0, 100),
        Qualification("TOOL-1", "CH-1", 3, 0, 100, revoked_at=40),
    ]
    assert qualification_decision("TOOL-1", "CH-1", 2, 50, qualifications).allowed
    assert not qualification_decision("TOOL-1", "CH-1", 3, 50, qualifications).allowed


@pytest.mark.parametrize(
    "values,rule",
    [([0, 0, 3.1], "one_beyond_3_sigma"), ([2.1, 2.2, 0], "two_of_three_beyond_2_sigma_same_side"), ([1.1, 1.2, 0, 1.3, 1.4], "four_of_five_beyond_1_sigma_same_side")],
)
def test_spc_rules(values, rule):
    assert rule in western_electric_rules(values, 0, 1)


def test_metrology_sample_covers_required_zones_and_is_deterministic():
    wafers = [Wafer(f"W{i}", i, "EDGE" if i in {0, 5} else "CENTER") for i in range(6)]
    first = deterministic_sample(wafers, 3, {"EDGE", "CENTER"})
    assert first == deterministic_sample(list(reversed(wafers)), 3, {"EDGE", "CENTER"})
    assert {wafer.zone for wafer in first} == {"EDGE", "CENTER"}


def test_maintenance_assignment_combines_both_clocks_and_planned_window():
    policy = MaintenancePolicy(2, run_interval=100, calendar_interval=50)
    assert "calendar_maintenance_due" in assignment_decision(policy, 0, 10, 60, TimeBox(70, 80), []).reasons
    assert "planned_maintenance_overlap" in assignment_decision(policy, 50, 10, 60, TimeBox(70, 80), [TimeBox(75, 90)]).reasons


def test_chamber_matching_rejects_missing_non_finite_and_outside_parameters():
    tolerances = {"pressure": Tolerance(absolute=1), "flow": Tolerance(relative=0.1)}
    reasons = mismatch_reasons({"pressure": 10, "flow": 100}, {"pressure": 12}, tolerances)
    assert reasons == ("flow:missing", "pressure:outside_tolerance")


def test_hold_release_rejects_stale_version_missing_disposition_and_blockers():
    state = HoldState("H-1", 3, frozenset({"QA"}), frozenset(), frozenset({"CUSTOMER"}))
    assert release_decision(state, 2, 100).reasons == ("stale_hold_version",)
    reasons = release_decision(state, 3, 100).reasons
    assert reasons == ("missing_disposition:QA", "active_blocker:CUSTOMER")


def test_genealogy_traverses_both_directions_without_duplicates():
    graph = GenealogyGraph({"B": "A", "C": "A", "D": "B", "A": None})
    assert graph.related("D", "upstream", 5) == ("A", "B")
    assert graph.related("A", "downstream", 5) == ("B", "C", "D")
    with pytest.raises(ValueError):
        GenealogyGraph({"A": "B", "B": "A"})


def test_chemical_age_merges_and_clips_freeze_periods():
    periods = [TimeBox(-5, 5), TimeBox(3, 8), TimeBox(20, 30)]
    assert effective_age(0, 10, periods) == 2
    assert shelf_life_remaining(0, 5, 10, periods) == 3


def test_cleanroom_event_time_dedup_and_hysteresis():
    assert excursion_periods([(2, 6), (1, 5), (2, 6), (3, 1)], 4, 2) == [(1, 3)]
    assert active_excursion([(1, 6), (2, 6), (3, 3), (4, 6)], 5, 4, 2, 2)
    with pytest.raises(ValueError):
        excursion_periods([(1, 5), (1, 6)], 4, 1)


def test_coincident_demand_groups_timestamps_and_uses_half_open_window():
    loads = [(0, 2), (5, 3), (5, 4), (10, 7)]
    assert coincident_peak(loads, 10) == 14
    assert demand_violations(loads, [(5, 10), (10, 13)]) == (10,)


def test_rework_authorization_combines_budget_defect_and_engineer_gate():
    policy = ReworkPolicy(4, {"ETCH": 2}, frozenset({"D1", "D2"}), frozenset({"D2"}))
    assert authorization_decision(policy, "ETCH", 1, "D1", None).allowed
    assert authorization_decision(policy, "ETCH", 2, "D2", None).reasons == (
        "rework_limit_exhausted",
        "engineer_approval_missing",
    )


def test_lot_merge_preserves_quantity_and_rejects_held_input():
    lots = [Lot("LOT-A", "P", "R", "S", 1, 10), Lot("LOT-B", "P", "R", "S", 1, 15)]
    merged = merge_lots("REQ-1", lots)
    assert merged.quantity == 25
    with pytest.raises(ValueError):
        merge_lots("REQ-2", [lots[0], Lot("LOT-C", "P", "R", "S", 1, 2, frozenset({"QA"}))])


def test_split_plan_sizes_children_by_idle_capacity_and_conserves_quantity():
    lot = Lot("LOT-P", "P", "R", "S", 2, 20)
    plan = plan_split("REQ-1", lot, 5, [MachineCapacity("TOOL-B", "CH-1", 7), MachineCapacity("TOOL-A", "CH-1", 8)])
    assert [(child.tool_id, child.quantity) for child in plan.children] == [("TOOL-A", 8), ("TOOL-B", 7)]
    assert plan.residual_quantity == 5
    assert sum(plan.after_quantities.values()) == 20
    assert plan.parent_of == {child.lot_id: "LOT-P" for child in plan.children}
    again = plan_split("REQ-1", lot, 5, [MachineCapacity("TOOL-A", "CH-1", 8), MachineCapacity("TOOL-B", "CH-1", 7)])
    assert again == plan
    (child,) = plan.wip_lots()[:1]
    assert (child.product, child.route, child.step, child.recipe_revision, child.quantity) == ("P", "R", "S", 2, 8)


def test_dedication_uses_effective_revision_and_deny_overrides_allow():
    policies = [
        DedicationPolicy(1, 0, {"P": frozenset({"T1", "T2"})}),
        DedicationPolicy(2, 10, {"P": frozenset({"T1", "T2"})}, frozenset({("P", "T2")})),
    ]
    assert policy_decision("P", "T2", 5, policies).allowed
    assert policy_decision("P", "T2", 15, policies).reasons == ("product_tool_explicitly_blocked",)


def test_contamination_matrix_is_symmetric_and_cleaning_changes_residuals():
    matrix = {"CU": {"FE"}}
    assert incompatible_pairs({"CU", "FE"}, matrix) == (("CU", "FE"),)
    assert sequence_compatible({"CU"}, {"FE"}, matrix, cleaning_removes={"CU"})


def test_takt_uses_intervals_and_reports_duplicate_completions():
    result = takt_result([0, 10, 10, 20], 0, 20)
    assert result.takt == 10 and result.interval_count == 2 and result.duplicate_count == 1


def test_route_yield_is_ordered_multiplicative_and_complete():
    steps = [StepYield("B", 100, 80), StepYield("A", 100, 90)]
    assert route_yield(steps, ["A", "B"]) == pytest.approx(0.72)
    with pytest.raises(ValueError):
        route_yield(steps[:1], ["A", "B"])


def test_commit_ledger_is_atomic_idempotent_and_excludes_unavailable_units():
    ledger = CommitLedger(100, safety_units=5)
    ledger.mark_unavailable(pending_scrap=10, quarantine=5)
    first = ledger.allocate("REQ-1", "ORDER-1", 50)
    assert ledger.allocate("REQ-1", "ORDER-1", 50) == first
    assert ledger.available == 30
    with pytest.raises(ValueError):
        ledger.allocate("REQ-2", "ORDER-2", 31)
    assert ledger.available == 30


def test_material_readiness_combines_queue_and_chemical_clocks():
    decision = material_readiness(QueueLimit("ETCH", 80, 2), 100, 200, [TimeBox(110, 130)], 100, 90, [TimeBox(110, 130)])
    assert decision.allowed and decision.evidence["remaining_minutes"] == 0 and decision.evidence["remaining_shelf_hours"] == 10


def test_rank_releasable_applies_hold_decisions_before_dispatch():
    candidates = [DispatchCandidate("A", 1, 2), DispatchCandidate("B", 2, 2)]
    holds = {"A": HoldState("H", 1, frozenset({"QA"}))}
    assert [item.lot_id for item in rank_releasable(candidates, 10, holds, {"A": 1})] == ["B"]


def test_reserve_qualified_is_fail_closed_and_then_atomic():
    book = ReservationBook()
    reservation = Reservation("R", "J", TimeBox(10, 20), 3)
    qualifications = [Qualification("T", "C", 3, 0, 30)]
    assert reserve_qualified(book, "REQ", [reservation], {"J": ("T", "C")}, qualifications, 10).allowed
    assert len(book.reservations("R")) == 1


def test_qualified_assignment_combines_maintenance_and_recipe_state():
    decision = qualified_assignment(MaintenancePolicy(1, 100, 100), 0, 1, 10, TimeBox(20, 30), [], "T", "C", 2, [Qualification("T", "C", 2, 0, 40)])
    assert decision.allowed and decision.evidence["recipe_revision"] == 2


def test_sample_evaluation_keeps_selected_membership_with_spc_rules():
    wafers = [Wafer(f"W{i}", i, "EDGE" if i in {0, 4} else "CENTER") for i in range(5)]
    result = evaluate_sample(wafers, 3, {"EDGE", "CENTER"}, {f"W{i}": value for i, value in enumerate([0, 0, 0, 0, 4])}, 0, 1)
    assert len(result.wafer_ids) == 3 and "one_beyond_3_sigma" in result.rules


def test_chamber_sample_reports_parameter_and_coverage_failures_together():
    decision = chamber_sample_decision({"p": 10}, {"p": 12}, {"p": Tolerance(absolute=1)}, [Wafer("W", 0, "CENTER")], 1, {"EDGE"})
    assert decision.reasons[0] == "p:outside_tolerance" and decision.reasons[1].startswith("sample:")


def test_environmental_containment_combines_excursion_and_material_risk():
    decision = environmental_containment([(1, 6), (2, 6)], 5, 4, 2, 2, {"CU", "FE"}, {"CU": {"FE"}})
    assert decision.reasons == ("cleanroom_excursion", "incompatible:CU:FE")


def test_containment_scope_is_bounded_bidirectional_and_unique():
    graph = GenealogyGraph({"A": None, "B": "A", "C": "B", "D": "A"})
    assert containment_scope(graph, ["B"], 1) == ("A", "B", "C")


def test_containment_record_is_idempotent_and_binds_the_planned_scope():
    graph = GenealogyGraph({"A": None, "B": "A", "C": "B"})
    log = EventLog()
    first = record_containment(log, "RECALL-1", graph, ["B"], 1, 10)
    replay = record_containment(log, "RECALL-1", graph, ["B"], 1, 10)
    assert first == replay and first.scope == ("A", "B", "C") and first.event_sequence == 1


def test_powered_assignment_combines_peak_and_maintenance_reasons():
    decision = powered_assignment([(0, 7), (1, 7)], [(5, 10)], MaintenancePolicy(1, 100, 5), 0, 1, 10, TimeBox(10, 20), [])
    assert decision.reasons == ("demand_limit_exceeded:5", "calendar_maintenance_due")


def test_rework_release_requires_both_authorization_and_hold_completion():
    policy = ReworkPolicy(1, {"ETCH": 2}, frozenset({"D"}))
    hold = HoldState("H", 1, frozenset({"QA"}), frozenset())
    assert rework_release_decision(policy, "ETCH", 0, "D", None, hold, 1, 10).reasons == ("missing_disposition:QA",)


def test_merge_and_commit_preserves_quantity_across_services():
    lots = [Lot("A", "P", "R", "S", 1, 4), Lot("B", "P", "R", "S", 1, 6)]
    merged, allocation = merge_and_commit("REQ", lots, CommitLedger(20), "ORDER")
    assert merged.quantity == allocation.units == 10


def test_dedicated_qualification_combines_both_policy_gates():
    decision = dedicated_qualification("P", "T", "C", 2, 10, [DedicationPolicy(1, 0, {"P": frozenset({"T"})})], [Qualification("T", "C", 2, 0, 20)])
    assert decision.allowed and decision.evidence["policy_revision"] == 1


def test_performance_summary_uses_filtered_takt_and_ordered_route_yield():
    summary = performance_summary([0, 10, 20], 5, 20, [StepYield("A", 10, 9), StepYield("B", 9, 8)], ["A", "B"])
    assert summary.takt.takt == 10 and summary.route_yield == pytest.approx(0.8)


def test_split_plan_rebalances_residual_and_child_below_minimum():
    lot = Lot("LOT-P", "P", "R", "S", 2, 20)
    plan = plan_split("REQ-1", lot, 5, [MachineCapacity("TOOL-A", "CH-1", 8), MachineCapacity("TOOL-B", "CH-1", 8)])
    assert [child.quantity for child in plan.children] == [8, 7]
    assert plan.residual_quantity == 5
    consumed = plan_split("REQ-2", Lot("LOT-P", "P", "R", "S", 2, 9), 5, [MachineCapacity("TOOL-A", "CH-1", 9)])
    assert [child.quantity for child in consumed.children] == [9] and consumed.residual_quantity == 0
    with pytest.raises(ValueError):
        plan_split("REQ-3", Lot("LOT-P", "P", "R", "S", 2, 9), 5, [MachineCapacity("TOOL-A", "CH-1", 8)])


def test_split_plan_rejects_held_lot_bad_minimum_and_duplicate_machines():
    held = Lot("LOT-P", "P", "R", "S", 2, 20, frozenset({"QA"}))
    with pytest.raises(ValueError):
        plan_split("REQ-1", held, 5, [MachineCapacity("TOOL-A", "CH-1", 8)])
    with pytest.raises(ValueError):
        plan_split("REQ-1", Lot("LOT-P", "P", "R", "S", 2, 20), 0, [MachineCapacity("TOOL-A", "CH-1", 8)])
    with pytest.raises(ValueError):
        plan_split("REQ-1", Lot("LOT-P", "P", "R", "S", 2, 20), 5, [MachineCapacity("TOOL-A", "CH-1", 5), MachineCapacity("TOOL-A", "CH-1", 5)])


def test_split_ledger_replays_and_rejects_conflicting_reuse():
    ledger = SplitLedger()
    plan = plan_split("REQ-1", Lot("LOT-P", "P", "R", "S", 2, 20), 5, [MachineCapacity("TOOL-A", "CH-1", 8)])
    assert ledger.register(plan) is plan
    assert ledger.register(plan) is plan
    conflict = plan_split("REQ-1", Lot("LOT-P", "P", "R", "S", 2, 21), 5, [MachineCapacity("TOOL-A", "CH-1", 8)])
    with pytest.raises(ValueError):
        ledger.register(conflict)
    assert ledger.plan_for("REQ-1") == plan
    assert [child.parent_id for child in ledger.children_of("LOT-P")] == ["LOT-P"]
    assert ledger.parent_of() == plan.parent_of


def test_split_for_idle_capacity_excludes_unmatched_machines_and_commits_atomically():
    book, ledger, log = ReservationBook(), SplitLedger(), EventLog()
    lot = Lot("LOT-P", "P", "R", "S", 2, 20)
    machines = [MachineCapacity("TOOL-A", "CH-1", 8), MachineCapacity("TOOL-B", "CH-1", 7), MachineCapacity("TOOL-C", "CH-1", 9)]
    qualifications = [Qualification("TOOL-A", "CH-1", 2, 0, 100), Qualification("TOOL-B", "CH-1", 2, 0, 100)]
    outcome = split_for_idle_capacity(book, ledger, log, "REQ-1", lot, 5, machines, qualifications, ["RET-1", "RET-2"], TimeBox(10, 20), 50)
    assert outcome.decision.allowed
    assert [(child.tool_id, child.quantity) for child in outcome.plan.children] == [("TOOL-A", 8), ("TOOL-B", 7)]
    assert outcome.plan.residual_quantity == 5
    assert outcome.decision.evidence["rejected_machines"] == {"TOOL-C/CH-1": ("qualification_missing",)}
    assert [reservation.reticle_id for reservation in outcome.reservations] == ["RET-1", "RET-2"]
    assert outcome.event_sequence == 1
    replay = split_for_idle_capacity(book, ledger, log, "REQ-1", lot, 5, machines, qualifications, ["RET-1", "RET-2"], TimeBox(10, 20), 50)
    assert (replay.plan, replay.reservations, replay.event_sequence) == (outcome.plan, outcome.reservations, outcome.event_sequence)
    assert len(log.read("split:REQ-1")) == 1
    graph = GenealogyGraph({**ledger.parent_of(), "LOT-P": None})
    assert set(graph.related("LOT-P", "downstream", 1)) == {child.lot_id for child in outcome.plan.children}
    event = log.read("split:REQ-1")[0]
    assert event.data["residual_quantity"] == 5 and len(event.data["children"]) == 2


def test_split_for_idle_capacity_denies_without_partial_state():
    book, ledger, log = ReservationBook(), SplitLedger(), EventLog()
    lot = Lot("LOT-P", "P", "R", "S", 2, 20)
    machines = [MachineCapacity("TOOL-A", "CH-1", 8)]
    outcome = split_for_idle_capacity(book, ledger, log, "REQ-1", lot, 5, machines, [], ["RET-1"], TimeBox(10, 20), 50)
    assert not outcome.decision.allowed and outcome.decision.reasons == ("qualification_missing",)
    book.reserve_batch("REQ-OTHER", [Reservation("RET-1", "JOB-X", TimeBox(0, 30), 2)])
    qualified = [Qualification("TOOL-A", "CH-1", 2, 0, 100)]
    blocked = split_for_idle_capacity(book, ledger, log, "REQ-2", lot, 5, machines, qualified, ["RET-1"], TimeBox(10, 20), 50)
    assert not blocked.decision.allowed and blocked.decision.reasons == ("reticle_unavailable_in_window",)
    assert ledger.plan_for("REQ-1") is None and ledger.plan_for("REQ-2") is None
    assert log.read("split:REQ-1") == () and log.read("split:REQ-2") == ()
    assert len(book.reservations("RET-1")) == 1


def test_split_for_idle_capacity_caps_parallelism_by_available_reticles():
    book, ledger, log = ReservationBook(), SplitLedger(), EventLog()
    lot = Lot("LOT-P", "P", "R", "S", 2, 24)
    machines = [MachineCapacity("TOOL-A", "CH-1", 8), MachineCapacity("TOOL-B", "CH-1", 8), MachineCapacity("TOOL-C", "CH-1", 8)]
    qualifications = [Qualification(tool, "CH-1", 2, 0, 100) for tool in ("TOOL-A", "TOOL-B", "TOOL-C")]
    outcome = split_for_idle_capacity(book, ledger, log, "REQ-1", lot, 5, machines, qualifications, ["RET-1"], TimeBox(10, 20), 50)
    assert outcome.decision.allowed
    assert [child.tool_id for child in outcome.plan.children] == ["TOOL-A"]
    assert outcome.plan.residual_quantity == 16


def test_split_for_idle_capacity_raises_on_infeasible_quantities_without_partial_state():
    book, ledger, log = ReservationBook(), SplitLedger(), EventLog()
    qualified = [Qualification("TOOL-A", "CH-1", 2, 0, 100)]
    with pytest.raises(ValueError):
        split_for_idle_capacity(book, ledger, log, "REQ-1", Lot("LOT-P", "P", "R", "S", 2, 9), 5, [MachineCapacity("TOOL-A", "CH-1", 8)], qualified, ["RET-1"], TimeBox(10, 20), 50)
    assert ledger.plan_for("REQ-1") is None and book.reservations("RET-1") == () and log.read("split:REQ-1") == ()


def test_split_for_idle_capacity_conflicting_replay_keeps_original_commit():
    book, ledger, log = ReservationBook(), SplitLedger(), EventLog()
    machines = [MachineCapacity("TOOL-A", "CH-1", 8)]
    qualified = [Qualification("TOOL-A", "CH-1", 2, 0, 100)]
    first = split_for_idle_capacity(book, ledger, log, "REQ-1", Lot("LOT-P", "P", "R", "S", 2, 20), 5, machines, qualified, ["RET-1"], TimeBox(10, 20), 50)
    with pytest.raises(ValueError):
        split_for_idle_capacity(book, ledger, log, "REQ-1", Lot("LOT-P", "P", "R", "S", 2, 21), 5, machines, qualified, ["RET-1"], TimeBox(10, 20), 50)
    assert ledger.plan_for("REQ-1") == first.plan
    assert len(book.reservations("RET-1")) == 1 and len(log.read("split:REQ-1")) == 1
