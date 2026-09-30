from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.timebox import TimeBox


def can_reserve(request_start: int, request_end: int, reservations: list[tuple[int, int]]) -> bool:
    requested = TimeBox(request_start, request_end)
    return not any(requested.overlaps(TimeBox(start, end)) for start, end in reservations)


@dataclass(frozen=True, slots=True)
class Reservation:
    reticle_id: str
    job_id: str
    window: TimeBox
    recipe_revision: int


class ReservationBook:
    def __init__(self) -> None:
        self._reservations: list[Reservation] = []
        self._requests: dict[str, tuple[Reservation, ...]] = {}

    def reserve_batch(self, request_id: str, requested: list[Reservation]) -> tuple[Reservation, ...]:
        frozen = tuple(requested)
        previous = self._requests.get(request_id)
        if previous is not None:
            if previous != frozen:
                raise ValueError("request id reused with different reservations")
            return previous
        if not requested:
            raise ValueError("empty reservation batch")
        candidate = [*self._reservations]
        for reservation in requested:
            if reservation.recipe_revision <= 0:
                raise ValueError("invalid recipe revision")
            conflicts = (
                existing
                for existing in candidate
                if existing.reticle_id == reservation.reticle_id and existing.job_id != reservation.job_id
            )
            if any(reservation.window.overlaps(existing.window) for existing in conflicts):
                raise ValueError("reticle reservation conflict")
            candidate.append(reservation)
        self._reservations = candidate
        self._requests[request_id] = frozen
        return frozen

    def reservations(self, reticle_id: str) -> tuple[Reservation, ...]:
        return tuple(item for item in self._reservations if item.reticle_id == reticle_id)

    def held_by(self, request_id: str) -> tuple[Reservation, ...]:
        return self._requests.get(request_id, ())
