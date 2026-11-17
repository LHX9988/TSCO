"""Explicit, deterministic disturbances for frozen-policy DC-grid tests.

Events specify equipment indices and half-open physical-time windows.
Policy weights remain fixed. Generator outages return an availability mask
applied to the existing commitment schedule; this module leaves generator
cost curves unchanged and does not trigger replanning. Line maintenance
sets the selected branch rating to zero, representing an open circuit.

Every simultaneously active combination of line outages is checked before a
schedule is accepted. Connectivity checks establish only that maintenance is
non-islanding; they do not certify DC dispatch feasibility, AC feasibility, or
N-1 security. Those outcomes must be evaluated by the actual rollout.

Example::

    schedule = Schedule(case, [Disturbance(
        'generator_outage', index=1, start_hour=72, duration_hours=24)])
    changed_case, available, demand_multiplier = schedule.at(interval)
    actual_commitment = planned_commitment * available.astype(int)

Indices passed to the API are zero-based. A line can instead be identified by
its explicitly supplied one-based ``from_bus_id`` and ``to_bus_id``; the code
then resolves that pair in the supplied grid case. ``descriptions`` exports
one-based equipment and bus IDs. No missing event time or multiplier is filled.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math
import numpy as np

from .schema import GridCase


@dataclass(frozen=True)
class Disturbance:
    kind: str
    index: int | None = None
    start_hour: float | None = None
    duration_hours: float | None = None
    value: float | None = None
    from_bus_id: int | None = None
    to_bus_id: int | None = None


def _connected(case: GridCase, open_lines=()):
    """Check electrical connectivity after opening the supplied line indices."""
    if case.n_bus <= 1: return True
    active = np.asarray(case.branch_limit) > 0
    if len(open_lines): active[np.asarray(list(open_lines), int)] = False
    neighbors = [[] for _ in range(case.n_bus)]
    for f, t in zip(case.branch_from[active], case.branch_to[active]):
        neighbors[int(f)].append(int(t)); neighbors[int(t)].append(int(f))
    found, pending = {0}, [0]
    while pending:
        for node in neighbors[pending.pop()]:
            if node not in found:
                found.add(node); pending.append(node)
    return len(found) == case.n_bus


class Schedule:
    """A non-mutating disturbance schedule on the manuscript's five-minute grid.

    Overlapping demand multipliers multiply. Outage windows use ``start <=
    time < end``; equipment is restored automatically at the ending boundary.
    All event times are in hours relative to the start of the test rollout.
    """
    def __init__(self, case: GridCase, events=()):
        self._base = copy.deepcopy(case)
        normalized = []
        for item in events:
            event = Disturbance(**item) if isinstance(item, Mapping) else item
            if not isinstance(event, Disturbance):
                raise TypeError('Events must be Disturbance instances or mappings of their fields')
            if event.kind == 'line_outage' and event.index is None:
                pair = (event.from_bus_id, event.to_bus_id)
                if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                       or not 1 <= v <= self._base.n_bus for v in pair):
                    raise ValueError('Supply a line index or its one-based from_bus_id and to_bus_id')
                f, t = (int(v) - 1 for v in pair)
                matching = np.flatnonzero(((self._base.branch_from == f) & (self._base.branch_to == t)) |
                                         ((self._base.branch_from == t) & (self._base.branch_to == f)))
                if len(matching) != 1:
                    raise ValueError('The supplied bus pair must identify exactly one line; otherwise provide its index')
                event = replace(event, index=int(matching[0]))
            self._validate_event(event)
            normalized.append(event)
        self.events = tuple(sorted(normalized, key=lambda e: (e.start_hour, e.kind,
                                                              -1 if e.index is None else e.index)))
        self._windows = tuple((int(round(e.start_hour * 12)),
                               int(round((e.start_hour + e.duration_hours) * 12)))
                              for e in self.events)
        lines = [i for i, e in enumerate(self.events) if e.kind == 'line_outage']
        if lines:
            if not _connected(self._base):
                raise ValueError('Non-islanding maintenance requires a connected base grid')
            # Active sets only change at event boundaries. Checking each unique
            # boundary also catches combinations that individually are safe.
            boundaries = sorted({point for i in lines for point in self._windows[i]})
            for tick in boundaries:
                opened = {self.events[i].index for i in lines
                          if self._windows[i][0] <= tick < self._windows[i][1]}
                if not _connected(self._base, opened):
                    ids = ','.join(str(i + 1) for i in sorted(opened))
                    raise ValueError(f'Line outage combination [{ids}] is islanding at hour {tick / 12:g}')

    def _validate_event(self, event):
        allowed = {'generator_outage', 'line_outage', 'demand_scale'}
        if event.kind not in allowed:
            raise ValueError(f'Unknown disturbance kind {event.kind!r}; choose one of {sorted(allowed)}')
        for name in ('start_hour', 'duration_hours'):
            value = getattr(event, name)
            if not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite')
            if not math.isclose(float(value) * 12, round(float(value) * 12), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f'{name} must align to five-minute intervals')
        if event.start_hour < 0 or event.duration_hours <= 0:
            raise ValueError('start_hour must be nonnegative and duration_hours positive')
        if event.kind == 'demand_scale':
            if event.index is not None:
                raise ValueError('demand_scale is a system-wide multiplier and does not take an equipment index')
            if not isinstance(event.value, (int, float, np.number)) or not math.isfinite(float(event.value)) or event.value <= 0:
                raise ValueError('Demand multiplier must be positive and finite')
        else:
            count = self._base.n_gen if event.kind == 'generator_outage' else len(self._base.branch_from)
            if isinstance(event.index, (bool, np.bool_)) or not isinstance(event.index, (int, np.integer)) or not 0 <= event.index < count:
                raise ValueError(f'{event.kind} index must be an integer in [0,{count})')
            if event.kind == 'line_outage' and self._base.branch_limit[event.index] <= 0:
                raise ValueError('The selected line is already out of service in the base case')
            if event.kind == 'line_outage' and (event.from_bus_id is not None or event.to_bus_id is not None):
                actual = {int(self._base.branch_from[event.index]) + 1,
                          int(self._base.branch_to[event.index]) + 1}
                if {event.from_bus_id, event.to_bus_id} != actual:
                    raise ValueError('Line index and supplied bus IDs do not identify the same line')

    def at(self, interval, dt_seconds=300):
        """Return ``(independent case copy, generator mask, demand multiplier)``.

        This API deliberately rejects non-five-minute grids, so an event cannot
        silently occur inside a simulated physical interval.
        """
        if isinstance(interval, (bool, np.bool_)) or not isinstance(interval, (int, np.integer)) or interval < 0:
            raise ValueError('interval must be a nonnegative integer')
        if not np.isfinite(dt_seconds) or dt_seconds != 300:
            raise ValueError('Disturbance schedules use five-minute (300 s) intervals')
        changed = copy.deepcopy(self._base)
        available = np.ones(changed.n_gen, dtype=bool)
        multiplier = 1.0
        for event, (start, end) in zip(self.events, self._windows):
            if not start <= interval < end: continue
            if event.kind == 'generator_outage':
                available[event.index] = False
            elif event.kind == 'line_outage':
                changed.branch_limit[event.index] = 0.0
            else:
                multiplier *= float(event.value)
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError('Combined demand multipliers must remain positive and finite')
        return changed, available, multiplier

    def descriptions(self):
        """Return JSON-ready event definitions with one-based equipment/bus IDs."""
        records = []
        for event in self.events:
            record = {'kind': event.kind, 'start_hour': float(event.start_hour),
                      'duration_hours': float(event.duration_hours),
                      'end_hour_exclusive': float(event.start_hour + event.duration_hours)}
            if event.kind == 'generator_outage':
                record.update(equipment_id=int(event.index) + 1,
                              bus_id=int(self._base.gen_bus[event.index]) + 1)
            elif event.kind == 'line_outage':
                record.update(equipment_id=int(event.index) + 1,
                              from_bus=int(self._base.branch_from[event.index]) + 1,
                              to_bus=int(self._base.branch_to[event.index]) + 1)
            else:
                record['demand_multiplier'] = float(event.value)
            records.append(record)
        return records
