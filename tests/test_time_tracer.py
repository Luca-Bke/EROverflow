import itertools
from unittest.mock import patch

import pytest

from agents.terminal_bench_supplementary.utils import TimeTracer

PERF_COUNTER = "agents.terminal_bench_supplementary.utils.time.perf_counter"


@pytest.fixture(autouse=True)
def reset_time_tracer():
    TimeTracer.timer_sessions = []
    TimeTracer.current_session = []
    TimeTracer.last_end_time = {}
    yield
    TimeTracer.timer_sessions = []
    TimeTracer.current_session = []
    TimeTracer.last_end_time = {}


def _clock(*values):
    """A perf_counter stand-in that yields `values` in order, then repeats the last
    forever — so any incidental extra call doesn't blow up the test with StopIteration."""
    return itertools.chain(values, itertools.repeat(values[-1]))


# ── timed: measures the wrapped call's own execution time ──────────────────

def test_timed_sync_records_elapsed_duration():
    @TimeTracer.timed("work")
    def work():
        return "done"

    with patch(PERF_COUNTER, side_effect=_clock(10.0, 10.25)):
        result = work()

    assert result == "done"
    assert TimeTracer.current_session == [{"[work]": pytest.approx(0.25)}]


async def test_timed_async_records_elapsed_duration():
    @TimeTracer.timed("async_work")
    async def work():
        return "done"

    with patch(PERF_COUNTER, side_effect=_clock(5.0, 5.4)):
        result = await work()

    assert result == "done"
    assert TimeTracer.current_session == [{"[async_work]": pytest.approx(0.4)}]


def test_timed_records_even_when_func_raises():
    @TimeTracer.timed("boom")
    def boom():
        raise ValueError("nope")

    with patch(PERF_COUNTER, side_effect=_clock(1.0, 1.1)):
        with pytest.raises(ValueError):
            boom()

    assert TimeTracer.current_session == [{"[boom]": pytest.approx(0.1)}]


# ── inverse_timed: measures the idle gap between calls, not call duration ──

def test_inverse_timed_first_call_records_nothing():
    @TimeTracer.inverse_timed("gap")
    def step():
        return "ok"

    with patch(PERF_COUNTER, side_effect=_clock(100.0, 100.05)):
        step()

    # No prior call to measure a gap against, so nothing is recorded yet —
    # only the end time is stashed for the *next* call to measure against.
    assert TimeTracer.current_session == []
    assert TimeTracer.last_end_time["gap"] == pytest.approx(100.05)


def test_inverse_timed_measures_gap_between_calls_not_call_duration():
    @TimeTracer.inverse_timed("gap")
    def step():
        return "ok"

    # call 1: start=0.0 -> end=0.1            (first call: nothing recorded)
    # ... 2.0s of idle time passes ...
    # call 2: start=2.1 -> end=7.1             (this call itself takes 5s to
    #                                           run, but inverse_timed should
    #                                           record the 2.0s *gap* before
    #                                           it started, not its 5s runtime)
    with patch(PERF_COUNTER, side_effect=_clock(0.0, 0.1, 2.1, 7.1)):
        step()
        step()

    assert TimeTracer.current_session == [{"[gap]": pytest.approx(2.0)}]


async def test_inverse_timed_async_measures_gap_between_calls():
    @TimeTracer.inverse_timed("async_gap")
    async def step():
        return "ok"

    with patch(PERF_COUNTER, side_effect=_clock(0.0, 0.1, 3.6, 3.7)):
        await step()
        await step()

    assert TimeTracer.current_session == [{"[async_gap]": pytest.approx(3.5)}]


def test_inverse_timed_tracks_each_entry_name_independently():
    @TimeTracer.inverse_timed("first")
    def first():
        return "ok"

    @TimeTracer.inverse_timed("second")
    def second():
        return "ok"

    with patch(PERF_COUNTER, side_effect=_clock(0.0, 0.0, 1.0, 1.0, 4.0, 4.0)):
        first()   # 0.0 -> 0.0, nothing recorded yet for "first"
        second()  # 1.0 -> 1.0, nothing recorded yet for "second"
        first()   # 4.0 -> 4.0, gap since "first" last ended: 4.0 - 0.0 = 4.0

    assert TimeTracer.current_session == [{"[first]": pytest.approx(4.0)}]


# ── new_session(): rolls the current session into history and resets it ───

def test_new_session_rolls_current_into_history_and_resets():
    @TimeTracer.timed("a")
    def a():
        pass

    with patch(PERF_COUNTER, side_effect=_clock(0.0, 0.2)):
        a()

    TimeTracer.new_session()
    assert TimeTracer.timer_sessions == [[{"[a]": pytest.approx(0.2)}]]
    assert TimeTracer.current_session == []

    with patch(PERF_COUNTER, side_effect=_clock(1.0, 1.3)):
        a()

    TimeTracer.new_session()
    assert TimeTracer.timer_sessions == [
        [{"[a]": pytest.approx(0.2)}],
        [{"[a]": pytest.approx(0.3)}],
    ]
