"""Scenario eval harness.

Runs scripted multi-turn conversations from `scenarios.yaml` against a freshly
started mock API and asserts on (a) what the user saw, (b) which API calls were
made, and (c) the workflow outcome / handoff reason recorded in the trace.

    python3 -m evals.run_evals            # deterministic rule-based NLU (no key needed)
    python3 -m evals.run_evals --llm      # same scripts through the OpenAI NLU
    python3 -m evals.run_evals -k happy   # only scenarios whose id contains "happy"

Exit code is non-zero if any scenario fails.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from assistant.api_client import SchedulingClient  # noqa: E402
from assistant.nlu import build_nlu  # noqa: E402
from assistant.tracing import Tracer  # noqa: E402
from assistant.workflow import Assistant, State  # noqa: E402

Check = Callable[[str, Assistant], Optional[str]]  # returns an error message or None


@dataclass
class Turn:
    user: str
    checks: List[Check] = field(default_factory=list)
    set_mock_scenario: Optional[str] = None  # set/clear the outage header before this turn


@dataclass
class Scenario:
    id: str
    turns: List[Turn]
    final_checks: List[Check] = field(default_factory=list)


# ------------------------------------------------------------------ check builders


def says(*subs: str) -> Check:
    def check(reply: str, _a: Assistant) -> Optional[str]:
        missing = [s for s in subs if s.lower() not in reply.lower()]
        return f"reply missing {missing!r}" if missing else None
    return check


def says_not(*subs: str) -> Check:
    def check(reply: str, _a: Assistant) -> Optional[str]:
        leaked = [s for s in subs if s.lower() in reply.lower()]
        return f"reply leaked {leaked!r}" if leaked else None
    return check


def state_is(state: State) -> Check:
    def check(_r: str, a: Assistant) -> Optional[str]:
        return None if a.session.state == state else f"state is {a.session.state.value}, expected {state.value}"
    return check


def api_calls(method: str, path: str, count: int, status: Optional[int] = None) -> Check:
    def check(_r: str, a: Assistant) -> Optional[str]:
        calls = [e for e in a.tracer.find("api_call", method=method, path=path) if status is None or e.get("status") == status]
        return None if len(calls) == count else f"expected {count} {method} {path} (status={status}) calls, saw {len(calls)}"
    return check


def booked_once_with_confirmation() -> Check:
    def check(_r: str, a: Assistant) -> Optional[str]:
        posts = a.tracer.find("api_call", method="POST", path="/appointments")
        if len(posts) != 1:
            return f"expected exactly one POST /appointments, saw {len(posts)}"
        if posts[0]["body"].get("confirmed") is not True:
            return "POST /appointments sent without confirmed=true"
        if posts[0]["status"] != 201:
            return f"booking status was {posts[0]['status']}"
        return None
    return check


def handoff(reason: str) -> Check:
    def check(_r: str, a: Assistant) -> Optional[str]:
        evs = a.tracer.find("handoff_created", reason=reason)
        if not evs:
            failed = a.tracer.find("handoff_failed", reason=reason)
            if failed:
                return None  # handoff attempted with the right reason; the channel itself was down
            return f"no handoff with reason={reason}; handoffs seen: {[e.get('reason') for e in a.tracer.find('handoff_created')]}"
        return None
    return check


def no_handoff() -> Check:
    def check(_r: str, a: Assistant) -> Optional[str]:
        evs = a.tracer.find("handoff_created")
        return f"unexpected handoff(s): {[e.get('reason') for e in evs]}" if evs else None
    return check


def shows_only_api_slots() -> Check:
    """Every time shown to the user must come from the latest /availability response."""
    def check(reply: str, a: Assistant) -> Optional[str]:
        from assistant.workflow import _fmt_time
        shown = a.session.shown_slots
        if not shown:
            return "no slots recorded as shown"
        for sl in shown:
            if _fmt_time(sl["startTime"]) not in reply:
                return f"slot {sl['slotId']} from API not rendered"
        import re
        n_lines = len(re.findall(r"^\s*\d+\. ", reply, flags=re.M))
        return None if n_lines == len(shown) else f"rendered {n_lines} numbered slots but API returned {len(shown)}"
    return check


# ------------------------------------------------------------------ scenarios

SCENARIOS: List[Scenario] = [
    Scenario(
        "provider_lookup",
        turns=[
            Turn(
                "Which primary care providers are available downtown?",
                checks=[
                    says("Dr. Elena Brooks", "Dr. Marcus King"),
                    says_not("Dr. Priya Shah"),
                    api_calls("GET", "/providers", 1, 200),
                    api_calls("GET", "/patients/search", 0),
                    no_handoff(),
                ],
            ),
        ],
    ),
    Scenario(
        "happy_path_booking",
        turns=[
            Turn("I want to book a primary care appointment at the downtown clinic", checks=[says("phone", "date of birth"), state_is(State.IDENTIFY), api_calls("GET", "/patients/search", 0)]),
            Turn("My phone is 555-0101 and my DOB is 1985-04-12", checks=[says("Maya"), api_calls("GET", "/patients/search", 1, 200), api_calls("GET", "/availability", 1, 200), shows_only_api_slots(), state_is(State.CHOOSE_SLOT)]),
            Turn("1", checks=[says("yes", "Dr. Elena Brooks"), state_is(State.CONFIRM), api_calls("POST", "/appointments", 0)]),
            Turn("yes", checks=[says("booked", "Confirmation number: appt_"), booked_once_with_confirmation(), state_is(State.IDLE), no_handoff()]),
        ],
    ),
    Scenario(
        "decline_then_rebook",
        turns=[
            Turn("book an appointment please", checks=[state_is(State.IDENTIFY)]),
            Turn("555-0102, 1992-06-03", checks=[says("Jordan"), says("primary care or dermatology"), state_is(State.COLLECT_PREFS)]),
            Turn("primary care", checks=[says("downtown, uptown, or lakeside")]),
            Turn("any", checks=[state_is(State.CHOOSE_SLOT), shows_only_api_slots(), api_calls("GET", "/availability", 1, 200)]),
            Turn("the first one", checks=[state_is(State.CONFIRM), says("9:00 AM")]),
            Turn("no, a different time", checks=[says("nothing was booked"), state_is(State.CHOOSE_SLOT), api_calls("POST", "/appointments", 0)]),
            Turn("4", checks=[state_is(State.CONFIRM), says("uptown", "3:30 PM")]),
            Turn("yes", checks=[says("You're booked", "uptown"), booked_once_with_confirmation(), state_is(State.IDLE)]),
        ],
    ),
    Scenario(
        "lookup_existing_appointment",
        turns=[
            Turn("When is my appointment?", checks=[state_is(State.IDENTIFY), api_calls("GET", "/patients/search", 0)]),
            Turn("phone 555-0101, born 1985-04-12", checks=[says("Maya", "appt_3001", "Dr. Elena Brooks"), api_calls("GET", "/patients/pat_1001/appointments", 1, 200), state_is(State.IDLE), no_handoff()]),
        ],
    ),
    Scenario(
        "no_patient_match",
        turns=[
            Turn("Book an appointment", checks=[state_is(State.IDENTIFY)]),
            Turn("555-9999 and 1990-01-01", checks=[says("couldn't find"), says_not("pat_", "Maya", "Jordan", "Avery", "Sam"), api_calls("GET", "/patients/search", 1, 200), state_is(State.IDENTIFY), no_handoff()]),
            Turn("555-9999, 1990-01-01", checks=[says("handing this to our staff"), handoff("identity_unclear"), says_not("pat_", "Maya", "Jordan", "Avery", "Sam"), api_calls("GET", "/availability", 0), state_is(State.HANDED_OFF)]),
        ],
    ),
    Scenario(
        "multiple_patient_matches",
        turns=[
            Turn("Look up my appointment", checks=[state_is(State.IDENTIFY), api_calls("GET", "/patients/search", 0)]),
            Turn("555-0130 and 1978-09-22", checks=[says("more than one record", "zip code"), says_not("Avery", "Patel", "appt_", "pat_"), state_is(State.DISAMBIGUATE), api_calls("GET", "/patients/{patientId}/appointments", 0)]),
            Turn("70115", checks=[says("Avery", "don't see any appointments"), state_is(State.IDLE), no_handoff()]),
        ],
    ),
    Scenario(
        "multiple_matches_unresolved",
        turns=[
            Turn("I'd like to check my upcoming appointment"),
            Turn("555-0130 1978-09-22", checks=[state_is(State.DISAMBIGUATE)]),
            Turn("my zip is 99999", checks=[says("doesn't match"), handoff("identity_unclear"), says_not("Avery", "appt_")]),
        ],
    ),
    Scenario(
        "slot_conflict",
        turns=[
            Turn("Book the earliest downtown primary care slot"),
            Turn("555-0101 / 1985-04-12", checks=[state_is(State.CHOOSE_SLOT)]),
            Turn("the 8:30am one", checks=[state_is(State.CONFIRM), says("8:30 AM")]),
            Turn("yes", checks=[says("just taken", "nothing was booked"), says_not("You're booked"), api_calls("POST", "/appointments", 1, 409), api_calls("GET", "/availability", 2, 200), state_is(State.CHOOSE_SLOT), says_not("8:30 AM -")]),
            Turn("2", checks=[state_is(State.CONFIRM)]),
            Turn("confirm", checks=[says("You're booked"), api_calls("POST", "/appointments", 1, 201), state_is(State.IDLE)]),
        ],
    ),
    Scenario(
        "no_availability",
        turns=[
            Turn("Book a dermatology appointment at Lakeside"),
            Turn("555-0102 and 1992-06-03", checks=[says("no dermatology openings at lakeside", "offered at: uptown"), state_is(State.NO_AVAILABILITY), api_calls("GET", "/availability", 1, 200), says_not("slot_", "1.")]),
            Turn("yes please have staff find me a time", checks=[handoff("no_availability"), state_is(State.HANDED_OFF)]),
        ],
    ),
    Scenario(
        "api_failure",
        turns=[
            Turn("Book a primary care appointment"),
            Turn("555-0101, 1985-04-12", set_mock_scenario="api_failure", checks=[says("isn't responding", "Nothing has been booked"), handoff("api_failure"), says_not("Maya", "pat_"), api_calls("GET", "/patients/search", 2, 503)]),
        ],
    ),
    Scenario(
        "medical_advice",
        turns=[
            Turn("I have chest pain. Should I wait for an appointment?", checks=[says("not able to give medical advice", "911"), says_not("you should", "it's fine", "safe to wait."), handoff("medical_advice"), api_calls("GET", "/patients/search", 0)]),
        ],
    ),
    Scenario(
        "user_requests_human",
        turns=[
            Turn("I'd like to talk to a real person please", checks=[handoff("user_requested"), says("reference number")]),
        ],
    ),
    Scenario(
        "unsupported_specialty",
        turns=[
            Turn("Can I book a cardiology appointment?", checks=[handoff("unsupported_request"), says("primary care and dermatology")]),
        ],
    ),
]


# ------------------------------------------------------------------ runner


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "mock-api", "server.py"), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/providers", timeout=1).read()
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("mock API did not start")


def run_scenario(sc: Scenario, api_base: str, use_llm: bool, verbose: bool) -> List[str]:
    tracer = Tracer(path=None, echo=False, session_id=f"eval-{sc.id}"[:24])
    client = SchedulingClient(api_base, tracer, retry_backoff_s=0.05)
    assistant = Assistant(client, build_nlu(tracer, use_llm=use_llm), tracer)
    errors: List[str] = []
    for i, turn in enumerate(sc.turns, start=1):
        if turn.set_mock_scenario is not None:
            if turn.set_mock_scenario:
                client.extra_headers["X-Mock-Scenario"] = turn.set_mock_scenario
            else:
                client.extra_headers.pop("X-Mock-Scenario", None)
        result = assistant.handle(turn.user)
        if verbose:
            print(f"    you> {turn.user}")
            print("    assistant> " + result.reply.replace("\n", "\n               "))
        for check in turn.checks:
            err = check(result.reply, assistant)
            if err:
                errors.append(f"turn {i} ({turn.user[:40]!r}): {err}")
    for check in sc.final_checks:
        err = check("", assistant)
        if err:
            errors.append(f"final: {err}")
    summaries = tracer.find("turn_summary")
    total_ms = sum(e.get("latency_ms", 0) for e in summaries)
    if verbose:
        print(f"    (turns={len(summaries)} total_latency_ms={total_ms:.1f} api_calls={len(tracer.find('api_call'))})")
    if use_llm:
        failed = tracer.find("llm_call", ok=False)
        if failed:
            # Make it impossible to mistake a rules-fallback pass for an LLM pass.
            errors.append(f"LLM unavailable on {len(failed)}/{len(summaries)} turns ({failed[0].get('error')}); rules fallback was used")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Run through the OpenAI NLU (needs OPENAI_API_KEY).")
    parser.add_argument("--api-base", default=None, help="Use an already-running mock instead of starting one.")
    parser.add_argument("-k", default=None, help="Only run scenarios whose id contains this substring.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print the conversations.")
    args = parser.parse_args(argv)

    selected = [s for s in SCENARIOS if not args.k or args.k in s.id]
    failures = 0
    for sc in selected:
        print(f"[ {sc.id} ]")
        proc = None
        api_base = args.api_base
        if api_base is None:
            # Fresh mock per scenario so bookings in one scenario cannot affect another.
            port = free_port()
            proc = start_mock(port)
            api_base = f"http://127.0.0.1:{port}"
        try:
            errors = run_scenario(sc, api_base, args.llm, args.verbose)
        finally:
            if proc:
                proc.terminate()
                proc.wait(timeout=5)
        if errors:
            failures += 1
            for e in errors:
                print(f"    FAIL: {e}")
        else:
            print("    PASS")
    print(f"\n{len(selected) - failures}/{len(selected)} scenarios passed (nlu={'llm' if args.llm else 'rules'})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
