"""Interactive command-line front end.

    python3 -m assistant.cli                 # LLM NLU if OPENAI_API_KEY is set, else rules
    python3 -m assistant.cli --no-llm        # force the deterministic rule-based NLU
    python3 -m assistant.cli --verbose       # echo JSON traces to stderr
    python3 -m assistant.cli --mock-scenario api_failure   # force the mock's outage path

Slash commands inside the session:
    /reset                 start a new conversation
    /scenario api_failure  send X-Mock-Scenario: api_failure on every request (demo the outage)
    /scenario off          stop sending the header
    /state                 print the workflow state (debug)
    /quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .api_client import SchedulingClient
from .nlu import build_nlu
from .tracing import Tracer
from .workflow import Assistant


def build_assistant(args: argparse.Namespace) -> Assistant:
    trace_path = args.trace_file
    if trace_path is None:
        trace_path = os.path.join("traces", time.strftime("session-%Y%m%d-%H%M%S.jsonl"))
    tracer = Tracer(path=None if trace_path == "-" else trace_path, echo=args.verbose)
    headers = {"X-Mock-Scenario": args.mock_scenario} if args.mock_scenario else None
    client = SchedulingClient(args.api_base, tracer, extra_headers=headers)
    use_llm = not args.no_llm
    nlu = build_nlu(tracer, use_llm=use_llm)
    return Assistant(client, nlu, tracer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Text-based appointment scheduling assistant.")
    parser.add_argument("--api-base", default=os.environ.get("SCHEDULING_API_BASE", "http://localhost:4010"))
    parser.add_argument("--no-llm", action="store_true", help="Use the deterministic rule-based NLU instead of OpenAI.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Echo JSON trace events to stderr.")
    parser.add_argument("--trace-file", default=None, help="JSONL trace path (default traces/session-<ts>.jsonl, '-' to disable).")
    parser.add_argument("--mock-scenario", default=None, help="Value for the X-Mock-Scenario header (e.g. api_failure).")
    args = parser.parse_args(argv)

    assistant = build_assistant(args)
    backend = getattr(assistant.nlu, "name", "rules")
    print(f"(nlu backend: {backend}; api: {args.api_base}; trace session: {assistant.tracer.session_id})")
    print("(type /quit to exit, /reset to start over, /scenario api_failure to simulate an outage)\n")
    print("assistant> " + assistant.greeting().replace("\n", "\n           "))

    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "reset":
                assistant.reset()
                print("assistant> " + assistant.greeting().replace("\n", "\n           "))
                continue
            if cmd == "scenario":
                if arg and arg != "off":
                    assistant.client.extra_headers["X-Mock-Scenario"] = arg
                    print(f"(mock scenario header set to '{arg}')")
                else:
                    assistant.client.extra_headers.pop("X-Mock-Scenario", None)
                    print("(mock scenario header cleared)")
                continue
            if cmd == "state":
                s = assistant.session
                print(f"(state={s.state.value} goal={s.goal} patient={'yes' if s.patient else 'no'} specialty={s.specialty} location={s.location} shown_slots={len(s.shown_slots)})")
                continue
            print("(unknown command)")
            continue
        result = assistant.handle(text)
        failed = [e for e in assistant.tracer.find("llm_call", ok=False) if e["turn"] == assistant.tracer.turn]
        if failed:
            print(f"(llm unavailable: {failed[0].get('error')}; used rule-based parser for this turn)")
        print("assistant> " + result.reply.replace("\n", "\n           "))
    assistant.tracer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
