"""Deterministic scheduling workflow.

Division of labour (the point of the exercise):

  * The NLU (LLM or rules) proposes: intent + entities. Nothing else.
  * This module decides: state transitions, when identity is required, when a
    confirmation is explicit, which API to call, how to handle each failure,
    and when to hand off. It is plain Python and fully testable offline.
  * The API decides: who the patient is, which slots exist, whether a booking
    succeeded. Every provider, slot, appointment, or handoff id shown to the
    user is copied from an API response - the assistant never composes one.

User-facing text is rendered by templates in this module from API data, so the
model cannot fabricate a slot, a patient, or a confirmation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .api_client import ApiError, ApiUnavailable, SchedulingClient, SlotTaken
from .nlu import NLUContext, Parse
from .tracing import Tracer, redact_text

MAX_IDENTITY_ATTEMPTS = 2  # failed lookups before we hand off instead of asking again

HUMAN_NEXT_STEP = "If you'd rather not wait, you can also call the clinic's front desk directly."
EMERGENCY_LINE = "If you think this could be an emergency, please call 911 or go to the nearest emergency room now."


class State(str, Enum):
    IDLE = "IDLE"
    IDENTIFY = "IDENTIFY"  # waiting for phone and/or DOB
    DISAMBIGUATE = "DISAMBIGUATE"  # multiple patient matches; waiting for zip
    COLLECT_PREFS = "COLLECT_PREFS"  # waiting for specialty / location
    CHOOSE_SLOT = "CHOOSE_SLOT"  # slots shown; waiting for a pick
    CONFIRM = "CONFIRM"  # one slot selected; waiting for explicit yes/no
    NO_AVAILABILITY = "NO_AVAILABILITY"  # nothing found; offering alternatives or a handoff
    HANDED_OFF = "HANDED_OFF"  # a handoff was created this session


@dataclass
class Session:
    state: State = State.IDLE
    goal: Optional[str] = None  # "book" | "lookup"
    patient: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    phone: Optional[str] = None  # held only until the lookup runs
    dob: Optional[str] = None
    identity_attempts: int = 0
    specialty: Optional[str] = None
    location: Optional[str] = None  # None = not asked yet, "any" = no filter
    shown_slots: List[Dict[str, Any]] = field(default_factory=list)
    selected_slot: Optional[Dict[str, Any]] = None
    rejected_slot_ids: Set[str] = field(default_factory=set)  # slots that 409'd this session
    last_handoff: Optional[Dict[str, Any]] = None
    last_outcome: Optional[str] = None  # set each turn, for traces/evals
    last_handoff_reason: Optional[str] = None


@dataclass
class TurnResult:
    reply: str
    state: State
    outcome: str
    handoff_reason: Optional[str] = None


class Assistant:
    def __init__(self, client: SchedulingClient, nlu: Any, tracer: Tracer):
        self.client = client
        self.nlu = nlu
        self.tracer = tracer
        self.session = Session()
        self._provider_names: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------ public API

    def greeting(self) -> str:
        return (
            "Hi! I can help you find a provider, book an appointment, or look up an existing appointment.\n"
            "What would you like to do?"
        )

    def handle(self, text: str) -> TurnResult:
        """Process one user message and return the assistant's reply."""
        s = self.session
        self.tracer.turn += 1
        started = time.perf_counter()
        state_before = s.state
        s.last_outcome, s.last_handoff_reason = None, None

        ctx = self._nlu_context()
        with self.tracer.timed("nlu", backend=getattr(self.nlu, "name", "?")) as t:
            parse: Parse = self.nlu.parse(text, ctx)
            t.update(result=parse.redacted())
        self.tracer.event("user_turn", text=redact_text(text), state=s.state.value)

        try:
            reply = self._dispatch(parse, text, ctx)
        except ApiUnavailable as err:
            reply = self._handoff(
                "api_failure",
                "Scheduling system unavailable during a scheduling request.",
                user_message=(
                    "I'm sorry - the scheduling system isn't responding right now, so I can't look anything up or book "
                    "for you at the moment. Nothing has been booked."
                ),
                api_error=err,
            )
        except SlotTaken:
            # Only POST /appointments raises this; handled in _book(), but keep a safe default.
            reply = "That slot is no longer available. Nothing was booked."
        except ApiError as err:
            self.tracer.event("api_error", status=err.status, code=err.code)
            reply = self._handoff(
                "other",
                f"Unexpected scheduling API error {err.status} {err.code}.",
                user_message="I ran into an unexpected problem with the scheduling system and couldn't finish that. Nothing has been booked.",
            )

        outcome = s.last_outcome or "reply"
        self.tracer.event(
            "turn_summary",
            intent=parse.intent,
            nlu_source=parse.source,
            state_before=state_before.value,
            state_after=s.state.value,
            outcome=outcome,
            handoff_reason=s.last_handoff_reason,
            api_calls=len([e for e in self.tracer.events if e["event"] == "api_call" and e["turn"] == self.tracer.turn]),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return TurnResult(reply=reply, state=s.state, outcome=outcome, handoff_reason=s.last_handoff_reason)

    def reset(self) -> None:
        self.session = Session()

    # ------------------------------------------------------------------ routing

    def _nlu_context(self) -> NLUContext:
        s = self.session
        expecting: List[str] = []
        if s.state == State.IDENTIFY:
            expecting = [k for k in ("phone", "dob") if getattr(s, k) is None]
        elif s.state == State.DISAMBIGUATE:
            expecting = ["zip"]
        elif s.state == State.COLLECT_PREFS:
            expecting = ["specialty"] if not s.specialty else ["location"]
        elif s.state == State.CHOOSE_SLOT:
            expecting = ["slot_choice"]
        elif s.state == State.CONFIRM:
            expecting = ["confirmation", "slot_choice"]
        elif s.state == State.NO_AVAILABILITY:
            expecting = ["yes_no", "location", "specialty"]
        shown = []
        for i, slot in enumerate(s.shown_slots, start=1):
            dt = _parse_time(slot["startTime"])
            shown.append({"index": i, "label": _fmt_time(slot["startTime"]), "hour": dt.hour if dt else None, "minute": dt.minute if dt else None})
        return NLUContext(state=s.state.value, expecting=expecting, shown_slots=shown)

    def _dispatch(self, p: Parse, text: str, ctx: NLUContext) -> str:
        s = self.session

        # 1. Safety / escalation intents win regardless of state (code decides, not the model).
        if p.intent == "medical_advice":
            return self._handoff(
                "medical_advice",
                "User asked for medical advice; no advice given.",
                user_message=(
                    "I'm not able to give medical advice or tell you whether it's safe to wait - that needs a clinician.\n"
                    f"{EMERGENCY_LINE}\n"
                    "I'm connecting you with our care team so a person can follow up with you."
                ),
            )
        if p.intent == "request_human" and s.state == State.NO_AVAILABILITY:
            return self._handle_no_availability(p)
        if p.intent == "request_human":
            return self._handoff(
                "user_requested",
                "User asked to speak with a human.",
                user_message="Of course. I'm handing this over to a member of our staff who will follow up with you.",
            )

        # 2. Absorb any entities the user supplied, in whatever order they arrived.
        self._absorb(p)

        # 3. Unsupported specialty / location -> out of scope, hand off.
        if p.specialty == "unsupported":
            return self._handoff(
                "unsupported_request",
                "User asked for a specialty we do not schedule (only primary_care, dermatology).",
                user_message=(
                    "I can only schedule primary care and dermatology appointments here. "
                    "I'm passing your request to our staff so they can help with that specialty."
                ),
            )
        if p.location == "unsupported":
            return self._handoff(
                "unsupported_request",
                "User asked for a location we do not serve (only downtown, uptown, lakeside).",
                user_message=(
                    "Our clinics are downtown, uptown, and lakeside - I don't have anything at that location. "
                    "I'm passing your request to our staff so they can help."
                ),
            )

        # 4. If the message answers what the workflow just asked, the answer wins over the
        #    model's intent label. The model often tags "1" or "primary care" as
        #    book_appointment / find_providers mid-flow; code, not the model, owns the state.
        if self._answers_expected(p, ctx):
            if s.state == State.CONFIRM:
                return self._handle_confirm(p)
            if s.state == State.CHOOSE_SLOT:
                return self._handle_slot_choice(p)
            if s.state == State.DISAMBIGUATE:
                return self._handle_disambiguation(p) or self._ask_zip()
            if s.state in (State.IDENTIFY, State.COLLECT_PREFS):
                return self._advance()

        # 5. Intent-driven transitions.
        if p.intent == "find_providers":
            return self._provider_lookup(p)
        if p.intent == "book_appointment":
            s.goal = "book"
            if s.state in (State.CHOOSE_SLOT, State.CONFIRM, State.NO_AVAILABILITY) and (p.specialty or p.location):
                # New search criteria mid-flow: start the slot search over.
                s.shown_slots, s.selected_slot = [], None
                s.state = State.COLLECT_PREFS
            return self._advance()
        if p.intent == "lookup_appointments":
            s.goal = "lookup"
            return self._advance()

        # 6. State-specific handling of anything else while mid-flow.
        if s.state == State.CONFIRM:
            return self._handle_confirm(p)
        if s.state == State.CHOOSE_SLOT:
            return self._handle_slot_choice(p)
        if s.state == State.NO_AVAILABILITY:
            return self._handle_no_availability(p)
        if s.state == State.DISAMBIGUATE:
            return self._handle_disambiguation(p) or self._ask_zip()
        if s.state in (State.IDENTIFY, State.COLLECT_PREFS):
            return self._advance()

        if p.intent == "greeting":
            return self.greeting()
        if p.intent == "provide_info" and s.goal:
            return self._advance()
        if p.intent == "provide_info" and (p.specialty or p.location):
            # Criteria without a stated goal: treat as a provider lookup (no identity needed).
            return self._provider_lookup(p)
        return (
            "I can help you find a provider, book a primary care or dermatology appointment, or look up an existing "
            "appointment. Which would you like? (You can also ask for a person at any time.)"
        )

    @staticmethod
    def _answers_expected(p: Parse, ctx: NLUContext) -> bool:
        """True when the parse supplies a value the current state asked for."""
        exp = set(ctx.expecting)
        if "slot_choice" in exp and p.slot_choice is not None:
            return True
        if "confirmation" in exp and p.confirmation is not None:
            return True
        if "zip" in exp and p.zip_code is not None:
            return True
        if "specialty" in exp and p.specialty in ("primary_care", "dermatology"):
            return True
        if "location" in exp and p.location in ("downtown", "uptown", "lakeside", "any"):
            return True
        if exp & {"phone", "dob"} and (p.phone or p.dob):
            return True
        return False

    def _absorb(self, p: Parse) -> None:
        s = self.session
        if p.phone and s.patient is None:
            s.phone = p.phone
        if p.dob and s.patient is None:
            s.dob = p.dob
        if p.specialty in ("primary_care", "dermatology"):
            s.specialty = p.specialty
        if p.location in ("downtown", "uptown", "lakeside", "any"):
            s.location = p.location

    # ------------------------------------------------------------------ provider lookup (no identity needed)

    def _provider_lookup(self, p: Parse) -> str:
        s = self.session
        specialty = p.specialty if p.specialty in ("primary_care", "dermatology") else None
        location = p.location if p.location in ("downtown", "uptown", "lakeside") else None
        providers = self.client.providers(specialty=specialty, location=location)
        s.last_outcome = "providers_listed"
        scope = " ".join(x for x in [_label(specialty), "providers" if specialty else "providers", f"at {location}" if location else ""] if x).strip()
        if not providers:
            return (
                f"I didn't find any {scope} in the scheduling system."
                + ("\n" + self._provider_hint(specialty) if specialty else "")
                + "\nWould you like me to search a different specialty or location?"
            )
        lines = [f"Here are the {scope} I found:"]
        for prov in providers:
            lines.append(
                f"  - {prov['name']} ({_label(prov['specialty'])}) - locations: {', '.join(prov.get('locations', []))}; "
                f"visit types: {', '.join(m.replace('_', ' ') for m in prov.get('modalities', []))}"
            )
        lines.append("Would you like to book an appointment with one of them?")
        return "\n".join(lines)

    def _provider_hint(self, specialty: Optional[str]) -> str:
        """Tell the user where a specialty IS offered, using real provider data."""
        if not specialty:
            return ""
        try:
            provs = self.client.providers(specialty=specialty)
        except ApiError:
            return ""
        locations = sorted({loc for prov in provs for loc in prov.get("locations", [])})
        if not locations:
            return f"We don't currently list any {_label(specialty)} providers."
        return f"{_label(specialty).capitalize()} is offered at: {', '.join(locations)}."

    def _cache_provider_names(self, providers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        if self._provider_names is None:
            try:
                providers = providers if providers is not None else self.client.providers()
                self._provider_names = {prov["providerId"]: prov["name"] for prov in providers}
            except ApiError:
                return {}
        return self._provider_names

    # ------------------------------------------------------------------ the main driver

    def _advance(self) -> str:
        """Move the current goal forward: identify -> collect prefs -> search -> choose."""
        s = self.session
        if s.goal is None:
            return "Would you like to book an appointment or look up an existing one?"

        # Identity gate: patient-specific actions require a verified patient.
        if s.patient is None:
            if s.state == State.DISAMBIGUATE:
                return self._ask_zip()
            if s.phone and s.dob:
                return self._identify()
            s.state = State.IDENTIFY
            missing = [m for m, v in (("the phone number on your file", s.phone), ("your date of birth", s.dob)) if not v]
            lead = "To book that, I first need to verify who you are. " if s.goal == "book" else "To look up your appointments, I first need to verify who you are. "
            return lead + f"Please share {' and '.join(missing)}" + (" (for example 555-0101 and 1985-04-12)." if len(missing) == 2 else ".")

        if s.goal == "lookup":
            return self._show_appointments()

        # goal == "book"
        if not s.specialty:
            s.state = State.COLLECT_PREFS
            return "What kind of appointment do you need - primary care or dermatology?"
        if s.location is None:
            s.state = State.COLLECT_PREFS
            return "Which clinic would you prefer - downtown, uptown, or lakeside? You can also say 'any'."
        return self._search_slots()

    def _identify(self) -> str:
        s = self.session
        matches = self.client.search_patients(s.phone, s.dob)
        self.tracer.event("identity_check", match_count=len(matches), attempt=s.identity_attempts + 1)
        if len(matches) == 1:
            s.patient = matches[0]
            s.candidates = []
            s.phone, s.dob = None, None  # don't keep identifiers around longer than needed
            s.last_outcome = "patient_identified"
            self.tracer.event("patient_identified", patientId=s.patient["patientId"])
            return f"Thanks, {s.patient['firstName']} - I found your record.\n" + self._advance()
        if len(matches) > 1:
            s.candidates = matches
            s.state = State.DISAMBIGUATE
            s.last_outcome = "identity_ambiguous"
            return self._ask_zip(first=True)
        # No match.
        s.identity_attempts += 1
        s.phone, s.dob = None, None
        if s.identity_attempts >= MAX_IDENTITY_ATTEMPTS:
            return self._handoff(
                "identity_unclear",
                f"No patient record matched after {s.identity_attempts} attempts.",
                user_message=(
                    "I still couldn't find a record matching those details, so I can't continue with the booking myself. "
                    "I'm handing this to our staff, who can verify your information and help you book."
                ),
            )
        s.state = State.IDENTIFY
        s.last_outcome = "patient_not_found"
        return (
            "I couldn't find a patient record matching that phone number and date of birth. "
            "Could you double-check them and send both again? If you're a new patient, say 'human' and our staff will get you set up."
        )

    def _ask_zip(self, first: bool = False) -> str:
        s = self.session
        if s.state == State.DISAMBIGUATE and not first:
            # We asked already; see if a zip arrived.
            return "I still need the zip code on your file to tell which record is yours - what is it?"
        return (
            "I found more than one record matching those details. To make sure I pick the right one, "
            "what is the zip code on your file?"
        )

    def _handle_disambiguation(self, p: Parse) -> Optional[str]:
        s = self.session
        if not p.zip_code:
            return None
        hits = [c for c in s.candidates if c.get("zipCode") == p.zip_code]
        self.tracer.event("identity_disambiguation", candidate_count=len(s.candidates), match_count=len(hits))
        if len(hits) == 1:
            s.patient = hits[0]
            s.candidates = []
            s.state = State.IDLE
            s.last_outcome = "patient_identified"
            self.tracer.event("patient_identified", patientId=s.patient["patientId"])
            return f"Thanks, {s.patient['firstName']} - I found your record.\n" + self._advance()
        return self._handoff(
            "identity_unclear",
            "Multiple patient records matched phone+DOB and the zip code did not resolve them.",
            user_message=(
                "That zip code doesn't match the records I found, so I can't safely tell which one is yours. "
                "I'm handing this to our staff to verify your identity and help you from there."
            ),
        )

    def _show_appointments(self) -> str:
        s = self.session
        appts = self.client.patient_appointments(s.patient["patientId"])
        names = self._cache_provider_names()
        s.goal = None
        s.state = State.IDLE
        s.last_outcome = "appointments_listed"
        if not appts:
            return "I don't see any appointments on your record. Would you like to book one?"
        lines = ["Here are the appointments on your record:"]
        for a in appts:
            lines.append(
                f"  - {_fmt_time(a['startTime'])} - {_label(a['specialty'])} with {names.get(a['providerId'], a['providerId'])} "
                f"at {a['location']} (status: {a['status']}, ref {a['appointmentId']})"
            )
        lines.append("Is there anything else I can help with?")
        return "\n".join(lines)

    # ------------------------------------------------------------------ booking

    def _search_slots(self) -> str:
        s = self.session
        location = None if s.location == "any" else s.location
        slots = self.client.availability(s.patient["patientId"], s.specialty, location=location)
        slots = [sl for sl in slots if sl.get("available", True) and sl["slotId"] not in s.rejected_slot_ids]
        if not slots:
            s.state = State.NO_AVAILABILITY
            s.shown_slots = []
            s.last_outcome = "no_availability"
            where = f"at {s.location}" if location else "at any of our clinics"
            hint = self._provider_hint(s.specialty)
            return (
                f"I'm sorry - there are no {_label(s.specialty)} openings {where} right now.\n"
                + (hint + "\n" if hint else "")
                + "Would you like me to check a different clinic or specialty, or would you prefer I hand this to our staff to find you a time?"
            )
        s.shown_slots = slots
        s.selected_slot = None
        s.state = State.CHOOSE_SLOT
        s.last_outcome = "slots_shown"
        return self._render_slots("Here's what's available:")

    def _render_slots(self, heading: str) -> str:
        s = self.session
        names = self._cache_provider_names()
        lines = [heading]
        for i, sl in enumerate(s.shown_slots, start=1):
            lines.append(f"  {i}. {_fmt_time(sl['startTime'])} - {names.get(sl['providerId'], sl['providerId'])} at {sl['location']}")
        lines.append("Which one would you like? Reply with the number.")
        return "\n".join(lines)

    def _handle_slot_choice(self, p: Parse) -> str:
        s = self.session
        if p.slot_choice is None:
            if p.confirmation == "no":
                s.state = State.NO_AVAILABILITY
                return "No problem. Would you like me to check a different clinic or specialty, or hand this to our staff?"
            return self._render_slots("Sorry, I didn't catch which one. Here are the options again:")
        s.selected_slot = s.shown_slots[p.slot_choice - 1]
        s.state = State.CONFIRM
        s.last_outcome = "awaiting_confirmation"
        names = self._cache_provider_names()
        sl = s.selected_slot
        return (
            f"You picked: {_fmt_time(sl['startTime'])} - {_label(sl['specialty'])} with {names.get(sl['providerId'], sl['providerId'])} at {sl['location']}.\n"
            "Shall I book this appointment? Please reply 'yes' to confirm or 'no' to choose a different time."
        )

    def _handle_confirm(self, p: Parse) -> str:
        s = self.session
        if p.confirmation == "yes":
            # A "yes" always applies to the slot named in the confirmation prompt, never to a
            # slot_choice the parser attached to it - the user confirmed what they saw.
            return self._book()
        if p.slot_choice is not None and s.shown_slots and s.shown_slots[p.slot_choice - 1] is not s.selected_slot:
            # The user changed their mind mid-confirmation: re-confirm the new pick.
            return self._handle_slot_choice(p)
        if p.confirmation == "no":
            s.selected_slot = None
            s.state = State.CHOOSE_SLOT
            s.last_outcome = "confirmation_declined"
            return self._render_slots("Okay, nothing was booked. Here are the options again:")
        s.last_outcome = "awaiting_confirmation"
        return "Just to be sure - should I book that appointment? Please reply 'yes' or 'no'."

    def _book(self) -> str:
        s = self.session
        sl = s.selected_slot
        assert sl is not None and s.patient is not None and s.state == State.CONFIRM
        try:
            appt = self.client.book(s.patient["patientId"], sl["slotId"], confirmed=True)
        except SlotTaken:
            s.rejected_slot_ids.add(sl["slotId"])
            s.selected_slot = None
            s.last_outcome = "slot_conflict"
            self.tracer.event("booking_conflict", slotId=sl["slotId"])
            # Re-fetch real availability rather than guessing what is left.
            remaining = self.client.availability(s.patient["patientId"], s.specialty, location=None if s.location == "any" else s.location)
            remaining = [r for r in remaining if r.get("available", True) and r["slotId"] not in s.rejected_slot_ids]
            lead = f"I'm sorry - that {_fmt_time(sl['startTime'])} slot was just taken by someone else, so nothing was booked."
            if not remaining:
                s.shown_slots = []
                s.state = State.NO_AVAILABILITY
                return lead + " There are no other openings matching your search right now. Would you like me to check another clinic or specialty, or hand this to our staff?"
            s.shown_slots = remaining
            s.state = State.CHOOSE_SLOT
            return lead + "\n" + self._render_slots("Here's what is still available:")

        names = self._cache_provider_names()
        s.state = State.IDLE
        s.goal = None
        s.shown_slots, s.selected_slot = [], None
        s.last_outcome = "booked"
        self.tracer.event("booking_confirmed", appointmentId=appt.get("appointmentId"), slotId=sl["slotId"])
        return (
            f"You're booked! {_label(appt.get('specialty', sl['specialty']))} with {names.get(appt.get('providerId'), appt.get('providerId'))} "
            f"at {appt.get('location')} on {_fmt_time(appt['startTime'])}.\n"
            f"Confirmation number: {appt.get('appointmentId')} (status: {appt.get('status')}).\n"
            "Is there anything else I can help with?"
        )

    def _handle_no_availability(self, p: Parse) -> str:
        s = self.session
        if p.specialty in ("primary_care", "dermatology") or p.location in ("downtown", "uptown", "lakeside", "any"):
            return self._advance()
        if p.confirmation == "yes" or p.intent == "request_human":
            return self._handoff(
                "no_availability",
                f"No availability for {s.specialty} at {s.location or 'any location'}; user asked staff to find a time.",
                user_message="I've asked our staff to find you a time. Someone will follow up with you.",
            )
        return "Tell me a different clinic (downtown, uptown, lakeside, or any) or specialty to check, or say 'staff' and I'll hand this over."

    # ------------------------------------------------------------------ handoff

    def _handoff(self, reason: str, summary: str, user_message: str, api_error: Optional[ApiError] = None) -> str:
        s = self.session
        patient_id = s.patient["patientId"] if s.patient else None
        s.last_handoff_reason = reason
        s.last_outcome = "handoff"
        try:
            result = self.client.create_handoff(reason, summary, patient_id=patient_id)
            s.last_handoff = {"reason": reason, **result}
            self.tracer.event("handoff_created", reason=reason, handoffId=result.get("handoffId"), api_error=str(api_error) if api_error else None)
            tail = f"\nYour reference number is {result.get('handoffId')}. {HUMAN_NEXT_STEP}"
        except ApiError as err:
            # Even the handoff channel is down: be truthful and give a real next step.
            self.tracer.event("handoff_failed", reason=reason, status=err.status, code=err.code)
            tail = (
                "\nI also couldn't reach our staff queue automatically, so please call the clinic's front desk directly "
                "and they will help you right away."
            )
        s.state = State.HANDED_OFF
        s.goal = None
        s.shown_slots, s.selected_slot = [], None
        return user_message + tail


# ---------------------------------------------------------------------- helpers


def _parse_time(iso: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _fmt_time(iso: str) -> str:
    dt = _parse_time(iso)
    if not dt:
        return str(iso)
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%a %b')} {dt.day}, {dt.year} at {hour}:{dt.minute:02d} {ampm}"


def _label(specialty: Optional[str]) -> str:
    return {"primary_care": "primary care", "dermatology": "dermatology"}.get(specialty or "", specialty or "")
