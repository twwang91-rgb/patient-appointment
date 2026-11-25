"""Natural-language understanding: turn one user message into a validated `Parse`.

Two implementations share one contract:

  * RuleNLU  - deterministic regex/keyword extraction. Always available, used for
               offline evals and as the fallback when the LLM is unavailable.
  * LLMNLU   - asks an OpenAI chat model for intent + entities as JSON, then runs
               every field through the same validators as RuleNLU. Identifiers that
               the rules already found (phone, DOB, zip) are masked *before* the text
               is sent to the model, so the model only sees "<PHONE>" / "<DOB>".

The model never produces user-facing text and never decides what to do; it only
proposes a classification that deterministic code validates and acts on.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from .api_client import LOCATIONS, SPECIALTIES
from .tracing import Tracer, redact_text

INTENTS = (
    "book_appointment",
    "find_providers",
    "lookup_appointments",
    "request_human",
    "medical_advice",
    "provide_info",
    "greeting",
    "other",
)


@dataclass
class Parse:
    intent: str = "other"
    phone: Optional[str] = None
    dob: Optional[str] = None
    zip_code: Optional[str] = None
    specialty: Optional[str] = None  # primary_care | dermatology | unsupported
    location: Optional[str] = None  # downtown | uptown | lakeside | any | unsupported
    slot_choice: Optional[int] = None  # 1-based index into the slots currently shown
    confirmation: Optional[str] = None  # yes | no
    source: str = "rules"
    notes: List[str] = field(default_factory=list)

    def redacted(self) -> Dict[str, Any]:
        """Representation that is safe to log."""
        return {
            "intent": self.intent,
            "has_phone": self.phone is not None,
            "has_dob": self.dob is not None,
            "has_zip": self.zip_code is not None,
            "specialty": self.specialty,
            "location": self.location,
            "slot_choice": self.slot_choice,
            "confirmation": self.confirmation,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass
class NLUContext:
    """What the workflow is currently waiting for; lets the parser resolve short replies."""

    state: str = "IDLE"
    expecting: List[str] = field(default_factory=list)  # e.g. ["phone", "dob"], ["slot_choice"], ["confirmation"]
    shown_slots: List[Dict[str, Any]] = field(default_factory=list)  # [{"index":1,"label":"Tue Dec 1, 9:00 AM"}]


# --------------------------------------------------------------------------- validators

_PHONE_10 = re.compile(r"(?<!\d)(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})(?!\d)")
_PHONE_7 = re.compile(r"(?<!\d)(\d{3})[-.\s]?(\d{4})(?!\d)")
_DOB_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOB_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_ZIP = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", re.I)


def normalize_phone(raw: Optional[str]) -> Optional[str]:
    """Return the mock's `NNN-NNNN` form (last seven digits) or None."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        digits = digits[3:]
    if len(digits) != 7:
        return None
    return f"{digits[:3]}-{digits[3:]}"


def normalize_dob(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y"):
        try:
            parsed = datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
        if date(1900, 1, 1) <= parsed <= date.today():
            return parsed.isoformat()
        return None
    return None


def normalize_zip(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits if len(digits) == 5 else None


_SPECIALTY_WORDS = {
    "primary_care": ("primary care", "primary-care", "pcp", "family doctor", "family medicine", "general practitioner", "gp ", "internal medicine", "physical", "checkup", "check-up", "annual"),
    "dermatology": ("dermatolog", "skin"),
}
_UNSUPPORTED_SPECIALTY_RE = re.compile(
    r"\b(cardiolog\w*|heart doctor|orthoped\w*|pediatric\w*|neurolog\w*|psychiatr\w*|therapist|dentist|dental|ophthalm\w*|"
    r"eye doctor|ob-?gyn|gynecolog\w*|urolog\w*|oncolog\w*|ent doctor|allerg\w*|physical therap\w*|chiropract\w*|podiatr\w*)\b",
    re.I,
)
_LOCATION_WORDS = {
    "downtown": ("downtown", "down town"),
    "uptown": ("uptown", "up town"),
    "lakeside": ("lakeside", "lake side"),
}
_ANY_LOCATION_WORDS = ("any location", "anywhere", "no preference", "either", "any clinic", "any of them", "doesn't matter", "does not matter", "dont care", "don't care", "wherever")
_UNSUPPORTED_LOCATION_WORDS = ("midtown", "westside", "eastside", "suburb", "airport", "mall", "northside", "southside")

_MEDICAL_WORDS = (
    "chest pain", "pain", "hurt", "bleeding", "should i wait", "should i go", "is it serious", "symptom", "fever",
    "dizzy", "shortness of breath", "can't breathe", "cannot breathe", "rash", "medication", "dosage", "dose",
    "diagnos", "prescri", "is this normal", "what should i take", "emergency", "er or", "urgent care or",
    "overdose", "swelling", "infection", "nausea", "vomit", "numb", "stroke", "heart attack", "allergic reaction",
    "what do i have", "treat",
)
_HUMAN_WORDS = ("human", "a person", "real person", "representative", "agent", "operator", "someone", "staff", "front desk", "receptionist", "talk to somebody", "speak to somebody", "speak with someone", "call me", "escalate", "transfer me")
_YES_WORDS = ("yes", "yes please", "yep", "yeah", "yup", "confirm", "confirmed", "correct", "book it", "go ahead", "please book", "sure", "ok", "okay", "that works", "sounds good", "do it", "proceed", "that's right", "thats right", "absolutely", "please do", "y")
_NO_WORDS = ("no", "nope", "nah", "cancel", "don't", "do not", "dont", "wait", "hold on", "stop", "never mind", "nevermind", "not that", "different", "another", "change", "n", "no thanks", "something else")
_ORDINALS = {"first": 1, "earliest": 1, "soonest": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}


def contains_medical_request(text: str) -> bool:
    """Deterministic floor for the medical-advice policy; runs regardless of the NLU backend."""
    t = " " + text.lower() + " "
    return any(w in t for w in _MEDICAL_WORDS)


def _match_words(text: str, table: Dict[str, tuple]) -> Optional[str]:
    t = " " + text.lower() + " "
    for value, words in table.items():
        if any(w in t for w in words):
            return value
    return None


def extract_specialty(text: str) -> Optional[str]:
    t = " " + text.lower() + " "
    found = _match_words(text, _SPECIALTY_WORDS)
    if found:
        return found
    if _UNSUPPORTED_SPECIALTY_RE.search(text):
        return "unsupported"
    return None


def extract_location(text: str) -> Optional[str]:
    t = " " + text.lower() + " "
    found = _match_words(text, _LOCATION_WORDS)
    if found:
        return found
    if any(w in t for w in _ANY_LOCATION_WORDS):
        return "any"
    if any(w in t for w in _UNSUPPORTED_LOCATION_WORDS):
        return "unsupported"
    return None


def extract_identifiers(text: str) -> Dict[str, Optional[str]]:
    """Find phone / DOB in free text with regexes. Returns normalized values and the masked text."""
    masked = text
    dob = None
    for rx in (_DOB_ISO, _DOB_US):
        m = rx.search(masked)
        if m:
            cand = normalize_dob(m.group(0)) if rx is _DOB_US else normalize_dob(
                f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            )
            if cand:
                dob = cand
                masked = masked[: m.start()] + "<DOB>" + masked[m.end():]
                break
    phone = None
    for rx in (_PHONE_10, _PHONE_7):
        m = rx.search(masked)
        if m:
            cand = normalize_phone(m.group(0))
            if cand:
                phone = cand
                masked = masked[: m.start()] + "<PHONE>" + masked[m.end():]
                break
    return {"phone": phone, "dob": dob, "masked": masked}


def extract_slot_choice(text: str, ctx: NLUContext) -> Optional[int]:
    n = len(ctx.shown_slots)
    if n == 0:
        return None
    t = text.lower().strip()
    m = re.fullmatch(r"(?:option|number|slot|#)?\s*(\d{1,2})\.?", t)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1))
    m = re.search(r"\b(?:option|number|slot|#)\s*(\d{1,2})\b", t)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1))
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", t) and idx <= n:
            return idx
    if re.search(r"\blast\b", t):
        return n
    # "the 11am one" -> match against the labels of the slots that were shown.
    m = _TIME.search(t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        pm = m.group(3).lower().startswith("p")
        if pm and hour != 12:
            hour += 12
        if not pm and hour == 12:
            hour = 0
        for slot in ctx.shown_slots:
            if slot.get("hour") == hour and slot.get("minute") == minute:
                return slot["index"]
    return None


def extract_confirmation(text: str) -> Optional[str]:
    t = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return None
    if t in _NO_WORDS or any(t.startswith(w + " ") for w in _NO_WORDS if len(w) > 1):
        return "no"
    if t in _YES_WORDS or any(t.startswith(w + " ") or t.startswith(w + ",") for w in _YES_WORDS if len(w) > 1):
        return "yes"
    words = set(t.split())
    if words & {"no", "nope", "cancel", "dont", "don't", "wait", "stop"}:
        return "no"
    if words & {"yes", "yep", "yeah", "confirm", "confirmed", "correct", "proceed"}:
        return "yes"
    return None


# --------------------------------------------------------------------------- rule NLU


class RuleNLU:
    name = "rules"

    def parse(self, text: str, ctx: NLUContext) -> Parse:
        p = Parse(source="rules")
        ids = extract_identifiers(text)
        p.phone, p.dob = ids["phone"], ids["dob"]
        lowered = text.lower()

        if "zip" in ctx.expecting:
            z = _ZIP.search(ids["masked"])
            p.zip_code = normalize_zip(z.group(1)) if z else None

        p.specialty = extract_specialty(text)
        p.location = extract_location(text)
        if p.location is None and "location" in ctx.expecting and re.fullmatch(r"\s*(any|either|whatever|anything|all)\s*[.!]?\s*", lowered):
            p.location = "any"
        if "slot_choice" in ctx.expecting or "confirmation" in ctx.expecting:
            p.slot_choice = extract_slot_choice(text, ctx)
        if "confirmation" in ctx.expecting or "yes_no" in ctx.expecting:
            p.confirmation = extract_confirmation(text)

        if contains_medical_request(text):
            p.intent = "medical_advice"
        elif any(w in " " + lowered + " " for w in _HUMAN_WORDS):
            p.intent = "request_human"
        elif re.search(r"\b(my|existing|upcoming|current)\b.*\bappointment", lowered) or re.search(r"\b(look\s*up|check|find|see|view|when is)\b.*\b(my )?appointment", lowered) or "do i have an appointment" in lowered:
            p.intent = "lookup_appointments"
        elif re.search(r"\b(book|schedule|make|set up|setup|get|need|want|arrange|reserve)\b.*\b(appointment|appt|visit|slot|see a|see the)\b", lowered) or re.search(r"\b(appointment|appt)\b.*\b(book|schedule|please)\b", lowered) or lowered.strip() in ("book", "book an appointment", "appointment"):
            p.intent = "book_appointment"
        elif re.search(r"\b(provider|doctor|physician|dermatologist|dr\b|pcp|who)s?\b", lowered) and re.search(r"\b(which|what|who|list|show|find|any|are there|available|do you have|tell me)\b", lowered):
            p.intent = "find_providers"
        elif re.fullmatch(r"\s*(hi|hello|hey|good (morning|afternoon|evening))[.! ]*", lowered):
            p.intent = "greeting"
        elif any([p.phone, p.dob, p.zip_code, p.specialty, p.location, p.slot_choice, p.confirmation]):
            p.intent = "provide_info"
        else:
            p.intent = "other"
        return p


# --------------------------------------------------------------------------- LLM NLU

_SYSTEM_PROMPT = """You are the natural-language understanding component of a medical appointment scheduling assistant.
You do NOT talk to the user and you do NOT decide what happens next. You only classify the user's latest message
and extract entities as strict JSON. Deterministic code validates your output and runs the workflow.

Return ONLY a JSON object with exactly these keys:
  intent: one of ["book_appointment","find_providers","lookup_appointments","request_human","medical_advice","provide_info","greeting","other"]
  phone: string or null   (only if the user wrote a phone number that was not already masked as <PHONE>)
  dob:   "YYYY-MM-DD" or null (date of birth, only if present and not already masked as <DOB>)
  zip:   5-digit string or null (only when the workflow is expecting a zip code)
  specialty: one of ["primary_care","dermatology","unsupported"] or null. "unsupported" = a specialty we do not offer (cardiology, dentistry, ...).
  location:  one of ["downtown","uptown","lakeside","any","unsupported"] or null. "any" = user has no preference.
  slot_choice: integer or null - the 1-based index of the shown slot the user picked (match by number, ordinal, or time).
  confirmation: "yes" | "no" | null - ONLY when the workflow is expecting a confirmation and the user clearly agreed or declined.

Rules:
- intent "medical_advice": the user asks about symptoms, whether to wait, what to take, urgency, diagnosis or treatment.
- intent "request_human": the user asks for a person, staff, representative, or to be transferred.
- intent "lookup_appointments": the user wants to see/check their existing or upcoming appointments.
- intent "book_appointment": the user wants a new appointment.
- intent "find_providers": the user asks which providers/doctors exist, without asking to book.
- intent "provide_info": the message mainly supplies information the workflow asked for (phone, DOB, zip, specialty, location, slot pick).
  When `expecting` is non-empty and the message answers it (e.g. "1", "the first one", "primary care", "yes"), use "provide_info",
  NOT "book_appointment" or "find_providers".
- Never invent values. If unsure, use null.
"""


class LLMNLU:
    """OpenAI chat-completions based parser with rule-based validation and fallback."""

    name = "llm"

    def __init__(
        self,
        tracer: Tracer,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 20.0,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.tracer = tracer
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.rules = RuleNLU()

    def parse(self, text: str, ctx: NLUContext) -> Parse:
        baseline = self.rules.parse(text, ctx)  # always computed: identifiers + fallback
        ids = extract_identifiers(text)
        masked_text = ids["masked"]

        user_payload = {
            "workflow_state": ctx.state,
            "expecting": ctx.expecting,
            "shown_slots": [{"index": s["index"], "label": s["label"]} for s in ctx.shown_slots],
            "message": masked_text,
        }
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }
        raw: Optional[Dict[str, Any]] = None
        with self.tracer.timed("llm_call", model=self.model) as t:
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    method="POST",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                raw = json.loads(content)
                usage = data.get("usage") or {}
                t.update(ok=True, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
            except urllib.error.HTTPError as err:
                t.update(ok=False, error=f"http_{err.code}")
            except Exception as err:  # noqa: BLE001 - any failure falls back to rules
                t.update(ok=False, error=type(err).__name__)

        if not isinstance(raw, dict):
            baseline.source = "rules_fallback"
            baseline.notes.append("llm_unavailable")
            return baseline
        return self._merge(raw, baseline, ctx, text)

    def _merge(self, raw: Dict[str, Any], baseline: Parse, ctx: NLUContext, text: str) -> Parse:
        p = Parse(source="llm")
        intent = raw.get("intent")
        p.intent = intent if intent in INTENTS else baseline.intent
        # Identifiers: prefer the locally-extracted values; accept LLM values only after validation.
        p.phone = baseline.phone or normalize_phone(raw.get("phone"))
        p.dob = baseline.dob or normalize_dob(raw.get("dob"))
        p.zip_code = baseline.zip_code or (normalize_zip(raw.get("zip")) if "zip" in ctx.expecting else None)
        spec = raw.get("specialty")
        p.specialty = spec if spec in SPECIALTIES + ("unsupported",) else baseline.specialty
        loc = raw.get("location")
        p.location = loc if loc in LOCATIONS + ("any", "unsupported") else baseline.location
        choice = raw.get("slot_choice")
        if "confirmation" in ctx.expecting and baseline.slot_choice is None:
            # While confirming, models tend to echo the selected slot as slot_choice. Accept a
            # slot change here only when the text itself contains a number/ordinal/time.
            p.slot_choice = None
        elif isinstance(choice, int) and 1 <= choice <= len(ctx.shown_slots) and ("slot_choice" in ctx.expecting or "confirmation" in ctx.expecting):
            p.slot_choice = choice
        else:
            p.slot_choice = baseline.slot_choice
        conf = raw.get("confirmation")
        if conf in ("yes", "no") and ("confirmation" in ctx.expecting or "yes_no" in ctx.expecting):
            p.confirmation = conf
        else:
            p.confirmation = baseline.confirmation
        # Policy floor: deterministic medical screen wins regardless of the model's label.
        if contains_medical_request(text):
            p.intent = "medical_advice"
            p.notes.append("medical_keyword_floor")
        return p


def build_nlu(tracer: Tracer, use_llm: bool) -> Any:
    """Pick the LLM parser when requested and configured, otherwise the rule parser."""
    if use_llm:
        try:
            nlu = LLMNLU(tracer)
            tracer.event("nlu_backend", backend="llm", model=nlu.model)
            return nlu
        except RuntimeError as err:
            tracer.event("nlu_backend", backend="rules", reason=str(err))
            return RuleNLU()
    tracer.event("nlu_backend", backend="rules")
    return RuleNLU()


__all__ = ["Parse", "NLUContext", "RuleNLU", "LLMNLU", "build_nlu", "redact_text", "contains_medical_request"]
