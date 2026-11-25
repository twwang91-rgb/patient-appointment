"""Offline unit tests for the deterministic pieces (no network, no mock server).

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.api_client import SchedulingClient  # noqa: E402
from assistant.nlu import (  # noqa: E402
    LLMNLU,
    NLUContext,
    RuleNLU,
    contains_medical_request,
    extract_confirmation,
    extract_identifiers,
    normalize_dob,
    normalize_phone,
)
from assistant.tracing import Tracer, fingerprint, redact_text  # noqa: E402


class RedactionTests(unittest.TestCase):
    def test_redacts_phone_dob_zip_in_free_text(self):
        out = redact_text("I'm 555-0101, born 1985-04-12, zip 70112, also (504) 555-0101 and 04/12/1985")
        self.assertNotIn("555-0101", out)
        self.assertNotIn("1985-04-12", out)
        self.assertNotIn("70112", out)
        self.assertNotIn("04/12/1985", out)
        self.assertIn("<PHONE>", out)
        self.assertIn("<DATE>", out)

    def test_fingerprint_is_stable_and_not_reversible(self):
        self.assertEqual(fingerprint("555-0101"), fingerprint("555-0101"))
        self.assertNotIn("0101", fingerprint("555-0101"))

    def test_api_client_redacts_sensitive_query_params(self):
        tracer = Tracer()
        client = SchedulingClient("http://127.0.0.1:1", tracer, retries=0)  # nothing listens here
        with self.assertRaises(Exception):
            client.search_patients("555-0101", "1985-04-12")
        call = tracer.find("api_call")[0]
        self.assertNotIn("555-0101", str(call))
        self.assertNotIn("1985-04-12", str(call))
        self.assertTrue(call["params"]["phone"].startswith("h:"))


class NormalizationTests(unittest.TestCase):
    def test_phone_formats(self):
        for raw in ("555-0101", "5550101", "555 0101", "(504) 555-0101", "504-555-0101", "+1 504 555 0101"):
            self.assertEqual(normalize_phone(raw), "555-0101", raw)
        self.assertIsNone(normalize_phone("12345"))

    def test_dob_formats(self):
        self.assertEqual(normalize_dob("1985-04-12"), "1985-04-12")
        self.assertEqual(normalize_dob("04/12/1985"), "1985-04-12")
        self.assertEqual(normalize_dob("April 12, 1985"), "1985-04-12")
        self.assertIsNone(normalize_dob("2999-01-01"))
        self.assertIsNone(normalize_dob("not a date"))

    def test_identifiers_are_masked_before_llm_sees_text(self):
        ids = extract_identifiers("call me at 555-0101, dob 1985-04-12")
        self.assertEqual(ids["phone"], "555-0101")
        self.assertEqual(ids["dob"], "1985-04-12")
        self.assertEqual(ids["masked"], "call me at <PHONE>, dob <DOB>")


class RuleNLUTests(unittest.TestCase):
    def setUp(self):
        self.nlu = RuleNLU()

    def test_intents(self):
        cases = {
            "Which primary care providers are downtown?": "find_providers",
            "I want to book a primary care appointment.": "book_appointment",
            "Look up my appointment": "lookup_appointments",
            "I have chest pain, should I wait?": "medical_advice",
            "Let me talk to a human": "request_human",
            "hello": "greeting",
        }
        for text, intent in cases.items():
            self.assertEqual(self.nlu.parse(text, NLUContext()).intent, intent, text)

    def test_appointment_does_not_trigger_unsupported_specialty(self):
        p = self.nlu.parse("Book an appointment", NLUContext())
        self.assertIsNone(p.specialty)

    def test_confirmation_words(self):
        self.assertEqual(extract_confirmation("yes"), "yes")
        self.assertEqual(extract_confirmation("Yes, book it"), "yes")
        self.assertEqual(extract_confirmation("no, a different time"), "no")
        self.assertIsNone(extract_confirmation("what time is it"))

    def test_zip_only_extracted_when_expected(self):
        self.assertIsNone(self.nlu.parse("70115", NLUContext()).zip_code)
        self.assertEqual(self.nlu.parse("70115", NLUContext(expecting=["zip"])).zip_code, "70115")

    def test_slot_choice_by_time(self):
        ctx = NLUContext(expecting=["slot_choice"], shown_slots=[
            {"index": 1, "label": "9:00 AM", "hour": 9, "minute": 0},
            {"index": 2, "label": "11:00 AM", "hour": 11, "minute": 0},
        ])
        self.assertEqual(self.nlu.parse("the 11am one", ctx).slot_choice, 2)
        self.assertEqual(self.nlu.parse("first", ctx).slot_choice, 1)
        self.assertIsNone(self.nlu.parse("7", ctx).slot_choice)

    def test_medical_floor(self):
        self.assertTrue(contains_medical_request("Should I wait or go to the ER?"))
        self.assertFalse(contains_medical_request("book me downtown"))


class LLMValidationTests(unittest.TestCase):
    """The LLM's JSON is a proposal; invalid values must be dropped by code."""

    def setUp(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            self.nlu = LLMNLU(Tracer())
        self.ctx = NLUContext(state="CHOOSE_SLOT", expecting=["slot_choice"], shown_slots=[{"index": 1, "label": "x", "hour": 9, "minute": 0}])

    def test_rejects_out_of_enum_and_out_of_range_values(self):
        raw = {"intent": "teleport", "specialty": "cardiology", "location": "mars", "slot_choice": 5, "confirmation": "yes", "zip": "70115", "dob": "garbage"}
        baseline = RuleNLU().parse("hmm", self.ctx)
        p = self.nlu._merge(raw, baseline, self.ctx, "hmm")
        self.assertEqual(p.intent, "other")
        self.assertIsNone(p.specialty)
        self.assertIsNone(p.location)
        self.assertIsNone(p.slot_choice)
        self.assertIsNone(p.confirmation)  # not expecting a confirmation in this state
        self.assertIsNone(p.zip_code)  # not expecting a zip
        self.assertIsNone(p.dob)

    def test_accepts_valid_values_and_applies_medical_floor(self):
        raw = {"intent": "provide_info", "slot_choice": 1, "specialty": "primary_care", "location": "any"}
        baseline = RuleNLU().parse("the 9am please, my chest pain is worse", self.ctx)
        p = self.nlu._merge(raw, baseline, self.ctx, "the 9am please, my chest pain is worse")
        self.assertEqual(p.slot_choice, 1)
        self.assertEqual(p.specialty, "primary_care")
        self.assertEqual(p.intent, "medical_advice")

    def test_yes_in_confirm_state_ignores_echoed_slot_choice(self):
        ctx = NLUContext(state="CONFIRM", expecting=["confirmation", "slot_choice"], shown_slots=[
            {"index": 1, "label": "9:00 AM", "hour": 9, "minute": 0},
            {"index": 2, "label": "11:00 AM", "hour": 11, "minute": 0},
        ])
        raw = {"intent": "provide_info", "slot_choice": 1, "confirmation": "yes"}
        p = self.nlu._merge(raw, RuleNLU().parse("confirm", ctx), ctx, "confirm")
        self.assertEqual(p.confirmation, "yes")
        self.assertIsNone(p.slot_choice)
        # An explicit change of mind still goes through.
        p = self.nlu._merge({"intent": "provide_info", "slot_choice": 1}, RuleNLU().parse("actually the 9am", ctx), ctx, "actually the 9am")
        self.assertEqual(p.slot_choice, 1)

    def test_falls_back_to_rules_when_llm_unreachable(self):
        self.nlu.base_url = "http://127.0.0.1:1"
        self.nlu.timeout = 0.5
        p = self.nlu.parse("I want to book a primary care appointment", NLUContext())
        self.assertEqual(p.intent, "book_appointment")
        self.assertEqual(p.source, "rules_fallback")


class BookingGuardTests(unittest.TestCase):
    def test_client_refuses_unconfirmed_booking_locally(self):
        client = SchedulingClient("http://127.0.0.1:1", Tracer(), retries=0)
        with self.assertRaises(ValueError):
            client.book("pat_1001", "slot_4001", confirmed=False)
        self.assertEqual(client.tracer.find("api_call"), [])  # never hit the network


if __name__ == "__main__":
    unittest.main()
