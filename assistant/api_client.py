"""Thin, typed client for the mock scheduling API (openapi/scheduling-api.yaml).

This is the only module that talks to the scheduling backend. It:
  * maps HTTP outcomes to typed exceptions the workflow can branch on,
  * retries once on 503 / connection errors, then gives up (no infinite retries),
  * traces every call with method, path, status and latency, redacting identifiers,
  * refuses to send a booking unless `confirmed is True` (local policy guard).

Standard library only.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .tracing import Tracer, fingerprint

SPECIALTIES = ("primary_care", "dermatology")
LOCATIONS = ("downtown", "uptown", "lakeside")
HANDOFF_REASONS = (
    "user_requested",
    "identity_unclear",
    "unsupported_request",
    "medical_advice",
    "api_failure",
    "no_availability",
    "other",
)

# Query/body keys whose values must never reach the logs in plaintext.
_SENSITIVE_KEYS = {"phone", "dob", "zip", "zipCode", "summary"}


class ApiError(Exception):
    """Non-2xx response (or malformed response) from the scheduling API."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class ApiUnavailable(ApiError):
    """503 from the API, or the API could not be reached at all."""


class SlotTaken(ApiError):
    """409: the slot was booked by someone else between search and booking."""


class SchedulingClient:
    def __init__(
        self,
        base_url: str,
        tracer: Tracer,
        timeout: float = 5.0,
        retries: int = 1,
        retry_backoff_s: float = 0.3,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.tracer = tracer
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        # e.g. {"X-Mock-Scenario": "api_failure"} to force the outage path on the mock.
        self.extra_headers: Dict[str, str] = dict(extra_headers or {})

    # ------------------------------------------------------------------ plumbing

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json", **self.extra_headers}
        if data is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Optional[ApiError] = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            status: Optional[int] = None
            payload: Any = None
            error: Optional[str] = None
            try:
                req = urllib.request.Request(url, data=data, method=method, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    payload = json.loads(resp.read().decode("utf-8") or "{}")
            except urllib.error.HTTPError as http_err:
                status = http_err.code
                try:
                    payload = json.loads(http_err.read().decode("utf-8") or "{}")
                except Exception:  # noqa: BLE001
                    payload = {}
                error = payload.get("code") if isinstance(payload, dict) else None
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as net_err:
                error = f"connection_error: {type(net_err).__name__}"
            finally:
                self.tracer.event(
                    "api_call",
                    method=method,
                    path=path,
                    params=self._redact(params),
                    body=self._redact(body) if body else None,
                    status=status,
                    error=error,
                    attempt=attempt + 1,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )

            if status is not None and 200 <= status < 300:
                return payload if isinstance(payload, dict) else {}

            code = (payload or {}).get("code", "unknown") if isinstance(payload, dict) else "unknown"
            message = (payload or {}).get("message", "") if isinstance(payload, dict) else ""
            if status is None or status == 503:
                last_exc = ApiUnavailable(status or 0, code if status else "unreachable", message or "Scheduling system unreachable.")
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_s)
                    continue
                raise last_exc
            if status == 409:
                raise SlotTaken(status, code, message)
            raise ApiError(status, code, message)
        raise last_exc or ApiUnavailable(0, "unreachable", "Scheduling system unreachable.")

    @staticmethod
    def _redact(mapping: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not mapping:
            return None
        return {k: (fingerprint(str(v)) if k in _SENSITIVE_KEYS else v) for k, v in mapping.items()}

    # ------------------------------------------------------------------ endpoints

    def search_patients(self, phone: str, dob: str) -> List[Dict[str, Any]]:
        return self._request("GET", "/patients/search", params={"phone": phone, "dob": dob}).get("matches", [])

    def patient_appointments(self, patient_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/patients/{urllib.parse.quote(patient_id)}/appointments").get("appointments", [])

    def providers(self, specialty: Optional[str] = None, location: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._request("GET", "/providers", params={"specialty": specialty, "location": location}).get("providers", [])

    def availability(
        self,
        patient_id: str,
        specialty: str,
        location: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = {
            "patientId": patient_id,
            "specialty": specialty,
            "location": location,
            "startDate": start_date,
            "endDate": end_date,
        }
        return self._request("GET", "/availability", params=params).get("slots", [])

    def book(self, patient_id: str, slot_id: str, confirmed: bool) -> Dict[str, Any]:
        # Local policy guard: the workflow must never reach this call without an
        # explicit user confirmation. The API enforces it too, but we fail closed here.
        if confirmed is not True:
            raise ValueError("Refusing to book: explicit user confirmation is required.")
        body = {"patientId": patient_id, "slotId": slot_id, "confirmed": True}
        return self._request("POST", "/appointments", body=body).get("appointment", {})

    def create_handoff(self, reason: str, summary: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        if reason not in HANDOFF_REASONS:
            raise ValueError(f"Unknown handoff reason: {reason}")
        body: Dict[str, Any] = {"reason": reason, "summary": summary}
        if patient_id:
            body["patientId"] = patient_id
        return self._request("POST", "/handoffs", body=body)
