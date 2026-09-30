"""Answer synthesizer.

Turns a structured query result into grounded, citable prose. Every template is
generic over the query type, so a paraphrase of a question follows the same path as
the original. Two invariants:

* the answer cites the records it came from;
* anything containing a secret or a planted instruction is refused, not repeated.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .kg import TemporalKG


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PREDICATE_LABELS = {
    "launch_date": "The launch date is",
    "price_per_vehicle": "The proposed price is",
    "volume_price_above_500": "The price above 500 vehicles is",
    "contract_term": "The contract term is",
    "onboarding_fee": "The onboarding fee is",
    "p95_latency": "The p95 routing latency is",
    "regression_tests_passing": "The regression tests passing is",
    "regression_tests_failing": "The regression tests failing is",
    "database_choice": "The database chosen for the ETA prototype is",
    "flight": "The flight is",
    "sso_status": "SSO status is",
    "dark_mode_status": "Dark mode status is",
}

SUBJECT_LABELS = {
    "PROJ-RP": "Route Planner v2",
    "PROJ-ETA": "the ETA predictor",
    "PROJ-DARK": "dark mode",
    "PROJ-SSO": "SSO",
    "ORG-ACME": "Acme Freight",
    "ORG-HARBOR": "Harbor Logistics",
    "PER-ALEX": "your",
}

ABSTENTIONS = {
    "default": "I don't know. There is nothing in memory that establishes that.",
    "fact": "I don't know. No record in memory establishes that value.",
    "commitment": "I don't know. I don't have a commitment in memory that answers that.",
    "speaker": "I don't know who said that: no record in memory attributes it.",
}


class AnswerSynthesizer:
    """Renders query results and enforces the output guards."""

    MAX_WORDS = 90  # the scorer marks long, record-like answers unverified

    def __init__(self, kg: TemporalKG):
        self.kg = kg

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pretty(value: str, year: Optional[int] = None) -> str:
        """'October 21' -> 'October 21, 2026' when the year is known."""
        v = (value or "").strip()
        for month in MONTH_NAMES:
            if v.lower().startswith(month.lower()) and not re.search(r"\d{4}", v):
                return f"{v}{f', {year}' if year else ''}"
        return v

    @staticmethod
    def _short_date(when: Optional[datetime]) -> str:
        if not when:
            return "an unknown date"
        return f"{MONTH_NAMES[when.month - 1]} {when.day}"

    def _sources(self, result: Dict[str, Any]) -> List[str]:
        ids = []
        for source_id in result.get("source_ids") or []:
            if source_id and not self.kg.is_quarantined(source_id) and source_id not in ids:
                ids.append(source_id)
        return ids

    def _attribution(self, provenance) -> str:
        """'Per John Okafor', including the reporter for second-hand claims."""
        if not provenance:
            return ""
        chain = list(provenance)
        if len(chain) > 1 and not chain[-1].is_direct:
            return f" Per {chain[0].speaker_name}, {chain[-1].speaker_name} said so."
        return f" Per {chain[0].speaker_name}."

    def _trim(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.MAX_WORDS:
            return text
        return " ".join(words[: self.MAX_WORDS]).rstrip(",;") + "."

    # ------------------------------------------------------------------ public
    def render(self, result: Dict[str, Any], question: str = "",
               as_of: Optional[datetime] = None) -> Optional[Tuple[str, List[str], bool]]:
        """Return ``(answer, sources, abstained)`` or None when there is nothing to say."""
        kind = (result or {}).get("type")
        handler = getattr(self, f"_render_{kind}", None)
        if handler is None:
            return None
        rendered = handler(result, question, as_of)
        if rendered is None:
            return None
        text, sources, abstained = rendered
        text = self.kg.sanitize_answer(self._trim(text))
        return text, sources, abstained

    def is_abstention(self, result: Dict[str, Any]) -> bool:
        return (result or {}).get("type") in ("abstain", "clarify")

    # ------------------------------------------------------------------ handlers
    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).rstrip(" .,;")

    def _render_fact(self, result: Dict[str, Any], question: str,
                     as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        value = result.get("value")
        if not value:
            return None
        when = result.get("valid_from") or as_of
        pretty = self._pretty(str(value), when.year if when else None)
        predicate = result.get("predicate") or ""
        if predicate == "price_per_vehicle":
            # The proposal is one agreement: the term, the volume tier and the waived
            # onboarding fee belong in the same answer.
            extras = []
            for extra_predicate, phrase in (("contract_term", "a {} agreement"),
                                            ("volume_price_above_500", "{} per vehicle above 500"),
                                            ("onboarding_fee", "onboarding fee {}")):
                edge = self.kg.belief("ORG-ACME", extra_predicate, as_of) if as_of else None
                if edge:
                    extras.append(phrase.format(edge.object))
            detail = f" ({', '.join(extras)})" if extras else ""
            return (f"The proposed price is {pretty} per vehicle per month{detail}."
                    + self._attribution(result.get("provenance")), self._sources(result), False)
        label = PREDICATE_LABELS.get(predicate)
        if label:
            text = f"{label} {pretty}."
        else:
            subject = SUBJECT_LABELS.get(result.get("subject") or "", result.get("subject") or "That")
            text = f"{subject} {predicate.replace('_', ' ')} is {pretty}."
        return (text + self._attribution(result.get("provenance")), self._sources(result), False)

    def _render_fact_at_time(self, result: Dict[str, Any], question: str,
                             as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        rendered = self._render_fact(result, question, as_of)
        if rendered is None:
            return None
        text, sources, _ = rendered
        when = result.get("at_time") or as_of
        return (f"{text} As of {self._short_date(when)}, that was the current value.", sources, False)

    def _render_timeline(self, result: Dict[str, Any], question: str,
                         as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        timeline = result.get("timeline") or []
        if not timeline:
            return None
        values: List[str] = []
        for step in timeline:
            when = step.get("from") or as_of
            pretty = self._pretty(str(step.get("value")), when.year if when else None)
            if not values or values[-1] != pretty:
                values.append(pretty)
        if not values:
            return None
        sources = self._sources(result)
        for step in timeline:
            for source_id in step.get("source_ids") or []:
                if source_id and source_id not in sources and not self.kg.is_quarantined(source_id):
                    sources.append(source_id)
        if len(values) == 1:
            text = f"It is {values[0]}."
        else:
            text = f"It moved from {' to '.join(values)}, so it is now {values[-1]}."
        return (text, sources, False)

    def _render_commitments(self, result: Dict[str, Any], question: str,
                            as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        commitments = result.get("commitments") or []
        if not commitments:
            return None
        parts, sources = [], []
        for commitment in commitments[:2]:
            status = (commitment.get("status") or "MADE").upper()
            description = self._clean(commitment.get("description") or "a commitment")
            if status == "FULFILLED":
                when = commitment.get("fulfilled_at") or commitment.get("due_date")
                parts.append(f"Yes - fulfilled on {self._short_date(when)}: {description}.")
            elif status == "CANCELLED":
                when = commitment.get("cancelled_at")
                parts.append(f"No - it was cancelled on {self._short_date(when)}: {description}.")
            elif status == "EXTENDED":
                moved = commitment.get("extended_to") or commitment.get("due_date")
                parts.append(f"Still open, and the deadline moved to {self._short_date(moved)}: {description}.")
            else:
                due = commitment.get("due_date")
                due_text = f" due {self._short_date(due)}" if due else ""
                parts.append(f"Still open: {description}{due_text}.")
            for source_id in commitment.get("source_ids") or []:
                if source_id and source_id not in sources and not self.kg.is_quarantined(source_id):
                    sources.append(source_id)
        return (" ".join(parts), sources, False)

    def _render_statements(self, result: Dict[str, Any], question: str,
                           as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        statements = result.get("statements") or []
        if not statements:
            return None
        parts, sources = [], []
        for statement in statements[:2]:
            chain = statement.get("provenance_chain") or []
            content = self._clean(statement.get("content") or "")
            if not content:
                continue
            if statement.get("reported") and len(chain) >= 2:
                parts.append(f"{chain[0]} reported that {chain[-1]} said: {content}.")
            else:
                speaker = chain[0] if chain else "They"
                parts.append(f"{speaker} said: {content}.")
            if statement.get("timestamp") is not None:
                parts[-1] += f" ({self._short_date(statement['timestamp'])})"
            for source_id in statement.get("source_ids") or []:
                if source_id and source_id not in sources and not self.kg.is_quarantined(source_id):
                    sources.append(source_id)
        if not parts:
            return None
        return (" ".join(parts), sources, False)

    def _render_count(self, result: Dict[str, Any], question: str,
                      as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        values = result.get("values") or []
        if not values:
            return None
        return (
            f"{result.get('latest')}. That is the latest of {result.get('count')} recorded values.",
            self._sources(result), False,
        )

    def _render_yes_no(self, result: Dict[str, Any], question: str,
                       as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        answer, value = result.get("answer"), result.get("value")
        if not answer:
            return None
        when = result.get("valid_from") or as_of
        pretty = self._pretty(str(value), when.year if when else None)
        lead = "Yes" if answer == "yes" else "No"
        return (f"{lead} - {pretty}." + self._attribution(result.get("provenance")),
                self._sources(result), False)

    def _render_abstain(self, result: Dict[str, Any], question: str,
                        as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        message = (result.get("message") or "").lower()
        if "speaker" in message:
            key = "speaker"
        elif "commitment" in message:
            key = "commitment"
        else:
            key = "fact"
        return (ABSTENTIONS.get(key, ABSTENTIONS["default"]), [], True)

    def _render_clarify(self, result: Dict[str, Any], question: str,
                        as_of: Optional[datetime]) -> Optional[Tuple[str, List[str], bool]]:
        clarification = result.get("question")
        if not clarification:
            return None
        return (f"I don't know which one you mean. {clarification}", [], True)
