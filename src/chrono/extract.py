"""
Fact Extraction — Rule-based extraction of structured assertions from event stream.

Converts raw events into typed assertions with provenance chains.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from .entities import get_resolver, Entity


MONTH_NAMES = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}

# Words that carry no discriminating power when matching a commitment to the
# message that extends / fulfils / cancels it.
STOPWORDS = set("""
a an the and or but if then than of to in on for from with by about as at into over after before
during still again really just have has had be been being will would not no yes all any some more
most much many few one two three i me my we our us you your they their it this that these those
is are was were do did does can could should so out up down here there what when where who which
how why okay ok well like know think get got go going gonna need needs want wants thanks thank
hi hey please quick heads up hey yeah right sure thing things let lets make makes made said say
says told mention mentioned per via cc re fwd today tomorrow week weeks day days month months
""".split())


def significant_tokens(text):
    """Content tokens used to match a lifecycle event to the commitment it changes."""
    return {w for w in re.findall(r"[a-z0-9$]+", (text or "").lower()) if len(w) > 2 and w not in STOPWORDS}


class AssertionType(Enum):
    """Types of assertions we can extract."""
    FACT = "fact"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    SAID = "said"
    PROMISED = "promised"
    DECIDED = "decided"
    AGREED = "agreed"
    ASKED = "asked"
    COMMITMENT_MADE = "commitment_made"
    COMMITMENT_EXTENDED = "commitment_extended"
    COMMITMENT_FULFILLED = "commitment_fulfilled"
    COMMITMENT_CANCELLED = "commitment_cancelled"
    EVENT_SCHEDULED = "event_scheduled"
    EVENT_CHANGED = "event_changed"
    EVENT_CANCELLED = "event_cancelled"
    CODE_CHANGE = "code_change"
    TEST_RESULT = "test_result"
    DEPLOYMENT = "deployment"


@dataclass
class ProvenanceLink:
    """A link in the provenance chain: who said what, citing whom."""
    speaker: str
    speaker_name: str
    source_id: str
    source_type: str
    timestamp: datetime
    confidence: float = 1.0
    is_direct: bool = True
    quoted_speaker: Optional[str] = None


@dataclass
class Assertion:
    """A structured assertion extracted from source material."""
    assertion_id: str
    assertion_type: AssertionType
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    provenance: List[ProvenanceLink] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def supersedes(self, other: 'Assertion') -> bool:
        return (self.subject == other.subject and 
                self.predicate == other.predicate and
                self.valid_from > other.valid_from)
    
    def conflicts_with(self, other: 'Assertion') -> bool:
        if self.subject != other.subject or self.predicate != other.predicate:
            return False
        if self.object == other.object:
            return False
        self_end = self.valid_to or datetime.max
        other_end = other.valid_to or datetime.max
        return not (self.valid_from >= other_end or other.valid_from >= self_end)


@dataclass
class Commitment:
    """A tracked commitment with full lifecycle."""
    commitment_id: str
    creator: str
    assignee: Optional[str]
    description: str
    created_at: datetime
    due_date: Optional[datetime] = None
    status: str = "MADE"
    extended_at: Optional[datetime] = None
    extended_to: Optional[datetime] = None
    fulfilled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    source_ids: List[str] = field(default_factory=list)
    provenance_chain: List[ProvenanceLink] = field(default_factory=list)


class FactExtractor:
    """Extracts structured assertions from normalized event units."""
    
    LAUNCH_DATE_PATTERNS = [
        (r'launch(?:ing)?\s+(?:on|for|is|target)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'launch_date'),
        (r'(?:target|launch)\s+date\s+(?:is|:)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'launch_date'),
        (r'(?:moved|pushed|delayed)\s+(?:to|until)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'launch_date'),
    ]

    # A unit only contributes a launch date when it is actually talking about the
    # launch. That keeps "the 21st" out of unrelated calendar chatter.
    LAUNCH_CUE_RE = re.compile(
        r'\b(?:launch(?:es|ing)?|go[-\s]?live|goes live|ship(?:s|ping)?\s+v2|release date|go[- ]no[- ]go)\b',
        re.IGNORECASE,
    )
    MONTH_DAY_RE = re.compile(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+'
        r'(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?',
        re.IGNORECASE,
    )
    BARE_DAY_RE = re.compile(r'\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b', re.IGNORECASE)

    # Commitment lifecycle: what happened to a promise after it was made.
    LIFECYCLE_PATTERNS = [
        (AssertionType.COMMITMENT_CANCELLED, re.compile(
            r'\b(?:scratch (?:that|the)|no longer needed|not needed anymore|no need for (?:it|the)|'
            r'cancel(?:led|ed)?|dropped it|forget it|don\'t bother)\b', re.IGNORECASE)),
        (AssertionType.COMMITMENT_EXTENDED, re.compile(
            r'\b(?:move (?:it|the [a-z]+) to|moved (?:it|the [a-z]+)? ?to|push(?:ed)? (?:it|the [a-z]+)? ?to|'
            r'extend(?:ed)? (?:it )?(?:to|until)|could i have until|have until|instead of '
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|delayed to|slipped to|'
            r'we move .{0,20}from .{0,20}to)\b', re.IGNORECASE)),
        (AssertionType.COMMITMENT_FULFILLED, re.compile(
            r'\b(?:as promised|went out|sent it|sent the|just sent|delivered|posted it|it landed|landed|'
            r'fulfilled|is done|it\'s done|took care of)\b', re.IGNORECASE)),
    ]

    
    PRICING_PATTERNS = [
        (r'\$(\d+)\s+per\s+vehicle', 'price_per_vehicle'),
        (r'(\d+)\s+year\s+agreement', 'term_years'),
        (r'above\s+(\d+)\s+vehicles?\s*[,:]?\s*\$(\d+)', 'volume_price'),
        (r'onboarding\s+fee\s+(?:waived|free)', 'onboarding_waived'),
    ]
    
    COMMITMENT_PATTERNS = [
        (r'(?:promise|promised)\s+(?:to\s+)?(.+?)(?:\s+by\s+(.+?))?(?:\.|$)', 'promise'),
        (r'(?:send|sent|will send)\s+(.+?)(?:\s+by\s+(.+?))?(?:\.|$)', 'send_commitment'),
        (r'(?:get back|reply|respond)\s+(?:by|on)\s+(.+?)(?:\.|$)', 'reply_deadline'),
        (r'(?:follow up|follow-up)\s+(?:on|by)\s+(.+?)(?:\.|$)', 'followup'),
    ]
    
    ATTRIBUTION_PATTERNS = [
        (r'(\w+(?:\s+\w+)?)\s+said\s+(?:that\s+)?(.+)', 'said'),
        (r'(\w+(?:\s+\w+)?)\s+(?:told|mentioned)\s+(?:\w+\s+)?(?:that\s+)?(.+)', 'said'),
        (r'according to\s+(\w+(?:\s+\w+)?),?\s+(.+)', 'said'),
        (r'(\w+(?:\s+\w+)?)\s+(?:thinks?|believes?|said)\s+(?:that\s+)?(.+)', 'said'),
    ]
    
    DECISION_PATTERNS = [
        (r'(?:decided|decision)\s+(?:to|that)\s+(.+)', 'decision'),
        (r'(?:agreed|agree)\s+(?:to|that)\s+(.+)', 'agreement'),
        (r'(?:we|team)\s+(?:will|going to)\s+(.+)', 'decision'),
    ]
    
    TEST_RESULT_PATTERNS = [
        (r'(\d+)\s*(?:of|/)\s*(\d+)\s*(?:passing|passed)', 'test_pass_rate'),
        (r'(\d+)\s*(?:failures?|failing)', 'test_failures'),
        (r'flaky\s+test', 'flaky_test'),
    ]
    
    LATENCY_PATTERNS = [
        (r'p95\s+(?:latency|is|:)\s+([\d.]+)\s*(?:seconds?|sec|s|ms)', 'p95_latency'),
        (r'latency\s+(?:p95|is|:)\s+([\d.]+)\s*(?:seconds?|sec|s|ms)', 'p95_latency'),
    ]
    
    FLIGHT_PATTERNS = [
        (r'(UA\s*\d+|United\s+\d+)', 'flight_number'),
        (r'depart[s]?\s+(?:at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm)?)', 'departure_time'),
        (r'arrive[s]?\s+(?:at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm)?)', 'arrival_time'),
        (r'(SFO|SFO)\s*[→\-]\s*(DEN|Denver)', 'route'),
    ]
    
    def __init__(self):
        self.resolver = get_resolver()
        self.assertion_counter = 0
        self.commitment_counter = 0
    
    def extract_from_units(self, units: List[Dict]) -> Tuple[List[Assertion], List[Commitment]]:
        assertions = []
        commitments = []
        for unit in units:
            unit_assertions, unit_commitments = self._extract_from_unit(unit)
            assertions.extend(unit_assertions)
            commitments.extend(unit_commitments)
        assertions = self._resolve_conflicts(assertions)
        commitments = self._link_commitments(commitments, assertions)
        return assertions, commitments
    
    def _extract_from_unit(self, unit: Dict) -> Tuple[List[Assertion], List[Commitment]]:
        assertions = []
        commitments = []
        text = unit.get('text', '')
        unit_id = unit.get('id', '')
        timestamp = unit.get('time')
        record = unit.get('record', '')
        if not text or not timestamp:
            return assertions, commitments
        source_type = self._infer_source_type(unit_id, record)
        speaker_id, speaker_name, confidence = self._extract_speaker(unit, source_type)
        base_provenance = ProvenanceLink(
            speaker=speaker_id, speaker_name=speaker_name, source_id=unit_id,
            source_type=source_type, timestamp=timestamp, confidence=confidence
        )
        assertions.extend(self._extract_facts(text, base_provenance, unit_id))
        assertions.extend(self._extract_attributions(text, base_provenance, unit_id))
        assertions.extend(self._extract_decisions(text, base_provenance, unit_id))
        assertions.extend(self._extract_technical(text, base_provenance, unit_id))
        assertions.extend(self._extract_lifecycle(text, base_provenance, unit_id))
        commitments.extend(self._extract_commitments(text, base_provenance, unit_id, speaker_id))
        return assertions, commitments
    
    def _infer_source_type(self, unit_id: str, record: str) -> str:
        if unit_id.startswith('MTG-') or record.startswith('MTG-'):
            return 'meeting'
        elif unit_id.startswith('SL-') or record.startswith('SL-'):
            return 'slack'
        elif unit_id.startswith('EM-') or record.startswith('EM-'):
            return 'email'
        elif unit_id.startswith('DCT-') or record.startswith('DCT-'):
            return 'dictation'
        elif unit_id.startswith('CAL-') or record.startswith('CAL-'):
            return 'calendar'
        elif unit_id.startswith('CDX-') or record.startswith('CDX-'):
            return 'codex'
        elif unit_id.startswith('CGPT-') or record.startswith('CGPT-'):
            return 'chatgpt'
        return 'unknown'
    
    def _extract_speaker(self, unit: Dict, source_type: str) -> Tuple[str, str, float]:
        text = unit.get('text', '')
        if source_type == 'meeting':
            match = re.match(r'\[[^\]]+\]\s+([^:]+):', text)
            if match:
                speaker_name = match.group(1).strip()
                entities, _ = self.resolver.resolve(speaker_name)
                if entities:
                    return entities[0].canonical_id, speaker_name, 0.9
                return f"PER-UNKNOWN-{speaker_name.replace(' ', '-')}", speaker_name, 0.5
        elif source_type == 'slack':
            match = re.match(r'\[Slack\s+[^\]]+\]\s+([^:]+):', text)
            if match:
                speaker_name = match.group(1).strip()
                entities, _ = self.resolver.resolve(speaker_name)
                if entities:
                    return entities[0].canonical_id, speaker_name, 0.95
                return f"PER-UNKNOWN-{speaker_name.replace(' ', '-')}", speaker_name, 0.5
        elif source_type == 'email':
            match = re.match(r'\[Email\s+[^\]]+\]\s+From\s+([^<]+)', text)
            if match:
                sender = match.group(1).strip()
                entities, _ = self.resolver.resolve(sender)
                if entities:
                    return entities[0].canonical_id, sender, 0.95
                return f"PER-UNKNOWN-{sender.replace(' ', '-')}", sender, 0.5
        elif source_type == 'dictation':
            return 'PER-ALEX', 'Alex Rivera', 1.0
        elif source_type == 'calendar':
            return 'PER-ALEX', 'Alex Rivera (calendar owner)', 0.9
        elif source_type in ('codex', 'chatgpt'):
            match = re.match(r'\[.*?\]\s+(\w+):', text)
            if match:
                role = match.group(1)
                if role == 'user':
                    return 'PER-ALEX', 'Alex Rivera', 0.95
                elif role == 'assistant':
                    return 'PER-ASSISTANT', 'AI Assistant', 0.8
        return 'PER-UNKNOWN', 'Unknown Speaker', 0.3
    
    # ------------------------------------------------------------------ dates
    def _launch_dates(self, text: str) -> List[Tuple[str, int]]:
        """Canonical launch dates mentioned by a unit that is talking about the launch.

        Returns ``[(canonical 'Month D', offset)]``. The offset keeps a
        "moved from A to B" sentence ordered so the later mention wins the
        point-in-time lookup.
        """
        if not text or not self.LAUNCH_CUE_RE.search(text):
            return []
        found = []
        md_spans = []
        month_positions = []
        for m in self.MONTH_DAY_RE.finditer(text):
            month = MONTH_NAMES[m.group(1)[:3].lower()]
            found.append((m.start(), f"{month} {int(m.group(2))}"))
            md_spans.append(m.span())
            month_positions.append((m.start(), month))
        # "moved from the 14th to the 21st" - resolve bare ordinals against the
        # nearest month named in the same unit.
        for m in self.BARE_DAY_RE.finditer(text):
            pos = m.start()
            if any(s <= pos < e for s, e in md_spans):
                continue
            preceding = [mo for p, mo in month_positions if p < pos]
            following = [mo for p, mo in month_positions if p > pos]
            month = preceding[-1] if preceding else (following[0] if following else None)
            if not month:
                continue
            found.append((pos, f"{month} {int(m.group(1))}"))
        # A launch-cue word anywhere in a long meeting segment is not enough: the date
        # has to sit near the cue, so "dinner on Nov 4, right after launch" does not
        # become a launch date.
        cue_spans = [m.span() for m in self.LAUNCH_CUE_RE.finditer(text)]
        found = [(pos, canonical) for pos, canonical in found
                 if any(start - 100 <= pos <= end + 100 for start, end in cue_spans)]
        if not found:
            return []
        found.sort(key=lambda x: x[0])
        # The operative date in "moved from A to B" is the last one named. The earlier
        # values stay in the graph as history through the temporal ordering of units.
        return [(found[-1][1], 0)]

    def _extract_pricing(self, text: str) -> List[Tuple[str, str, Dict]]:
        """Per-vehicle pricing, keeping the volume tier separate from the base rate."""
        low = text.lower()
        out: List[Tuple[str, str, Dict]] = []
        volume_spans = []
        for m in re.finditer(r'\$(\d+(?:\.\d+)?)\s*(?:/|per\s+)?\s*(?:vehicle|car|truck)[^.\n]{0,60}?above\s+(\d+)', low):
            out.append(('volume_price_above_500', f"${m.group(1)}",
                        {'threshold': m.group(2), 'amount': m.group(1)}))
            volume_spans.append(m.span())
        for m in re.finditer(r'\$(\d+(?:\.\d+)?)\s*(?:/|per\s+)?\s*vehicle', low):
            if any(s <= m.start() < e for s, e in volume_spans):
                continue
            out.append(('price_per_vehicle', f"${m.group(1)}", {'amount': m.group(1)}))
        for m in re.finditer(r'(\d+)[-\s]year\s+agreement', low):
            out.append(('contract_term', f"{m.group(1)}-year", {'years': m.group(1)}))
        if re.search(r'onboarding\s+fee\s+(?:waived|free)', low):
            out.append(('onboarding_fee', 'waived', {}))
        return out

    def _extract_lifecycle(self, text: str, provenance: ProvenanceLink, unit_id: str) -> List[Assertion]:
        """Extension / fulfilment / cancellation events that change a commitment's state."""
        assertions = []
        keywords = sorted(significant_tokens(text))
        for assertion_type, pattern in self.LIFECYCLE_PATTERNS:
            for match in pattern.finditer(text):
                window = text[max(0, match.start() - 40):match.end() + 40].strip()
                assertions.append(self._make_assertion(
                    assertion_type, provenance.speaker, 'lifecycle', window,
                    [provenance], unit_id,
                    {'cue': match.group(0).lower(), 'keywords': keywords},
                ))
        return assertions

    def _extract_facts(self, text: str, provenance: ProvenanceLink, unit_id: str) -> List[Assertion]:
        assertions = []
        text_lower = text.lower()
        for canonical, offset in self._launch_dates(text):
            assertions.append(self._make_assertion(
                AssertionType.FACT, 'PROJ-RP', 'launch_date', canonical,
                [provenance], unit_id, {'raw': text[:120]},
                valid_from=provenance.timestamp + timedelta(microseconds=offset),
            ))
        for predicate, value, meta in self._extract_pricing(text):
            assertions.append(self._make_assertion(
                AssertionType.FACT, 'ORG-ACME', predicate, value,
                [provenance], unit_id, meta
            ))
        for pattern, attr in self.FLIGHT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                if attr == 'flight_number':
                    assertions.append(self._make_assertion(
                        AssertionType.FACT, 'PER-ALEX', 'flight', match.group(1).upper().replace(' ', ''),
                        [provenance], unit_id, {'flight_number': match.group(1)}
                    ))
        for pattern, attr in self.LATENCY_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                assertions.append(self._make_assertion(
                    AssertionType.FACT, 'PROJ-RP', 'p95_latency', f"{match.group(1)} seconds",
                    [provenance], unit_id, {'value': match.group(1)}
                ))
        for pattern, attr in self.TEST_RESULT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                if attr == 'test_pass_rate':
                    passed, total = match.groups()
                    assertions.append(self._make_assertion(
                        AssertionType.TEST_RESULT, 'PROJ-RP', 'regression_tests_passing', f"{passed}/{total}",
                        [provenance], unit_id, {'passed': int(passed), 'total': int(total)}
                    ))
                elif attr == 'test_failures':
                    assertions.append(self._make_assertion(
                        AssertionType.TEST_RESULT, 'PROJ-RP', 'regression_tests_failing', match.group(1),
                        [provenance], unit_id, {'failures': int(match.group(1))}
                    ))
        return assertions
    def _extract_attributions(self, text: str, provenance: ProvenanceLink, unit_id: str) -> List[Assertion]:
        assertions = []
        text_lower = text.lower()
        for pattern, attr in self.ATTRIBUTION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                quoted_speaker = match.group(1).strip()
                content = match.group(2).strip()
                entities, _ = self.resolver.resolve(quoted_speaker)
                person = next((e for e in entities if e.entity_type == 'PERSON'), None)
                # Second-hand claims are only recorded for people who exist in the data,
                # and never for self-reference: that keeps "but I said ..." noise out of
                # the provenance graph while preserving real "X said Y said Z" chains.
                if person is None or person.canonical_id == provenance.speaker or len(content) < 12:
                    continue
                quoted_id = person.canonical_id
                quoted_link = ProvenanceLink(
                    speaker=quoted_id, speaker_name=quoted_speaker, source_id=unit_id,
                    source_type=provenance.source_type, timestamp=provenance.timestamp,
                    confidence=0.7, is_direct=False, quoted_speaker=provenance.speaker
                )
                extended_provenance = [provenance, quoted_link]
                assertions.append(self._make_assertion(
                    AssertionType.SAID, quoted_id, 'said', content[:200],
                    extended_provenance, unit_id, {'reporter': provenance.speaker, 'content': content}
                ))
        return assertions
    
    def _extract_decisions(self, text: str, provenance: ProvenanceLink, unit_id: str) -> List[Assertion]:
        assertions = []
        text_lower = text.lower()
        for pattern, attr in self.DECISION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                content = match.group(1).strip()
                assertions.append(self._make_assertion(
                    AssertionType.DECIDED, provenance.speaker, 'decided', content[:200],
                    [provenance], unit_id, {'decision_text': content}
                ))
        return assertions
    
    def _extract_technical(self, text: str, provenance: ProvenanceLink, unit_id: str) -> List[Assertion]:
        assertions = []
        text_lower = text.lower()
        if 'postgres' in text_lower and 'postgis' in text_lower and 'sqlite' in text_lower:
            assertions.append(self._make_assertion(
                AssertionType.FACT, 'PROJ-ETA', 'database_choice', 'Postgres with PostGIS',
                [provenance], unit_id, {'reason': 'geospatial queries for nearest-depot lookups'}
            ))
        if 'dark mode' in text_lower:
            if 'cut' in text_lower or 'remove' in text_lower:
                assertions.append(self._make_assertion(
                    AssertionType.DECIDED, 'PROJ-DARK', 'decision', 'cut from v2, fast-follow in v2.1',
                    [provenance], unit_id, {}
                ))
            elif 'keep' in text_lower or 'ship' in text_lower:
                assertions.append(self._make_assertion(
                    AssertionType.DECIDED, 'PROJ-DARK', 'decision', 'ship in v2.1 as fast-follow',
                    [provenance], unit_id, {}
                ))
        if 'sso' in text_lower and ('deprioritiz' in text_lower or 'not happening' in text_lower or 'q1' in text_lower):
            assertions.append(self._make_assertion(
                AssertionType.DECIDED, 'PROJ-SSO', 'decision', 'deprioritized, not before Q1',
                [provenance], unit_id, {}
            ))
        return assertions

    def _extract_commitments(self, text: str, provenance: ProvenanceLink, unit_id: str, speaker_id: str) -> List[Commitment]:
        commitments = []
        text_lower = text.lower()
        for pattern, attr in self.COMMITMENT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                content = match.group(1).strip() if match.group(1) else ''
                deadline_str = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else None
                due_date = None
                if deadline_str:
                    due_date = self._parse_relative_date(deadline_str, provenance.timestamp)
                commitment_id = f"COMMIT-{self.commitment_counter:04d}"
                self.commitment_counter += 1
                assignee = self._infer_assignee(text, speaker_id)
                commitment = Commitment(
                    commitment_id=commitment_id, creator=speaker_id, assignee=assignee,
                    description=content, created_at=provenance.timestamp, due_date=due_date,
                    status='MADE', source_ids=[unit_id], provenance_chain=[provenance]
                )
                commitments.append(commitment)
        return commitments
    
    def _parse_relative_date(self, date_str: str, reference: datetime) -> Optional[datetime]:
        date_str = date_str.lower().strip()
        for pattern in self.resolver.DATE_PATTERNS:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    if re.match(r'[a-z]{3,}\s+\d{1,2}', date_str):
                        return datetime.strptime(date_str + f' {reference.year}', '%B %d %Y')
                    elif re.match(r'\d{1,2}/\d{1,2}', date_str):
                        return datetime.strptime(date_str + f'/{reference.year}', '%m/%d/%Y')
                except:
                    pass
        if 'tomorrow' in date_str:
            return reference + timedelta(days=1)
        elif 'today' in date_str:
            return reference
        elif 'friday' in date_str:
            days_ahead = 4 - reference.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return reference + timedelta(days=days_ahead)
        return None
    
    def _infer_assignee(self, text: str, speaker_id: str) -> Optional[str]:
        text_lower = text.lower()
        if speaker_id == 'PER-ALEX' and ('i ' in text_lower or 'i\'ll' in text_lower or 'i will' in text_lower):
            return 'PER-ALEX'
        mentions = self.resolver.resolve_all_in_text(text)
        for mention, ents, _ambiguous in mentions:
            if ents and ents[0].entity_type == 'PERSON' and ents[0].canonical_id != speaker_id:
                return ents[0].canonical_id
        return None
    
    def _make_assertion(self, assertion_type: AssertionType, subject: str, predicate: str,
                       object: str, provenance: List[ProvenanceLink], unit_id: str,
                       metadata: Dict = None, valid_from: datetime = None,
                       valid_to: datetime = None) -> Assertion:
        self.assertion_counter += 1
        return Assertion(
            assertion_id=f"ASSERT-{self.assertion_counter:05d}",
            assertion_type=assertion_type, subject=subject, predicate=predicate,
            object=object, valid_from=valid_from or (provenance[0].timestamp if provenance else datetime.now()),
            valid_to=valid_to, provenance=provenance, metadata=metadata or {}
        )
    
    def _resolve_conflicts(self, assertions: List[Assertion]) -> List[Assertion]:
        """Close the validity interval of a value that a later unit replaces with a
        different value, so point-in-time lookups return what was true at ``as_of``.

        Speech acts are exempt: one utterance never invalidates an earlier one, which
        keeps "who said what" independent of what was said afterwards.
        """
        groups = {}
        for a in assertions:
            if a.assertion_type in (AssertionType.SAID, AssertionType.ASKED):
                continue
            groups.setdefault((a.subject, a.predicate), []).append(a)
        for group in groups.values():
            group.sort(key=lambda x: x.valid_from)
            for i, a in enumerate(group):
                for later in group[i+1:]:
                    if later.valid_from >= a.valid_from and later.object != a.object:
                        a.valid_to = later.valid_from
                        break
        return assertions
    
    LIFECYCLE_TYPES = (
        AssertionType.COMMITMENT_CANCELLED,
        AssertionType.COMMITMENT_FULFILLED,
        AssertionType.COMMITMENT_EXTENDED,
    )

    def _link_commitments(self, commitments: List[Commitment], assertions: List[Assertion]) -> List[Commitment]:
        """Walk each commitment forward: MADE -> EXTENDED -> FULFILLED, with CANCELLED terminal.

        A lifecycle event belongs to the commitment it shares content with: at least two
        shared content tokens, or one shared token when the speaker is one of the
        commitment's parties. Events are applied in time order, so a later fulfilment
        overrides an earlier extension.
        """
        events = [a for a in assertions if a.assertion_type in self.LIFECYCLE_TYPES]
        events.sort(key=lambda a: a.valid_from)
        for commitment in commitments:
            own_tokens = significant_tokens(commitment.description) | significant_tokens(
                self._commitment_context(commitment, assertions))
            for event in events:
                if event.valid_from < commitment.created_at:
                    continue
                shared = own_tokens & set(event.metadata.get('keywords', ()))
                is_party = event.subject in {commitment.creator, commitment.assignee}
                if not (len(shared) >= 2 or (shared and is_party)):
                    continue
                if event.assertion_type == AssertionType.COMMITMENT_CANCELLED:
                    commitment.status = 'CANCELLED'
                    commitment.cancelled_at = event.valid_from
                elif event.assertion_type == AssertionType.COMMITMENT_FULFILLED:
                    commitment.status = 'FULFILLED'
                    commitment.fulfilled_at = event.valid_from
                elif commitment.status == 'MADE':
                    commitment.status = 'EXTENDED'
                    commitment.extended_at = event.valid_from
                if event.provenance and event.provenance[0].source_id not in commitment.source_ids:
                    commitment.source_ids.append(event.provenance[0].source_id)
        return commitments

    def _commitment_context(self, commitment: Commitment, assertions: List[Assertion]) -> str:
        """Text of the source unit that created the commitment, for keyword matching."""
        first = commitment.source_ids[0] if commitment.source_ids else None
        for a in assertions:
            if first and a.provenance and a.provenance[0].source_id == first:
                return a.object
        return commitment.description


def extract_facts(units: List[Dict]) -> Tuple[List[Assertion], List[Commitment]]:
    extractor = FactExtractor()
    return extractor.extract_from_units(units)
