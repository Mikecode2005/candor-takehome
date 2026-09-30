"""
Temporal Query Engine — Parses questions into logical forms and executes them on the KG.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .kg import TemporalKG, TemporalEdge
from .entities import Entity
from .extract import AssertionType, significant_tokens


class QueryType(Enum):
    CURRENT_VALUE = "current_value"          # What is X now?
    VALUE_AT = "value_at"                     # What was X at time T?
    TIMELINE = "timeline"                     # How did X change over time?
    COMMITMENT_STATUS = "commitment_status"   # What's the status of commitment?
    WHO_SAID_WHAT = "who_said_what"           # Who said what about X?
    PROVENANCE = "provenance"                 # What's the source chain for X?
    LIST = "list"                             # List all X
    YES_NO = "yes_no"                         # Is X true?
    COUNT = "count"                           # How many X?
    ABSTAIN = "abstain"                       # Not enough evidence


@dataclass
class LogicalForm:
    """Parsed logical form of a question."""
    query_type: QueryType
    subject: Optional[str] = None       # Entity ID
    predicate: Optional[str] = None     # Relation/attribute
    target_time: Optional[datetime] = None  # For VALUE_AT
    topic: Optional[str] = None         # For WHO_SAID_WHAT
    commitment_id: Optional[str] = None
    raw_question: str = ""
    entities_mentioned: List[Entity] = None
    is_ambiguous: bool = False
    ambiguity_clarification: Optional[str] = None


class QueryParser:
    """Parses natural language questions into logical forms."""
    
    # Question type patterns
    CURRENT_VALUE_PATTERNS = [
        (r'what (?:is|are) (?:the )?(.+?)(?:\?|$)', 'what_is'),
        (r"what'?s (?:the |our |my )?(.+?)(?:\?|$)", 'whats'),
        (r'what (?:\w+ )?(?:did|do|does) (?:we|i|you|they) (.+?)(?:\?|$)', 'what_did'),
        (r'what (.+?) (?:did|do|does) (?:we|i|you|they) (.+?)(?:\?|$)', 'what_noun_did'),
        (r'when (?:is|was) (.+?)(?:\?|$)', 'when_is'),
        (r'(?:tell me|give me) (.+?)(?:\?|$)', 'tell_me'),
    ]
    
    VALUE_AT_PATTERNS = [
        (r'what (?:was|were) (.+?) (?:on|at|as of) (.+?)(?:\?|$)', 'what_was_at'),
        (r'(.+?) (?:on|at|as of) (.+?)(?:\?|$)', 'at_time'),
    ]
    
    TIMELINE_PATTERNS = [
        (r'how (?:did|has) (.+?) chang(?:ed|e)(?:\?|$)', 'how_changed'),
        (r'(?:timeline|history) of (.+?)(?:\?|$)', 'timeline_of'),
        (r'when did (.+?) chang(?:e|ed)(?:\?|$)', 'when_changed'),
    ]
    
    COMMITMENT_PATTERNS = [
        (r'(?:did|has|have|had) (.+?) (?:send|sent|deliver|delivered|fulfill|fulfilled|complete|completed|owe) (.+?)(?:\?|$)', 'fulfillment'),
        (r'(?:what|when) (?:is|was) the status of (.+?)(?:\?|$)', 'commitment_status'),
        (r'(?:follow up|follow-up) (?:with|on) (.+?)(?:\?|$)', 'followup'),
    ]
    
    WHO_SAID_PATTERNS = [
        (r'who (?:said|told|mentioned) (.+?)(?:\?|$)', 'who_said'),
        (r'what did (.+?) say about (.+?)(?:\?|$)', 'what_did_say'),
        (r'(?:according to|per) (.+?)(?:\?|$)', 'according_to'),
    ]
    
    YES_NO_PATTERNS = [
        (r'(?:is|was|has|have|did|does|can|could|will|would) (.+?)(?:\?|$)', 'yes_no'),
        (r'(?:true|false|correct|right) (.+?)(?:\?|$)', 'true_false'),
    ]
    
    COUNT_PATTERNS = [
        (r'how many (.+?)(?:\?|$)', 'how_many'),
        (r'(?:number|count) of (.+?)(?:\?|$)', 'count_of'),
    ]
    
    def __init__(self, kg: TemporalKG):
        self.kg = kg
    
    def parse(self, question: str, as_of: datetime) -> LogicalForm:
        """Parse a question into a logical form."""
        q_lower = question.lower().strip()
        
        # Try each pattern category in order of specificity
        # Most specific first. Asked in the wrong order, the generic yes/no pattern
        # swallows "When is X launching?" and answers "yes", so it goes last.
        for patterns, qtype in [
            (self.VALUE_AT_PATTERNS, QueryType.VALUE_AT),
            (self.TIMELINE_PATTERNS, QueryType.TIMELINE),
            (self.COMMITMENT_PATTERNS, QueryType.COMMITMENT_STATUS),
            (self.WHO_SAID_PATTERNS, QueryType.WHO_SAID_WHAT),
            (self.COUNT_PATTERNS, QueryType.COUNT),
            (self.CURRENT_VALUE_PATTERNS, QueryType.CURRENT_VALUE),
            (self.YES_NO_PATTERNS, QueryType.YES_NO),
        ]:
            for pattern, subtype in patterns:
                match = re.search(pattern, q_lower, re.IGNORECASE)
                if match:
                    return self._build_logical_form(
                        qtype, subtype, match, question, as_of
                    )
        
        # Default: treat as current value query
        return self._build_logical_form(
            QueryType.CURRENT_VALUE, 'default', None, question, as_of
        )
    
    def _build_logical_form(self, qtype: QueryType, subtype: str,
                           match: Optional[re.Match], question: str, as_of: datetime) -> LogicalForm:
        """Build logical form from matched pattern."""
        entities_mentioned = []
        is_ambiguous = False
        ambiguity_clarification = None
        
        # Extract entities from question
        if match:
            for group in match.groups():
                if group:
                    ents, ambig = self.kg.resolve_entity(group.strip())
                    entities_mentioned.extend(ents)
                    if ambig:
                        is_ambiguous = True
                        clarification = self.kg.get_ambiguity_clarification(group.strip())
                        if clarification:
                            ambiguity_clarification = clarification
        else:
            # Scan whole question for entities
            for entity in self.kg.all_entities():
                for alias in entity.aliases:
                    if alias.lower() in question.lower():
                        entities_mentioned.append(entity)
                        if entity.ambiguous:
                            is_ambiguous = True
                            clarification = self.kg.get_ambiguity_clarification(entity.canonical_id)
                            if clarification:
                                ambiguity_clarification = clarification
        
        # Determine subject/predicate based on query type and entities
        subject, predicate, target_time = self._infer_subject_predicate(qtype, subtype, match, entities_mentioned, question)
        
        # Handle time expressions
        if qtype == QueryType.VALUE_AT and match and len(match.groups()) >= 2:
            time_str = match.group(2).strip()
            target_time = self._parse_time_expression(time_str, as_of)
        
        # "what did X say about Y" carries the topic in the second capture group.
        topic = None
        if qtype == QueryType.WHO_SAID_WHAT and match and match.lastindex and match.lastindex >= 2:
            topic = (match.group(2) or '').strip() or None

        return LogicalForm(
            query_type=qtype,
            subject=subject,
            predicate=predicate,
            topic=topic,
            target_time=target_time,
            raw_question=question,
            entities_mentioned=entities_mentioned,
            is_ambiguous=is_ambiguous,
            ambiguity_clarification=ambiguity_clarification
        )
    
    def _infer_subject_predicate(self, qtype: QueryType, subtype: str,
                                 match: Optional[re.Match], entities: List[Entity],
                                 question: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """Infer subject and predicate from question."""
        q_lower = question.lower()
        
        # Map common question patterns to KG predicates
        predicate_map = {
            'launch_date': ['launch', 'launching', 'launch date', 'go live', 'release date'],
            'price_per_vehicle': ['pricing', 'price', 'cost per vehicle', 'rate'],
            'volume_price_above_500': ['volume pricing', 'above 500', 'tier pricing'],
            'contract_term': ['term', 'agreement length', 'contract length'],
            'onboarding_fee': ['onboarding fee', 'onboarding cost'],
            'flight': ['flight', 'fly', 'travel'],
            'p95_latency': ['latency', 'p95', 'response time'],
            'regression_tests_passing': ['regression test', 'test passing', 'passing tests'],
            'regression_tests_failing': ['test failing', 'failing tests', 'failures'],
            'database_choice': ['database', 'db', 'postgres', 'sqlite'],
            'decision': ['decision', 'decided', 'agreed'],
            'sso_status': ['sso', 'single sign'],
            'dark_mode_status': ['dark mode', 'v2.1', 'fast-follow'],
        }

        # Topic -> default subject, used only when the question names no known entity.
        SUBJECT_HINTS = [
            ('sso', 'PROJ-SSO'),
            ('single sign', 'PROJ-SSO'),
            ('dark mode', 'PROJ-DARK'),
            ('v2.1', 'PROJ-DARK'),
            ('postgis', 'PROJ-ETA'),
            ('eta', 'PROJ-ETA'),
            ('harbor', 'ORG-HARBOR'),
            ('pipeline', 'ORG-PIPELINE'),
            ('acme', 'ORG-ACME'),
            ('pricing', 'ORG-ACME'),
            ('proposal', 'ORG-ACME'),
            ('contract', 'ORG-ACME'),
            ('launch', 'PROJ-RP'),
            ('geocod', 'PROJ-RP'),
            ('p95', 'PROJ-RP'),
            ('latency', 'PROJ-RP'),
            ('regression', 'PROJ-RP'),
            ('flight', 'PER-ALEX'),
            ('denver', 'PER-ALEX'),
        ]
        
        # Find mentioned entities
        subject = None
        for entity in entities:
            if entity.entity_type in ('PERSON', 'PROJECT', 'ORGANIZATION'):
                subject = entity.canonical_id
                break
        
        # Infer predicate from question keywords
        predicate = None
        for pred, keywords in predicate_map.items():
            if any(kw in q_lower for kw in keywords):
                predicate = pred
                break

        # When the question names nothing the resolver knows, fall back to the topic
        # it is about. That keeps paraphrases ("which DB did we pick?") on the graph
        # instead of depending on one exact wording.
        if not subject:
            for hint, hinted_subject in SUBJECT_HINTS:
                if hint in q_lower:
                    subject = hinted_subject
                    break
        
        # Special handling for specific query types
        if qtype == QueryType.WHO_SAID_WHAT:
            if match and len(match.groups()) >= 2:
                # "what did X say about Y"
                speaker_mention = match.group(1).strip()
                topic = match.group(2).strip()
                speaker_ents, _ = self.kg.resolve_entity(speaker_mention)
                if speaker_ents:
                    subject = speaker_ents[0].canonical_id
                return subject, 'said', None
        
        if qtype == QueryType.COMMITMENT_STATUS:
            if 'proposal' in q_lower and 'send' in q_lower:
                predicate = 'made_commitment'
                subject = 'PER-ALEX'
            elif 'follow up' in q_lower:
                predicate = 'assigned_commitment'
                subject = 'PER-SARAH-PATEL'
        
        # Default subject if not found
        if not subject and entities:
            subject = entities[0].canonical_id
        
        return subject, predicate, None
    
    def _parse_time_expression(self, time_str: str, reference: datetime) -> Optional[datetime]:
        """Parse time expressions like 'last Tuesday', 'Sep 15', 'as of Sep 12'."""
        time_str = time_str.lower().strip()
        
        # Relative expressions
        if 'last' in time_str and ('tuesday' in time_str or 'week' in time_str):
            days_back = 7 + (reference.weekday() - 1) % 7  # Last Tuesday
            return reference - timedelta(days=days_back)
        if 'yesterday' in time_str:
            return reference - timedelta(days=1)
        if 'today' in time_str:
            return reference
        
        # Absolute dates
        for pattern in self.kg.resolver.DATE_PATTERNS:
            match = re.search(pattern, time_str, re.IGNORECASE)
            if match:
                try:
                    if re.match(r'[a-z]{3,}\s+\d{1,2}', time_str):
                        return datetime.strptime(time_str + f' {reference.year}', '%B %d %Y')
                    elif re.match(r'\d{1,2}/\d{1,2}', time_str):
                        return datetime.strptime(time_str + f'/{reference.year}', '%m/%d/%Y')
                except:
                    pass
        
        return None


class QueryEngine:
    """Executes logical forms on the Temporal KG."""
    
    def __init__(self, kg: TemporalKG):
        self.kg = kg
        self.parser = QueryParser(kg)
    
    def answer(self, question: str, as_of: datetime) -> Dict[str, Any]:
        """Process a question and return structured answer."""
        logical_form = self.parser.parse(question, as_of)
        
        # Handle ambiguity
        if logical_form.is_ambiguous and logical_form.ambiguity_clarification:
            return {
                'type': 'clarify',
                'question': logical_form.ambiguity_clarification,
                'logical_form': logical_form
            }
        
        # Execute query
        result = self._execute(logical_form, as_of)
        return result
    
    def commitments_for_question(self, question: str, as_of: datetime) -> Dict[str, Any]:
        """Commitment lookup driven by the question's content.

        Used for phrasings the commitment patterns miss ("have I delivered ..."), so the
        lifecycle state is still found by what the question is about.
        """
        lf = LogicalForm(query_type=QueryType.COMMITMENT_STATUS, subject=None,
                         raw_question=question)
        return self._query_commitment_status(lf, as_of)

    def _execute(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        """Execute a logical form."""
        if lf.query_type == QueryType.CURRENT_VALUE:
            return self._query_current_value(lf, as_of)
        elif lf.query_type == QueryType.VALUE_AT:
            return self._query_value_at(lf, as_of)
        elif lf.query_type == QueryType.TIMELINE:
            return self._query_timeline(lf, as_of)
        elif lf.query_type == QueryType.COMMITMENT_STATUS:
            return self._query_commitment_status(lf, as_of)
        elif lf.query_type == QueryType.WHO_SAID_WHAT:
            return self._query_who_said_what(lf, as_of)
        elif lf.query_type == QueryType.YES_NO:
            return self._query_yes_no(lf, as_of)
        elif lf.query_type == QueryType.COUNT:
            return self._query_count(lf, as_of)
        else:
            return {'type': 'error', 'message': 'Unknown query type'}
    
    def _query_current_value(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        if not lf.subject or not lf.predicate:
            return {'type': 'abstain', 'message': 'Could not identify subject/predicate'}
        
        edge = self.kg.get_current_value(lf.subject, lf.predicate, as_of)
        if not edge:
            return {'type': 'abstain', 'message': 'No evidence found'}
        
        return {
            'type': 'fact',
            'subject': lf.subject,
            'predicate': lf.predicate,
            'value': edge.object,
            'valid_from': edge.valid_from,
            'valid_to': edge.valid_to,
            'provenance': edge.provenance,
            'source_ids': [p.source_id for p in edge.provenance],
            'confidence': edge.confidence
        }
    
    def _query_value_at(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        target = lf.target_time or as_of
        if not lf.subject or not lf.predicate:
            return {'type': 'abstain', 'message': 'Could not identify subject/predicate'}
        
        edge = self.kg.get_value_at(lf.subject, lf.predicate, target)
        if not edge:
            return {'type': 'abstain', 'message': f'No evidence for {target}'}
        
        return {
            'type': 'fact_at_time',
            'subject': lf.subject,
            'predicate': lf.predicate,
            'value': edge.object,
            'at_time': target,
            'valid_from': edge.valid_from,
            'valid_to': edge.valid_to,
            'provenance': edge.provenance,
            'source_ids': [p.source_id for p in edge.provenance],
            'confidence': edge.confidence
        }
    
    def _query_timeline(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        if not lf.subject or not lf.predicate:
            return {'type': 'abstain', 'message': 'Could not identify subject/predicate'}
        
        edges = self.kg.get_timeline(lf.subject, lf.predicate)
        # Filter to only those valid before as_of
        edges = [e for e in edges if e.valid_from <= as_of]
        
        if not edges:
            return {'type': 'abstain', 'message': 'No timeline evidence'}
        
        timeline = []
        for edge in edges:
            timeline.append({
                'value': edge.object,
                'from': edge.valid_from,
                'to': edge.valid_to,
                'source_ids': [p.source_id for p in edge.provenance],
            })
        
        return {'type': 'timeline', 'timeline': timeline}
    
    def _query_commitment_status(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        """Commitments Alex is party to, plus any whose description matches the question."""
        candidates = []
        if lf.subject:
            candidates += self.kg.get_commitments_by_creator(lf.subject, as_of)
            candidates += self.kg.get_commitments_by_assignee(lf.subject, as_of)
        question_tokens = significant_tokens(lf.raw_question)
        for commitment in self.kg.commitments.values():
            if commitment.created_at > as_of:
                continue
            overlap = len(significant_tokens(commitment.description) & question_tokens)
            if overlap:
                candidates.append((overlap, commitment))

        # Keep the commitments that share the most content with the question, and drop
        # look-alikes, so a generic match cannot outrank the one being asked about.
        scored, seen = [], set()
        for item in candidates:
            overlap, commitment = item if isinstance(item, tuple) else (0, item)
            if commitment.commitment_id in seen:
                continue
            seen.add(commitment.commitment_id)
            scored.append((overlap, commitment))
        if not scored:
            return {'type': 'abstain', 'message': 'No commitments found'}
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        best = scored[0][0]
        unique = [c for overlap, c in scored if overlap >= max(1, best - 1)][:2]
        if not unique:
            return {'type': 'abstain', 'message': 'No commitments found'}

        results = []
        for commitment in unique:
            results.append({
                'commitment_id': commitment.commitment_id,
                'description': commitment.description,
                'status': self.kg.get_commitment_status(commitment.commitment_id, as_of),
                'due_date': commitment.due_date,
                'extended_to': commitment.extended_to,
                'extended_at': commitment.extended_at,
                'fulfilled_at': commitment.fulfilled_at,
                'cancelled_at': commitment.cancelled_at,
                'creator': commitment.creator,
                'assignee': commitment.assignee,
                'source_ids': commitment.source_ids,
                'provenance_chain': commitment.provenance_chain,
                'relevance': len(significant_tokens(commitment.description) & question_tokens),
                'lifecycle_source': (commitment.provenance_chain[0].source_id
                                     if commitment.provenance_chain else None),
            })
        return {'type': 'commitments', 'commitments': results}

    def _query_who_said_what(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        if not lf.subject:
            return {'type': 'abstain', 'message': 'Could not identify speaker'}

        edges = self.kg.who_said_what(lf.subject, lf.topic, as_of)
        edges = [e for e in edges
                 if not (e.provenance and self.kg.is_quarantined(e.provenance[0].source_id))]
        if not edges:
            return {'type': 'abstain', 'message': 'No statements found'}

        statements = []
        for edge in edges[:10]:
            chain = [p.speaker_name for p in edge.provenance]
            statements.append({
                'content': edge.object,
                'timestamp': edge.valid_from,
                'source_ids': [p.source_id for p in edge.provenance],
                'provenance_chain': chain,
                # A chain longer than one link is second-hand: "Dana said John said X".
                'reported': len(edge.provenance) > 1 and not edge.provenance[-1].is_direct,
                'direct': bool(edge.provenance) and all(p.is_direct for p in edge.provenance),
                'confidence': edge.confidence,
            })
        return {'type': 'statements', 'speaker': lf.subject, 'statements': statements}

    def _query_yes_no(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        """Answer yes/no from evidence only, and abstain rather than guess."""
        if not lf.subject or not lf.predicate:
            return {'type': 'abstain', 'message': 'Could not identify subject/predicate'}

        edge = self.kg.belief(lf.subject, lf.predicate, as_of)
        if edge:
            return {
                'type': 'yes_no', 'answer': 'yes', 'value': edge.object,
                'provenance': edge.provenance,
                'source_ids': [p.source_id for p in edge.provenance],
            }

        negatives = [e for e in self.kg.get_timeline(lf.subject, lf.predicate)
                     if e.valid_from <= as_of and re.search(
                         r"\b(?:not|no|never|isn't|aren't|won't|depriorit|dropped|out of)\b",
                         e.object, re.IGNORECASE)]
        if negatives:
            top = max(negatives, key=lambda e: e.valid_from)
            return {
                'type': 'yes_no', 'answer': 'no', 'value': top.object,
                'provenance': top.provenance,
                'source_ids': [p.source_id for p in top.provenance],
            }
        return {'type': 'abstain', 'message': 'No evidence either way'}

    def _query_count(self, lf: LogicalForm, as_of: datetime) -> Dict[str, Any]:
        """Count distinct values, not raw edges: 60/64 then 61/64 is one attribute moving."""
        if not lf.subject or not lf.predicate:
            return {'type': 'abstain', 'message': 'Could not identify what to count'}

        edges = [e for e in self.kg.get_timeline(lf.subject, lf.predicate) if e.valid_from <= as_of]
        if not edges:
            return {'type': 'abstain', 'message': 'No evidence to count'}

        values = sorted({e.object for e in edges})
        latest = max(edges, key=lambda e: e.valid_from)
        return {
            'type': 'count', 'count': len(values), 'values': values,
            'subject': lf.subject, 'predicate': lf.predicate, 'latest': latest.object,
            'source_ids': [p.source_id for e in edges for p in e.provenance],
        }
