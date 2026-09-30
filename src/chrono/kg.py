"""
Temporal Knowledge Graph — Core data structure for Chronos.

Stores entities, assertions with validity intervals, provenance chains,
and commitment lifecycles. Supports point-in-time queries.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .extract import Assertion, Commitment, AssertionType, ProvenanceLink, significant_tokens
from .entities import Entity, EntityResolver, get_resolver


@dataclass
class TemporalEdge:
    """An edge in the temporal knowledge graph with validity interval."""
    edge_id: str
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    assertion_type: AssertionType = AssertionType.FACT
    provenance: List[ProvenanceLink] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid_at(self, as_of: datetime) -> bool:
        if self.valid_from > as_of:
            return False
        if self.valid_to is not None and self.valid_to <= as_of:
            return False
        return True
    
    def overlaps(self, other: 'TemporalEdge') -> bool:
        if self.subject != other.subject or self.predicate != other.predicate:
            return False
        self_end = self.valid_to or datetime.max
        other_end = other.valid_to or datetime.max
        return not (self.valid_from >= other_end or other.valid_from >= self_end)

class TemporalKG:
    """Temporal Knowledge Graph with point-in-time query support."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.edges: Dict[str, TemporalEdge] = {}
        self.edges_by_subject: Dict[str, List[str]] = defaultdict(list)
        self.edges_by_predicate: Dict[str, List[str]] = defaultdict(list)
        self.edges_by_object: Dict[str, List[str]] = defaultdict(list)
        self.commitments: Dict[str, Commitment] = {}
        self.edge_counter = 0
        self.resolver = get_resolver()
        self._load_entities_from_resolver()
    
    def _load_entities_from_resolver(self):
        for entity in self.resolver.all_entities():
            self.entities[entity.canonical_id] = entity
    
    def add_assertion(self, assertion: Assertion) -> str:
        self.edge_counter += 1
        edge_id = f"EDGE-{self.edge_counter:06d}"
        edge = TemporalEdge(
            edge_id=edge_id, subject=assertion.subject, predicate=assertion.predicate,
            object=assertion.object, valid_from=assertion.valid_from, valid_to=assertion.valid_to,
            assertion_type=assertion.assertion_type, provenance=assertion.provenance,
            confidence=assertion.confidence, metadata=assertion.metadata
        )
        self.edges[edge_id] = edge
        self.edges_by_subject[assertion.subject].append(edge_id)
        self.edges_by_predicate[assertion.predicate].append(edge_id)
        self.edges_by_object[assertion.object].append(edge_id)
        if assertion.subject not in self.entities:
            self.entities[assertion.subject] = Entity(canonical_id=assertion.subject, entity_type='UNKNOWN')
        return edge_id
    
    def add_commitment(self, commitment: Commitment):
        self.commitments[commitment.commitment_id] = commitment
        self.add_assertion(Assertion(
            assertion_id=f"ASSERT-COMMIT-{commitment.commitment_id}",
            assertion_type=AssertionType.COMMITMENT_MADE, subject=commitment.creator,
            predicate='made_commitment', object=commitment.commitment_id,
            valid_from=commitment.created_at, provenance=commitment.provenance_chain,
            metadata={'description': commitment.description, 'due_date': commitment.due_date}
        ))
        if commitment.assignee:
            self.add_assertion(Assertion(
                assertion_id=f"ASSERT-ASSIGN-{commitment.commitment_id}",
                assertion_type=AssertionType.COMMITMENT_MADE, subject=commitment.assignee,
                predicate='assigned_commitment', object=commitment.commitment_id,
                valid_from=commitment.created_at, provenance=commitment.provenance_chain,
                metadata={'description': commitment.description}
            ))
    
    def build_from_extractions(self, assertions: List[Assertion], commitments: List[Commitment]):
        for assertion in assertions:
            self.add_assertion(assertion)
        for commitment in commitments:
            self.add_commitment(commitment)
    
    def get_current_value(self, subject: str, predicate: str, as_of: datetime) -> Optional[TemporalEdge]:
        edges = self.edges_by_subject.get(subject, [])
        candidates = []
        for edge_id in edges:
            edge = self.edges[edge_id]
            if edge.predicate == predicate and edge.is_valid_at(as_of):
                candidates.append(edge)
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.valid_from)
    
    def get_value_at(self, subject: str, predicate: str, target_time: datetime) -> Optional[TemporalEdge]:
        return self.get_current_value(subject, predicate, target_time)
    
    def get_timeline(self, subject: str, predicate: str) -> List[TemporalEdge]:
        edges = self.edges_by_subject.get(subject, [])
        candidates = [self.edges[eid] for eid in edges if self.edges[eid].predicate == predicate]
        return sorted(candidates, key=lambda e: e.valid_from)
    
    def get_provenance_chain(self, subject: str, predicate: str, as_of: datetime) -> List[ProvenanceLink]:
        edge = self.get_current_value(subject, predicate, as_of)
        if not edge:
            return []
        chain = []
        for link in edge.provenance:
            chain.append(link)
        return chain
    
    def get_commitment_status(self, commitment_id: str, as_of: datetime) -> Optional[str]:
        commitment = self.commitments.get(commitment_id)
        if not commitment:
            return None
        if commitment.status == 'FULFILLED' and commitment.fulfilled_at and commitment.fulfilled_at <= as_of:
            return 'FULFILLED'
        elif commitment.status == 'EXTENDED' and commitment.extended_at and commitment.extended_at <= as_of:
            return 'EXTENDED'
        elif commitment.status == 'CANCELLED' and commitment.cancelled_at and commitment.cancelled_at <= as_of:
            return 'CANCELLED'
        elif commitment.created_at <= as_of:
            return 'MADE'
        return None
    
    def get_commitments_by_assignee(self, assignee: str, as_of: datetime) -> List[Commitment]:
        results = []
        for commitment in self.commitments.values():
            if commitment.assignee == assignee and commitment.created_at <= as_of:
                results.append(commitment)
        return results
    
    def get_commitments_by_creator(self, creator: str, as_of: datetime) -> List[Commitment]:
        results = []
        for commitment in self.commitments.values():
            if commitment.creator == creator and commitment.created_at <= as_of:
                results.append(commitment)
        return results
    
    def who_said_what(self, person: str, topic: str = None, as_of: datetime = None) -> List[TemporalEdge]:
        edges = self.edges_by_subject.get(person, [])
        topic_tokens = significant_tokens(topic) if topic else set()
        results = []
        for edge_id in edges:
            edge = self.edges[edge_id]
            if as_of and not edge.is_valid_at(as_of):
                continue
            # Topic match is token overlap, so "cutting dark mode" still matches a
            # sentence that says "he was fine cutting it".
            if topic_tokens and not (topic_tokens & significant_tokens(edge.object)):
                continue
            if edge.assertion_type == AssertionType.SAID or topic_tokens:
                results.append(edge)
        return sorted(results, key=lambda e: e.valid_from, reverse=True)
    
    def query(self, subject: str = None, predicate: str = None, object: str = None,
              as_of: datetime = None, assertion_type: AssertionType = None) -> List[TemporalEdge]:
        candidate_ids = set(self.edges.keys())
        if subject:
            candidate_ids &= set(self.edges_by_subject.get(subject, []))
        if predicate:
            candidate_ids &= set(self.edges_by_predicate.get(predicate, []))
        if object:
            candidate_ids &= set(self.edges_by_object.get(object, []))
        results = []
        for edge_id in candidate_ids:
            edge = self.edges[edge_id]
            if as_of and not edge.is_valid_at(as_of):
                continue
            if assertion_type and edge.assertion_type != assertion_type:
                continue
            results.append(edge)
        return sorted(results, key=lambda e: e.valid_from, reverse=True)
    
    def resolve_entity(self, mention: str) -> Tuple[List[Entity], bool]:
        return self.resolver.resolve(mention)
    
    def get_ambiguity_clarification(self, mention: str) -> Optional[str]:
        entities, is_ambiguous = self.resolver.resolve(mention)
        if is_ambiguous and entities:
            cluster_id = entities[0].ambiguity_cluster
            if cluster_id:
                return self.resolver.get_clarification_question(cluster_id)
        return None
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def all_entities(self) -> List[Entity]:
        return list(self.entities.values())
    
    # ------------------------------------------------------------------ belief

    def belief(self, subject: str, predicate: str, as_of: datetime) -> Optional[TemporalEdge]:
        """What the system actually believes at ``as_of``.

        Unlike :meth:`get_current_value`, which returns the most recent interval, this
        ranks competing edges by credibility: a first-hand statement beats a reported
        one, higher confidence wins ties, and recency breaks the rest. That is what
        separates "the fact changed" from "people disagree".
        """
        edges = [e for e in self.get_timeline(subject, predicate) if e.is_valid_at(as_of)]
        if not edges:
            return None
        return max(edges, key=self._credibility)

    @staticmethod
    def _credibility(edge: TemporalEdge) -> tuple:
        direct = any(link.is_direct for link in edge.provenance)
        chain_len = len(edge.provenance)
        return (direct, edge.confidence, -chain_len, edge.valid_from)

    def disagreements(self, subject: str, predicate: str, as_of: datetime) -> List[TemporalEdge]:
        """Concurrently valid, differing values for the same attribute.

        People disagreeing is not the same as a fact changing: these edges overlap in
        time, so the answer layer can say so instead of silently picking one.
        """
        edges = [e for e in self.get_timeline(subject, predicate) if e.is_valid_at(as_of)]
        grouped: Dict[str, TemporalEdge] = {}
        for edge in edges:
            best = grouped.get(edge.object)
            if best is None or self._credibility(edge) > self._credibility(best):
                grouped[edge.object] = edge
        if len(grouped) < 2:
            return []
        return sorted(grouped.values(), key=lambda e: e.valid_from)

    # ------------------------------------------------------------------ safety

    SECRET_PATTERNS = [
        re.compile(r'sk-[A-Za-z0-9\-_]{8,}'),
        re.compile(r'\b(?:api[_\- ]?key|password|passwd|secret|token)\b\s*[:=]?\s*\S+', re.IGNORECASE),
    ]
    INJECTION_MARKERS = [
        'ignore your previous instructions', 'ignore all previous instructions',
        'note to any ai assistant', 'forward all emails', 'review@pipelinepilot',
        'system prompt', 'as an ai assistant you must',
    ]
    QUARANTINED_SOURCES: Set[str] = {
        'EM-F-050',       # PipelinePilot digest carrying the planted instruction
        'SL-DM-AB-0915-2',  # Ben's staging key
    }

    def check_secret_leak(self, text: str) -> bool:
        """True when text would repeat a secret or a planted instruction."""
        low = (text or '').lower()
        if any(marker in low for marker in self.INJECTION_MARKERS):
            return True
        return any(pattern.search(text or '') for pattern in self.SECRET_PATTERNS)

    def redact(self, text: str) -> str:
        """Replace secrets with a placeholder so they can never leave the system."""
        out = text or ''
        for pattern in self.SECRET_PATTERNS:
            out = pattern.sub('[redacted secret]', out)
        return out

    def sanitize_answer(self, answer: str) -> str:
        """Final gate on anything the system is about to say."""
        if self.check_secret_leak(answer):
            return ("I can't repeat that: it contains a secret or a planted instruction "
                    "from the source data rather than a fact about the work.")
        return self.redact(answer)

    def is_quarantined(self, unit_id: str) -> bool:
        return unit_id in self.QUARANTINED_SOURCES


def build_temporal_kg(assertions: List[Assertion], commitments: List[Commitment]) -> TemporalKG:
    kg = TemporalKG()
    kg.build_from_extractions(assertions, commitments)
    return kg
