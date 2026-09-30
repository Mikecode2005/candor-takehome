"""
Entity Resolution with Ambiguity Clusters

Resolves mentions across sources (Slack, email, meetings, calendar, etc.)
into canonical entities with ambiguity tracking.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class Entity:
    """Canonical entity with all known aliases and source references."""
    canonical_id: str
    entity_type: str  # PERSON, PROJECT, DATE, COMMITMENT, DOCUMENT, ORGANIZATION
    aliases: Set[str] = field(default_factory=set)
    source_refs: Dict[str, List[str]] = field(default_factory=dict)  # source -> [ref_ids]
    attributes: Dict[str, str] = field(default_factory=dict)  # email, title, etc.
    ambiguous: bool = False
    ambiguity_cluster: Optional[str] = None  # cluster ID if ambiguous


@dataclass
class AmbiguityCluster:
    """A set of entities that could be confused with each other."""
    cluster_id: str
    entities: List[Entity]
    trigger_terms: Set[str] = field(default_factory=set)  # terms that cause ambiguity


class EntityResolver:
    """
    Resolves entity mentions to canonical entities.
    Tracks ambiguity explicitly for clarification.
    """
    
    PERSON_PATTERNS = {
        'alex rivera': {'id': 'PER-ALEX', 'email': 'alex@brightline.example.com', 'title': 'VP Product'},
        'john okafor': {'id': 'PER-JOHN', 'email': 'john@brightline.example.com', 'title': 'CTO'},
        'sarah kim': {'id': 'PER-SARAH-KIM', 'email': 'sarah.kim@brightline.example.com', 'title': 'Engineering Lead'},
        'sarah patel': {'id': 'PER-SARAH-PATEL', 'email': 'sarah.patel@acmefreight.example.com', 'title': 'Acme Freight'},
        'dana lee': {'id': 'PER-DANA', 'email': 'dana@brightline.example.com', 'title': 'Designer'},
        'marcus webb': {'id': 'PER-MARCUS', 'email': 'marcus@brightline.example.com', 'title': 'Sales'},
        'ben carter': {'id': 'PER-BEN', 'email': 'ben@brightline.example.com', 'title': 'Engineer'},
        'priya nair': {'id': 'PER-PRIYA', 'email': 'priya@brightline.example.com', 'title': 'QA Lead'},
        'leah brooks': {'id': 'PER-LEAH', 'email': 'leah@brightline.example.com', 'title': 'Office Manager'},
        'rachel gomez': {'id': 'PER-RACHEL', 'email': 'rachel@brightline.example.com', 'title': 'Recruiter'},
    }
    
    PROJECT_PATTERNS = {
        'route planner': {'id': 'PROJ-RP', 'aliases': ['route planner v2', 'rp v2', 'v2']},
        'eta predictor': {'id': 'PROJ-ETA', 'aliases': ['eta', 'eta model']},
        'driver eta notifications': {'id': 'PROJ-ETA-NOTIF', 'aliases': ['eta notifications']},
        'dark mode': {'id': 'PROJ-DARK', 'aliases': ['dark mode v2.1']},
        'sso': {'id': 'PROJ-SSO', 'aliases': ['single sign on', 'single sign-on']},
        'harbor': {'id': 'PROJ-HARBOR', 'aliases': []},
        'acme freight': {'id': 'ORG-ACME', 'aliases': ['acme']},
        'pipeline pilot': {'id': 'ORG-PIPELINE', 'aliases': ['pipelinepilot']},
        'pinecrest carriers': {'id': 'ORG-PINECREST', 'aliases': ['pinecrest']},
    }
    
    DATE_PATTERNS = [
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b',
        r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(today|tomorrow|yesterday)\b',
        r'\b(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b',
    ]
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.mention_index: Dict[str, str] = {}  # normalized mention -> canonical_id
        self.ambiguity_clusters: Dict[str, AmbiguityCluster] = {}
        # Mentions that identify one specific person (full name or email), unlike a
        # shared first name such as "Sarah".
        self.disambiguating_aliases: Set[str] = set()
        self._build_known_entities()
        self._build_ambiguity_clusters()
    
    def _build_known_entities(self):
        """Build canonical entities from known patterns."""
        # People
        for name, info in self.PERSON_PATTERNS.items():
            eid = info['id']
            entity = Entity(
                canonical_id=eid,
                entity_type='PERSON',
                aliases={name, name.split()[0]},  # full name + first name
                attributes={'email': info['email'], 'title': info['title']}
            )
            self.entities[eid] = entity
            for alias in entity.aliases:
                self.mention_index[alias.lower()] = eid
            # Also index by email
            self.mention_index[info['email'].lower()] = eid
            self.disambiguating_aliases.add(name.lower())
            self.disambiguating_aliases.add(info['email'].lower())
        
        # Projects/Organizations
        for name, info in self.PROJECT_PATTERNS.items():
            eid = info['id']
            entity = Entity(
                canonical_id=eid,
                entity_type='ORGANIZATION' if eid.startswith('ORG') else 'PROJECT',
                aliases={name} | set(info.get('aliases', []))
            )
            self.entities[eid] = entity
            for alias in entity.aliases:
                self.mention_index[alias.lower()] = eid
    
    def _build_ambiguity_clusters(self):
        """Build explicit ambiguity clusters."""
        # Sarah Kim vs Sarah Patel - major ambiguity
        sarahs = [self.entities['PER-SARAH-KIM'], self.entities['PER-SARAH-PATEL']]
        cluster = AmbiguityCluster(
            cluster_id='AMB-SARAH',
            entities=sarahs,
            trigger_terms={'sarah'}
        )
        for e in sarahs:
            e.ambiguous = True
            e.ambiguity_cluster = 'AMB-SARAH'
        self.ambiguity_clusters['AMB-SARAH'] = cluster
        
        # John - only one John in data, but track anyway
        # Ben - only one Ben
        
        # Project name ambiguities
        # "launch" could mean Route Planner launch or other launches
        # "proposal" could mean Acme proposal or other proposals
    
    def resolve(self, mention: str, context: str = '') -> Tuple[List[Entity], bool]:
        """
        Resolve a mention to candidate entities.
        Returns (entities, is_ambiguous).
        """
        # Disambiguating mentions (a full name or an email) resolve to one person, so
        # "Sarah Patel" is not an ambiguous mention even though "Sarah" is.
        mention_norm = mention.lower().strip()
        if mention_norm in self.mention_index:
            eid = self.mention_index[mention_norm]
            entity = self.entities[eid]
            return [entity], (entity.ambiguous and mention_norm not in self.disambiguating_aliases)
        
        # Fuzzy match on aliases
        candidates = []
        for alias, eid in self.mention_index.items():
            if mention_norm in alias or alias in mention_norm:
                candidates.append(self.entities[eid])
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c.canonical_id not in seen:
                seen.add(c.canonical_id)
                unique.append(c)
        
        is_ambiguous = len(unique) > 1 or (len(unique) == 1 and unique[0].ambiguous)
        return unique, is_ambiguous
    
    def resolve_all_in_text(self, text: str) -> List[Tuple[str, List[Entity], bool]]:
        """Find and resolve all entity mentions in text."""
        results = []
        text_lower = text.lower()
        
        # Check each known alias
        for alias, eid in self.mention_index.items():
            if alias in text_lower:
                entity = self.entities[eid]
                # Find the actual span in original text
                idx = text_lower.index(alias)
                actual_mention = text[idx:idx+len(alias)]
                results.append((actual_mention, [entity], entity.ambiguous))
        
        return results
    
    def get_ambiguity_cluster(self, cluster_id: str) -> Optional[AmbiguityCluster]:
        return self.ambiguity_clusters.get(cluster_id)
    
    def get_clarification_question(self, cluster_id: str) -> str:
        """Generate a clarification question for an ambiguity cluster."""
        cluster = self.ambiguity_clusters.get(cluster_id)
        if not cluster:
            return "Which one do you mean?"
        
        options = []
        for e in cluster.entities:
            desc = e.canonical_id
            if e.attributes.get('title'):
                desc += f" ({e.attributes['title']})"
            options.append(desc)
        
        return f"Do you mean {', '.join(options[:-1])} or {options[-1]}?"
    
    def add_source_ref(self, entity_id: str, source: str, ref_id: str):
        """Add a source reference to an entity."""
        if entity_id in self.entities:
            self.entities[entity_id].source_refs.setdefault(source, []).append(ref_id)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def all_entities(self) -> List[Entity]:
        return list(self.entities.values())


# Global resolver instance
_resolver = None

def get_resolver() -> EntityResolver:
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
    return _resolver