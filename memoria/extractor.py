"""
Entity and relation extraction from conversation text.

This is the write path — the only part that requires an LLM call.
One call per conversation, not per query. We pay at write time
because writes are less frequent and latency-tolerant.
"""

from __future__ import annotations

import json
import re
import time

from .graph import KnowledgeGraph
from .database import transaction
from . import validation


EXTRACTION_PROMPT = """\
Extract structured knowledge from this conversation. Return a JSON object with:

1. "entities": list of {{"name": str, "type": str}}
   Types: person, project, tool, concept, file, organization, location, event

2. "facts": list of {{"subject": str, "predicate": str, "object": str,
   "confidence": float, "replace_existing": bool}}
   Predicates should be short verb phrases: "uses", "is_part_of", "decided_to", "prefers", etc.
   Confidence 0.0-1.0 based on how definitive the statement is.

3. "causal": list of {{"cause": str, "effect": str, "confidence": float}}
   Only extract when there's a clear causal relationship stated or strongly implied.

4. "decisions": list of {{"subject": str, "decision": str, "reason": str, "confidence": float}}
   Extract explicit decisions, preferences, or choices.

Rules:
- Extract ONLY what is stated or strongly implied. Do not infer.
- Use canonical entity names (normalize casing, expand abbreviations).
- Predicates should be present tense, lowercase, underscore-separated.
- Set replace_existing=true only for a stateful, single-valued attribute update
  (for example version/status/database changing). Set it false for additive
  relationships such as uses/includes/depends_on/knows.
- Confidence: 1.0 = explicitly stated fact, 0.7 = strongly implied, 0.5 = mentioned in passing.
- Skip greetings, meta-conversation, and filler.

Conversation:
{text}

Respond with ONLY the JSON object, no markdown fences."""


def extract_from_text(text: str, llm_call) -> dict:
    """Extract entities and relations from text using an LLM.

    Args:
        text: Conversation text to extract from.
        llm_call: Callable that takes a prompt string and returns a response string.
                  This keeps the extractor LLM-agnostic.

    Returns:
        Parsed extraction result with entities, facts, causal, decisions.
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])  # cap input size
    response = llm_call(prompt)

    # Parse JSON from response, handling potential markdown fences
    response = validation.text(response, "extraction response").strip()
    if response.startswith("```"):
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError("extraction response must contain a JSON object")

    return validate_extraction(result)


def validate_extraction(extraction: dict) -> dict:
    """Reject malformed provider output before any database changes."""
    if not isinstance(extraction, dict):
        raise ValueError("extraction must be a JSON object")
    required = {
        "entities": ("name",),
        "facts": ("subject", "predicate", "object"),
        "causal": ("cause", "effect"),
        "decisions": ("subject", "decision"),
    }
    result = {}
    for section, fields in required.items():
        items = extraction.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"extraction {section} must be a list")
        result[section] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"extraction {section} entries must be objects")
            item = dict(item)
            # Incomplete statements are harmlessly skipped, but wrong types
            # and invalid confidence values must never reach SQLite.
            if any(not isinstance(item.get(field, ""), str) for field in fields):
                raise ValueError(f"extraction {section} fields must be strings")
            if any(not item.get(field, "").strip() for field in fields):
                continue
            for field in fields:
                item[field] = validation.text(item.get(field), f"{section}.{field}").strip()
            if "confidence" in item:
                validation.number(item["confidence"], "confidence", minimum=0, maximum=1)
            if "replace_existing" in item:
                validation.boolean(item["replace_existing"], "replace_existing")
            if section == "entities":
                validation.text(item.get("type"), "entity type", optional=True)
            result[section].append(item)
    return result


def prepare_embeddings(extraction: dict, embedder) -> dict:
    """Embed unique entity names in one batch, before acquiring a write lock."""
    names = [item["name"] for item in extraction["entities"]]
    names.extend(item["subject"] for item in extraction["facts"])
    names.extend(item["subject"] for item in extraction["decisions"])
    for item in extraction["causal"]:
        names.extend([item["cause"], item["effect"]])
    names = list(dict.fromkeys(names))
    if not names or embedder is None:
        return {}
    vectors = embedder.embed(names)
    if len(vectors) != len(names):
        raise ValueError("embedder returned an unexpected number of vectors")
    return dict(zip(names, vectors))


def ingest_extraction(
    kg: KnowledgeGraph,
    extraction: dict,
    source_ref: str | None = None,
    embedder=None,
    embeddings: dict | None = None,
) -> dict:
    """Write extracted entities and relations into the knowledge graph.

    Returns summary of what was written: {entities_added, triples_added, contradictions}.
    """
    extraction = validate_extraction(extraction)
    if embeddings is None:
        embeddings = prepare_embeddings(extraction, embedder)
    with transaction(kg.db):
        return _ingest_extraction(kg, extraction, source_ref, embeddings)


def _ingest_extraction(kg, extraction, source_ref, embeddings):
    now = time.time()
    entity_map = {}  # name -> entity_id
    stats = {"entities_added": 0, "triples_added": 0, "contradictions": []}

    # Phase 1: Ensure all entities exist
    for e in extraction.get("entities", []):
        name = e.get("name", "").strip()
        if not name:
            continue
        etype = e.get("type")
        embedding = embeddings.get(name)
        existed = bool(kg.find_entities(name, exact=True))
        eid = kg.add_entity(name, entity_type=etype, embedding=embedding)
        entity_map[name.casefold()] = eid
        stats["entities_added"] += int(not existed)

    def resolve(name: str | None) -> str | None:
        """Resolve entity name to ID, creating if needed."""
        if not name:
            return None
        name = name.strip()
        if not name:
            return None
        key = name.casefold()
        if key in entity_map:
            return entity_map[key]
        existing = kg.find_entities(name, exact=True)
        # add_entity also backfills embeddings for previously unembedded names.
        eid = kg.add_entity(name, embedding=embeddings.get(name))
        entity_map[key] = eid
        stats["entities_added"] += int(not existing)
        return eid

    # Phase 2: Write fact triples
    for fact in extraction.get("facts", []):
        subj = resolve(fact.get("subject", ""))
        if not subj:
            continue

        predicate = fact.get("predicate", "related_to")
        obj_name = fact.get("object", "")
        confidence = float(fact.get("confidence", 0.8))

        # Determine if object is an entity or literal value
        obj_id = None
        obj_value = None
        if obj_name.casefold() in entity_map or any(
            e["name"].lower() == obj_name.lower()
            for e in extraction.get("entities", [])
        ):
            obj_id = resolve(obj_name)
        else:
            obj_value = obj_name

        tid, contras = kg.add_triple(
            subject_id=subj, predicate=predicate,
            object_id=obj_id, object_value=obj_value,
            relation_type="fact", confidence=confidence,
            valid_from=now, source_ref=source_ref,
            replace_existing=fact.get("replace_existing"),
        )
        stats["triples_added"] += 1
        if contras:
            stats["contradictions"].extend(contras)

    # Phase 3: Write causal triples
    for causal in extraction.get("causal", []):
        cause_id = resolve(causal.get("cause", ""))
        effect_id = resolve(causal.get("effect", ""))
        if not cause_id or not effect_id:
            continue

        tid, _ = kg.add_triple(
            subject_id=cause_id, predicate="caused",
            object_id=effect_id, relation_type="causal",
            confidence=float(causal.get("confidence", 0.7)),
            valid_from=now, source_ref=source_ref,
        )
        stats["triples_added"] += 1

    # Phase 4: Write decision triples
    for dec in extraction.get("decisions", []):
        subj = resolve(dec.get("subject", ""))
        if not subj:
            continue

        tid, contras = kg.add_triple(
            subject_id=subj, predicate="decided",
            object_value=dec.get("decision", ""),
            relation_type="fact",
            confidence=float(dec.get("confidence", 0.9)),
            valid_from=now, source_ref=source_ref,
            replace_existing=False,
        )
        stats["triples_added"] += 1
        if contras:
            stats["contradictions"].extend(contras)

    return stats
