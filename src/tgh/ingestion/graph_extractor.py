"""Deterministic offline graph extractor for OlympicGraphRAG.

Extracts domain entities and relationships strictly grounded in `corpus.jsonl`:
- Vertices: Event, Sport, Venue, Country, Person, Team, Entity
- Edges: HELD_AT, BELONGS_TO, PARTICIPATED_IN, REPRESENTS, MENTIONS, RESOLVES_TO

Strict grounding principles:
1. Every fact is directly derived from corpus text / infoboxes.
2. No external web knowledge or model memory.
3. Unresolved fields (e.g. missing venue) are tracked and left unresolved.
4. Deterministic, reproducible IDs (canonical names, doc_ids, NOCs).
5. Idempotent: repeated runs produce identical records.
6. MENTIONS provenance: grounded strictly to chunks containing entity surface text.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tgh.ingestion.chunker import SemanticChunker

# Regex to extract Olympic sport and year from document title:
# e.g. "Badminton at the 2016 Summer Olympics – Women's singles"
RE_SPORT_TITLE = re.compile(
    r"^(.+?) at the \d{4} (Summer|Winter) Olympics",
    re.IGNORECASE,
)

# Regex to extract 4-digit year from games string or title:
RE_YEAR = re.compile(r"\b(18\d\d|19\d\d|20\d\d)\b")

# Regex to detect team events:
RE_TEAM_PATTERNS = re.compile(
    r"\b(team|relay|doubles|pairs?|duet|tournament|two-man|four-man|two-woman|"
    r"coxless|coxed|eight|quadruple sculls|double sculls|synchronized|"
    r"team pursuit|team sprint|ice dance)\b|"
    r"4\s*[×x]\s*\d+|K-2|K-4|C-2|C-4",
    re.IGNORECASE,
)

# Regex to detect explicit multi-crew / group team competitions:
RE_MULTI_CREW_PATTERNS = re.compile(
    r"\b(49er|470|nacra(?:\s*17)?|yngling|group)\b",
    re.IGNORECASE,
)

# Medalist keys in [Infobox Olympic event]
MEDAL_KEYS = [
    ("gold", "goldNOC"),
    ("silver", "silverNOC"),
    ("bronze", "bronzeNOC"),
    ("gold2", "goldNOC2"),
    ("silver2", "silverNOC2"),
    ("bronze2", "bronzeNOC2"),
]


def slugify(text: str) -> str:
    """Deterministic slugification with NFKD normalization and ASCII chars."""
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = re.sub(r"[^\w\s-]", "", normalized, flags=re.UNICODE).strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned).strip("_")


def make_event_id(doc_id: str) -> str:
    """Event ID maps 1:1 to Document doc_id (e.g. Q25301483)."""
    return doc_id.strip()


def make_sport_id(sport_name: str) -> str:
    """Sport ID from slugified sport name (e.g. sport_badminton)."""
    return f"sport_{slugify(sport_name)}"


def make_venue_id(venue_name: str) -> str:
    """Venue ID from slugified venue name (e.g. venue_riocentro_pavilion_4)."""
    return f"venue_{slugify(venue_name)}"


def make_country_id(noc: str) -> str:
    """Country ID from 3-letter NOC code (e.g. country_ESP)."""
    return f"country_{noc.strip().upper()}"


def make_person_id(person_name: str) -> str:
    """Person ID from slugified athlete name (e.g. person_carolina_marin)."""
    return f"person_{slugify(person_name)}"


def make_team_id(team_name: str, doc_id: str, medal_key: str) -> str:
    """Team ID from slugified team/roster name or fallback to doc_id + medal."""
    slug = slugify(team_name)
    if len(slug) > 100 or len(slug) < 3:
        return f"team_{doc_id}_{medal_key}"
    return f"team_{slug}"


def make_entity_id(entity_type: str, canonical_id: str) -> str:
    """Entity hub ID (e.g. entity_venue_riocentro_pavilion_4)."""
    return f"entity_{canonical_id}"


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    name: str
    year: int
    description: str


@dataclass(frozen=True)
class SportRecord:
    sport_id: str
    name: str
    description: str


@dataclass(frozen=True)
class VenueRecord:
    venue_id: str
    name: str
    city: str
    country: str


@dataclass(frozen=True)
class CountryRecord:
    country_id: str
    name: str
    code: str


@dataclass(frozen=True)
class PersonRecord:
    person_id: str
    name: str
    description: str


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    name: str
    description: str


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    name: str
    entity_type: str
    description: str


@dataclass(frozen=True)
class EdgeRecord:
    source_type: str
    source_id: str
    edge_type: str
    target_type: str
    target_id: str


@dataclass
class ExtractionStats:
    total_docs_processed: int = 0
    olympic_event_docs: int = 0
    skipped_non_olympic_docs: int = 0
    event_count: int = 0
    sport_count: int = 0
    venue_count: int = 0
    country_count: int = 0
    person_count: int = 0
    team_count: int = 0
    entity_count: int = 0
    held_at_count: int = 0
    belongs_to_count: int = 0
    participated_in_count: int = 0
    represents_count: int = 0
    mentions_count: int = 0
    resolves_to_count: int = 0
    unresolved_venues: int = 0
    unresolved_countries: int = 0
    unresolved_medalists: int = 0
    ambiguous_mappings: int = 0
    person_classifications: int = 0
    team_classifications: int = 0
    sport_title_only_no_mentions: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_docs_processed": self.total_docs_processed,
            "olympic_event_docs": self.olympic_event_docs,
            "skipped_non_olympic_docs": self.skipped_non_olympic_docs,
            "event_count": self.event_count,
            "sport_count": self.sport_count,
            "venue_count": self.venue_count,
            "country_count": self.country_count,
            "person_count": self.person_count,
            "team_count": self.team_count,
            "entity_count": self.entity_count,
            "held_at_count": self.held_at_count,
            "belongs_to_count": self.belongs_to_count,
            "participated_in_count": self.participated_in_count,
            "represents_count": self.represents_count,
            "mentions_count": self.mentions_count,
            "resolves_to_count": self.resolves_to_count,
            "unresolved_venues": self.unresolved_venues,
            "unresolved_countries": self.unresolved_countries,
            "unresolved_medalists": self.unresolved_medalists,
            "ambiguous_mappings": self.ambiguous_mappings,
            "person_classifications": self.person_classifications,
            "team_classifications": self.team_classifications,
            "sport_title_only_no_mentions": self.sport_title_only_no_mentions,
        }


@dataclass
class ExtractedGraphData:
    events: dict[str, EventRecord] = field(default_factory=dict)
    sports: dict[str, SportRecord] = field(default_factory=dict)
    venues: dict[str, VenueRecord] = field(default_factory=dict)
    countries: dict[str, CountryRecord] = field(default_factory=dict)
    persons: dict[str, PersonRecord] = field(default_factory=dict)
    teams: dict[str, TeamRecord] = field(default_factory=dict)
    entities: dict[str, EntityRecord] = field(default_factory=dict)

    # Edge sets:
    # held_at: (event_id, venue_id)
    held_at_edges: set[tuple[str, str]] = field(default_factory=set)
    # belongs_to: (event_id, sport_id)
    belongs_to_edges: set[tuple[str, str]] = field(default_factory=set)
    # participated_in: (src_type, src_id, event_id)
    participated_in_edges: set[tuple[str, str, str]] = field(default_factory=set)
    # represents: (src_type, src_id, country_id)
    represents_edges: set[tuple[str, str, str]] = field(default_factory=set)
    # mentions: (chunk_id, entity_id)
    mentions_edges: set[tuple[str, str]] = field(default_factory=set)
    # resolves_to: (entity_id, target_type, target_id)
    resolves_to_edges: set[tuple[str, str, str]] = field(default_factory=set)

    stats: ExtractionStats = field(default_factory=ExtractionStats)


def parse_infobox_olympic_event(text: str) -> dict[str, str] | None:
    """Extract key-value pairs from [Infobox Olympic event]."""
    if "[Infobox Olympic event]" not in text:
        return None

    ib_part = text.split("[Infobox Olympic event]", 1)[1]
    lines_block = ib_part.split("\n\n", 1)[0]
    data: dict[str, str] = {}
    for line in lines_block.split("\n"):
        line_str = line.strip()
        if ":" in line_str:
            k, v = line_str.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def is_team_event(title: str, event_name: str, infobox_data: dict[str, Any]) -> bool:
    """Determine deterministically whether an event is a team competition.

    Priority order:
    1. Explicit "solo" or "individual" forces Person classification.
    2. Multi-crew / team patterns (49er, 470, Nacra, Yngling, group) force Team.
    3. Standard team keywords (team, relay, doubles, pairs, duet, tournament, etc.).
    4. Infobox `teams:` field (checked against `competitors:` for single-handed boats).
    """
    combined = f"{title} {event_name}".lower()

    # Priority 1: Explicit "solo" or "individual" forces Person classification
    if re.search(r"\b(solo|individual)\b", combined):
        return False

    # Priority 2: Explicit multi-crew / team patterns
    if RE_MULTI_CREW_PATTERNS.search(combined):
        return True

    # Priority 3: Existing team patterns
    if RE_TEAM_PATTERNS.search(title) or RE_TEAM_PATTERNS.search(event_name):
        return True

    # Priority 4: teams field in infobox (check competitors if single-handed)
    if "teams" in infobox_data:
        teams_val = str(infobox_data.get("teams", ""))
        comp_val = str(infobox_data.get("competitors", ""))
        t_num = re.search(r"\d+", teams_val)
        c_num = re.search(r"\d+", comp_val)
        if t_num and c_num and int(t_num.group(0)) == int(c_num.group(0)):
            return False
        return True

    return False


def extract_graph_from_corpus(
    corpus_path: Path,
    chunker: SemanticChunker | None = None,
) -> ExtractedGraphData:
    """Extract all deterministic graph vertices and edges from the corpus."""
    if chunker is None:
        chunker = SemanticChunker(target_tokens=768)

    data = ExtractedGraphData()
    stats = data.stats

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            stats.total_docs_processed += 1
            doc = json.loads(line_str)
            doc_id = doc["doc_id"]
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""

            infobox = parse_infobox_olympic_event(text)
            if infobox is None:
                stats.skipped_non_olympic_docs += 1
                continue

            stats.olympic_event_docs += 1

            # 1. Sport
            sport_match = RE_SPORT_TITLE.match(title)
            sport_id: str | None = None
            sport_name: str | None = None
            if sport_match:
                sport_name = sport_match.group(1).strip()
                sport_id = make_sport_id(sport_name)
                if sport_id not in data.sports:
                    data.sports[sport_id] = SportRecord(
                        sport_id=sport_id,
                        name=sport_name,
                        description=f"Olympic sport: {sport_name}",
                    )
            else:
                stats.ambiguous_mappings += 1

            # 2. Event
            event_name = infobox.get("event", "") or title
            games_str = infobox.get("games", "")
            year_match = RE_YEAR.search(games_str) or RE_YEAR.search(title)
            year_int = int(year_match.group(1)) if year_match else 0
            event_id = make_event_id(doc_id)

            data.events[event_id] = EventRecord(
                event_id=event_id,
                name=event_name,
                year=year_int,
                description=title,
            )

            # Edge: Event -> Sport (BELONGS_TO)
            if sport_id:
                data.belongs_to_edges.add((event_id, sport_id))

            # 3. Venue
            venue_str = infobox.get("venue", "") or infobox.get("venues", "")
            venue_id: str | None = None
            if venue_str:
                venue_id = make_venue_id(venue_str)
                if venue_id not in data.venues:
                    data.venues[venue_id] = VenueRecord(
                        venue_id=venue_id,
                        name=venue_str,
                        city="",
                        country="",
                    )
                # Edge: Event -> Venue (HELD_AT)
                data.held_at_edges.add((event_id, venue_id))
            else:
                stats.unresolved_venues += 1

            # 4. Medalists (Person / Team / Country)
            team_event_flag = is_team_event(title, event_name, infobox)

            # (ent_id, ent_type, ent_name, tgt_type, tgt_id, surf, is_noc, is_sport_ev)
            doc_entities: list[tuple[str, str, str, str, str, str, bool, bool]] = []

            if sport_id and sport_name:
                doc_entities.append((
                    make_entity_id("sport", sport_id),
                    "Sport",
                    data.sports[sport_id].name,
                    "Sport",
                    sport_id,
                    sport_name,
                    False,
                    True,
                ))
            if venue_id and venue_str:
                doc_entities.append((
                    make_entity_id("venue", venue_id),
                    "Venue",
                    data.venues[venue_id].name,
                    "Venue",
                    venue_id,
                    venue_str,
                    False,
                    False,
                ))
            doc_entities.append((
                make_entity_id("event", f"event_{event_id}"),
                "Event",
                event_name,
                "Event",
                event_id,
                event_name,
                False,
                True,
            ))

            for medal_key, noc_key in MEDAL_KEYS:
                medalist_val = infobox.get(medal_key, "").strip()
                noc_val = infobox.get(noc_key, "").strip()

                if not medalist_val:
                    continue

                if medalist_val.lower() in ["vacant", "none", "tbd", "stripped"]:
                    stats.unresolved_medalists += 1
                    continue

                # Country resolution
                country_id: str | None = None
                if noc_val and len(noc_val) == 3 and noc_val.isupper():
                    country_id = make_country_id(noc_val)
                    if country_id not in data.countries:
                        data.countries[country_id] = CountryRecord(
                            country_id=country_id,
                            name=noc_val,
                            code=noc_val,
                        )
                    doc_entities.append((
                        make_entity_id("country", country_id),
                        "Country",
                        noc_val,
                        "Country",
                        country_id,
                        noc_val,
                        True,
                        False,
                    ))
                else:
                    stats.unresolved_countries += 1

                # Person vs Team resolution
                if team_event_flag:
                    stats.team_classifications += 1
                    team_id = make_team_id(medalist_val, event_id, medal_key)
                    if team_id not in data.teams:
                        data.teams[team_id] = TeamRecord(
                            team_id=team_id,
                            name=medalist_val,
                            description=f"Medalist team for {title}",
                        )
                    # Edge: Team -> Event (PARTICIPATED_IN)
                    data.participated_in_edges.add(("Team", team_id, event_id))
                    # Edge: Team -> Country (REPRESENTS)
                    if country_id:
                        data.represents_edges.add(("Team", team_id, country_id))
                    doc_entities.append((
                        make_entity_id("team", team_id),
                        "Team",
                        medalist_val,
                        "Team",
                        team_id,
                        medalist_val,
                        False,
                        False,
                    ))
                else:
                    stats.person_classifications += 1
                    person_id = make_person_id(medalist_val)
                    if person_id not in data.persons:
                        data.persons[person_id] = PersonRecord(
                            person_id=person_id,
                            name=medalist_val,
                            description="Olympic medalist",
                        )
                    # Edge: Person -> Event (PARTICIPATED_IN)
                    data.participated_in_edges.add(("Person", person_id, event_id))
                    # Edge: Person -> Country (REPRESENTS)
                    if country_id:
                        data.represents_edges.add(("Person", person_id, country_id))
                    doc_entities.append((
                        make_entity_id("person", person_id),
                        "Person",
                        medalist_val,
                        "Person",
                        person_id,
                        medalist_val,
                        False,
                        False,
                    ))

            # 5. Entity Hub and Chunk Mentions (Grounded Provenance Scanning)
            chunks = chunker.chunk_document(doc_id, text)

            for (
                ent_id,
                ent_type,
                ent_name,
                tgt_type,
                tgt_id,
                surf,
                is_noc,
                is_sport_ev,
            ) in doc_entities:
                if ent_id not in data.entities:
                    data.entities[ent_id] = EntityRecord(
                        entity_id=ent_id,
                        name=ent_name,
                        entity_type=ent_type,
                        description=f"{ent_type} entity mention: {ent_name}",
                    )
                # Edge: Entity -> Target (RESOLVES_TO)
                data.resolves_to_edges.add((ent_id, tgt_type, tgt_id))

                # Scan all chunks of this document for the actual surface string
                found_in_any_chunk = False
                for c in chunks:
                    matched = False
                    if is_noc:
                        # Match 3-letter NOC as whole word
                        if re.search(r"\b" + re.escape(surf) + r"\b", c.text):
                            matched = True
                    elif is_sport_ev:
                        # Match sport/event case-insensitively in chunk text
                        if surf in c.text or surf.lower() in c.text.lower():
                            matched = True
                    else:
                        # Verbatim match for venue and athlete/team names
                        if surf in c.text:
                            matched = True

                    if matched:
                        data.mentions_edges.add((c.chunk_id, ent_id))
                        found_in_any_chunk = True

                # If title-derived Sport is absent from chunks, do not fabricate
                if not found_in_any_chunk and ent_type == "Sport":
                    stats.sport_title_only_no_mentions += 1

    # Update counts
    stats.event_count = len(data.events)
    stats.sport_count = len(data.sports)
    stats.venue_count = len(data.venues)
    stats.country_count = len(data.countries)
    stats.person_count = len(data.persons)
    stats.team_count = len(data.teams)
    stats.entity_count = len(data.entities)

    stats.held_at_count = len(data.held_at_edges)
    stats.belongs_to_count = len(data.belongs_to_edges)
    stats.participated_in_count = len(data.participated_in_edges)
    stats.represents_count = len(data.represents_edges)
    stats.mentions_count = len(data.mentions_edges)
    stats.resolves_to_count = len(data.resolves_to_edges)

    return data


def validate_extracted_graph(data: ExtractedGraphData) -> list[str]:
    """Validate referential integrity and consistency of the extracted graph."""
    errors: list[str] = []

    all_event_ids = set(data.events.keys())
    all_sport_ids = set(data.sports.keys())
    all_venue_ids = set(data.venues.keys())
    all_country_ids = set(data.countries.keys())
    all_person_ids = set(data.persons.keys())
    all_team_ids = set(data.teams.keys())
    all_entity_ids = set(data.entities.keys())

    # 1. Check non-empty IDs
    for name, s in [
        ("events", all_event_ids),
        ("sports", all_sport_ids),
        ("venues", all_venue_ids),
        ("countries", all_country_ids),
        ("persons", all_person_ids),
        ("teams", all_team_ids),
        ("entities", all_entity_ids),
    ]:
        if "" in s or None in s:  # type: ignore
            errors.append(f"Empty or None ID found in {name}")

    # 2. Check BELONGS_TO: Event -> Sport
    for e_id, s_id in data.belongs_to_edges:
        if e_id not in all_event_ids:
            errors.append(f"BELONGS_TO source Event {e_id} missing")
        if s_id not in all_sport_ids:
            errors.append(f"BELONGS_TO target Sport {s_id} missing")

    # 3. Check HELD_AT: Event -> Venue
    for e_id, v_id in data.held_at_edges:
        if e_id not in all_event_ids:
            errors.append(f"HELD_AT source Event {e_id} missing")
        if v_id not in all_venue_ids:
            errors.append(f"HELD_AT target Venue {v_id} missing")

    # 4. Check PARTICIPATED_IN: Person -> Event or Team -> Event
    for src_type, src_id, e_id in data.participated_in_edges:
        if src_type == "Person" and src_id not in all_person_ids:
            errors.append(f"PARTICIPATED_IN Person {src_id} missing")
        elif src_type == "Team" and src_id not in all_team_ids:
            errors.append(f"PARTICIPATED_IN Team {src_id} missing")
        if e_id not in all_event_ids:
            errors.append(f"PARTICIPATED_IN target Event {e_id} missing")

    # 5. Check REPRESENTS: Person -> Country or Team -> Country
    for src_type, src_id, c_id in data.represents_edges:
        if src_type == "Person" and src_id not in all_person_ids:
            errors.append(f"REPRESENTS Person {src_id} missing")
        elif src_type == "Team" and src_id not in all_team_ids:
            errors.append(f"REPRESENTS Team {src_id} missing")
        if c_id not in all_country_ids:
            errors.append(f"REPRESENTS target Country {c_id} missing")

    # 6. Check RESOLVES_TO: Entity -> TypedVertex
    target_id_maps = {
        "Sport": all_sport_ids,
        "Venue": all_venue_ids,
        "Country": all_country_ids,
        "Event": all_event_ids,
        "Person": all_person_ids,
        "Team": all_team_ids,
    }
    for ent_id, tgt_type, tgt_id in data.resolves_to_edges:
        if ent_id not in all_entity_ids:
            errors.append(f"RESOLVES_TO Entity {ent_id} missing")
        if tgt_type not in target_id_maps:
            errors.append(f"RESOLVES_TO unknown target type {tgt_type}")
        elif tgt_id not in target_id_maps[tgt_type]:
            errors.append(f"RESOLVES_TO target {tgt_type} {tgt_id} missing")

    return errors
