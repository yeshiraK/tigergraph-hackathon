"""Deterministic graph-computation tools over TigerGraph OlympicGraphRAG."""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Any

from tgh.mcp.contracts import (
    CountryAggregation,
    CountryRepresentation,
    EventCandidate,
    EventContext,
    ToolExecutionResult,
)
from tgh.telemetry.trace import TraceRecorder


def normalize_string(text: str) -> str:
    """Deterministic NFKD normalization, lowercasing, and whitespace cleanup."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", text)
    cleaned = re.sub(r"[^\w\s-]", " ", norm, flags=re.UNICODE).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_date_score(date_cue: str, text: str) -> float:
    """Compute a simple match score between date_cue and event text."""
    if not date_cue:
        return 0.0
    text_norm = normalize_string(text)
    date_norm = normalize_string(date_cue)
    if not date_norm:
        return 0.0
    score = 0.0
    if date_norm in text_norm:
        score += 100.0

    tokens = [t for t in date_norm.split() if t]
    if not tokens:
        return score

    matches = sum(1 for t in tokens if t in text_norm)
    score += (matches / len(tokens)) * 10.0
    return score


@dataclass
class GraphToolBounds:
    """Safety bounds for graph operations."""

    max_events: int = 50
    max_participants: int = 50
    max_countries: int = 50
    max_provenance_paths: int = 50


class TigerGraphTools:
    """Deterministic, bounded graph computation tools for Stage 3 / A2."""

    def __init__(
        self,
        conn: Any,
        bounds: GraphToolBounds | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        """Initialize tools with TigerGraph connection and safety bounds."""
        self.conn = conn
        self.bounds = bounds or GraphToolBounds()
        self.recorder = recorder or TraceRecorder()

        # In-memory edge and vertex caches
        self._edge_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self._vertex_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _get_edges(
        self,
        vertex_type: str,
        vertex_id: str,
        edge_type: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch cached edges with URL-safe primary ID quoting."""
        cache_key = (vertex_type, vertex_id, edge_type)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        quoted_id = urllib.parse.quote(vertex_id, safe="")
        try:
            edges = self.conn.getEdges(vertex_type, quoted_id, edgeType=edge_type)
            if not isinstance(edges, list):
                edges = []
        except Exception:
            edges = []

        self._edge_cache[cache_key] = edges
        return edges

    def _get_vertices_by_id(
        self,
        vertex_type: str,
        vertex_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch vertices by ID with local caching and batch retrieval."""
        res_map: dict[str, dict[str, Any]] = {}
        missing_ids: list[str] = []

        for vid in vertex_ids:
            cache_key = (vertex_type, vid)
            if cache_key in self._vertex_cache:
                res_map[vid] = self._vertex_cache[cache_key]
            else:
                missing_ids.append(vid)

        if missing_ids:
            try:
                # pyTigerGraph getVerticesById accepts a list of IDs
                fetched = self.conn.getVerticesById(vertex_type, missing_ids)
                if isinstance(fetched, list):
                    for v in fetched:
                        vid = v.get("v_id", "")
                        if vid:
                            self._vertex_cache[(vertex_type, vid)] = v
                            res_map[vid] = v
            except Exception:
                pass

        return res_map

    # -------------------------------------------------------------------------
    # Primitive 1: PERSON / TEAM -> EVENTS
    # -------------------------------------------------------------------------
    def get_events_for_participant(
        self,
        participant_type: str,
        participant_id: str,
    ) -> ToolExecutionResult:
        """Traverse Person/Team -> PARTICIPATED_IN -> Event candidates."""
        trace = self.recorder.start_trace(
            "get_events_for_participant",
            {"participant_type": participant_type, "participant_id": participant_id},
        )
        if participant_type not in ("Person", "Team"):
            err = (
                f"Invalid participant_type: {participant_type}. "
                "Must be 'Person' or 'Team'."
            )
            trace.finish(success=False, error=err)
            return ToolExecutionResult(
                success=False,
                data=[],
                latency_ms=trace.latency_ms,
                error=err,
                trace=trace,
            )

        edges = self._get_edges(participant_type, participant_id, "PARTICIPATED_IN")
        event_ids = [e.get("to_id", "") for e in edges if e.get("to_id")]
        event_ids = [eid for eid in event_ids if eid][: self.bounds.max_events]

        vertices_map = self._get_vertices_by_id("Event", event_ids)
        candidates: list[EventCandidate] = []
        provenances: list[str] = []

        for eid in event_ids:
            v = vertices_map.get(eid, {})
            attrs = v.get("attributes", {})
            name = attrs.get("name", "")
            year = attrs.get("year", 0)
            desc = attrs.get("description", "")
            prov = (
                f"({participant_type}:{participant_id}) "
                f"-[PARTICIPATED_IN]-> (Event:{eid})"
            )
            candidates.append(
                EventCandidate(
                    event_id=eid,
                    name=name,
                    year=year,
                    description=desc,
                    provenance_path=prov,
                )
            )
            provenances.append(prov)

        trace.finish(
            success=True,
            result_summary={"count": len(candidates), "event_ids": event_ids},
            provenance=provenances,
        )
        return ToolExecutionResult(
            success=True,
            data=candidates,
            provenance=provenances,
            latency_ms=trace.latency_ms,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Primitive 2: EVENT -> CONTEXT
    # -------------------------------------------------------------------------
    def get_event_context(
        self,
        event_id: str,
    ) -> ToolExecutionResult:
        """Traverse Event -> Sport, Venue, and reverse PARTICIPATED_IN participants."""
        trace = self.recorder.start_trace(
            "get_event_context",
            {"event_id": event_id},
        )
        ev_map = self._get_vertices_by_id("Event", [event_id])
        if event_id not in ev_map:
            err = f"Event '{event_id}' not found in graph."
            trace.finish(success=False, error=err)
            return ToolExecutionResult(
                success=False,
                data=None,
                latency_ms=trace.latency_ms,
                error=err,
                trace=trace,
            )

        ev_attrs = ev_map[event_id].get("attributes", {})
        ev_name = ev_attrs.get("name", "")
        ev_year = ev_attrs.get("year", 0)
        ev_desc = ev_attrs.get("description", "")

        all_edges = self._get_edges("Event", event_id)
        sport_info: dict[str, str] | None = None
        venue_info: dict[str, str] | None = None
        participants: list[dict[str, str]] = []
        provenances: list[str] = []

        sport_ids: list[str] = []
        venue_ids: list[str] = []
        part_lookups: list[tuple[str, str]] = []

        for e in all_edges:
            e_type = e.get("e_type", "")
            to_type = e.get("to_type", "")
            to_id = e.get("to_id", "")
            if not to_id:
                continue

            if e_type == "BELONGS_TO" and to_type == "Sport":
                sport_ids.append(to_id)
                provenances.append(
                    f"(Event:{event_id}) -[BELONGS_TO]-> (Sport:{to_id})"
                )
            elif e_type == "HELD_AT" and to_type == "Venue":
                venue_ids.append(to_id)
                provenances.append(f"(Event:{event_id}) -[HELD_AT]-> (Venue:{to_id})")
            elif e_type == "reverse_PARTICIPATED_IN" and to_type in ("Person", "Team"):
                part_lookups.append((to_type, to_id))
                provenances.append(
                    f"({to_type}:{to_id}) -[PARTICIPATED_IN]-> (Event:{event_id})"
                )

        if sport_ids:
            s_map = self._get_vertices_by_id("Sport", [sport_ids[0]])
            s_name = s_map.get(sport_ids[0], {}).get("attributes", {}).get("name", "")
            sport_info = {"sport_id": sport_ids[0], "name": s_name}

        if venue_ids:
            v_map = self._get_vertices_by_id("Venue", [venue_ids[0]])
            v_name = v_map.get(venue_ids[0], {}).get("attributes", {}).get("name", "")
            venue_info = {"venue_id": venue_ids[0], "name": v_name}

        # Resolve participant names
        person_ids = [pid for ptype, pid in part_lookups if ptype == "Person"][
            : self.bounds.max_participants
        ]
        team_ids = [tid for ptype, tid in part_lookups if ptype == "Team"][
            : self.bounds.max_participants
        ]

        p_map = self._get_vertices_by_id("Person", person_ids)
        t_map = self._get_vertices_by_id("Team", team_ids)

        for ptype, pid in part_lookups[: self.bounds.max_participants]:
            if ptype == "Person":
                p_name = p_map.get(pid, {}).get("attributes", {}).get("name", pid)
                participants.append({"id": pid, "name": p_name, "type": "Person"})
            else:
                t_name = t_map.get(pid, {}).get("attributes", {}).get("name", pid)
                participants.append({"id": pid, "name": t_name, "type": "Team"})

        ctx = EventContext(
            event_id=event_id,
            name=ev_name,
            year=ev_year,
            description=ev_desc,
            sport=sport_info,
            venue=venue_info,
            participants=participants,
            provenance_paths=provenances,
        )
        trace.finish(
            success=True,
            result_summary={
                "event_id": event_id,
                "sport": sport_info.get("name") if sport_info else None,
                "venue": venue_info.get("name") if venue_info else None,
                "participants_count": len(participants),
            },
            provenance=provenances,
        )
        return ToolExecutionResult(
            success=True,
            data=ctx,
            provenance=provenances,
            latency_ms=trace.latency_ms,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Primitive 3: PERSON / TEAM -> COUNTRY
    # -------------------------------------------------------------------------
    def get_country_for_participant(
        self,
        participant_type: str,
        participant_id: str,
    ) -> ToolExecutionResult:
        """Traverse Person/Team -> REPRESENTS -> Country."""
        trace = self.recorder.start_trace(
            "get_country_for_participant",
            {"participant_type": participant_type, "participant_id": participant_id},
        )
        if participant_type not in ("Person", "Team"):
            err = (
                f"Invalid participant_type: {participant_type}. "
                "Must be 'Person' or 'Team'."
            )
            trace.finish(success=False, error=err)
            return ToolExecutionResult(
                success=False,
                data=[],
                latency_ms=trace.latency_ms,
                error=err,
                trace=trace,
            )

        edges = self._get_edges(participant_type, participant_id, "REPRESENTS")
        country_ids = [e.get("to_id", "") for e in edges if e.get("to_id")]
        c_map = self._get_vertices_by_id("Country", country_ids)
        p_map = self._get_vertices_by_id(participant_type, [participant_id])
        p_name = (
            p_map.get(participant_id, {})
            .get("attributes", {})
            .get("name", participant_id)
        )

        results: list[CountryRepresentation] = []
        provenances: list[str] = []

        for cid in country_ids:
            c_attrs = c_map.get(cid, {}).get("attributes", {})
            c_name = c_attrs.get("name", cid)
            c_code = c_attrs.get("code", "")
            prov = (
                f"({participant_type}:{participant_id}) -[REPRESENTS]-> (Country:{cid})"
            )
            results.append(
                CountryRepresentation(
                    participant_type=participant_type,
                    participant_id=participant_id,
                    participant_name=p_name,
                    country_id=cid,
                    country_name=c_name,
                    country_code=c_code,
                    provenance_path=prov,
                )
            )
            provenances.append(prov)

        trace.finish(
            success=True,
            result_summary={"count": len(results), "country_ids": country_ids},
            provenance=provenances,
        )
        return ToolExecutionResult(
            success=True,
            data=results,
            provenance=provenances,
            latency_ms=trace.latency_ms,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Primitive 4: VENUE -> EVENTS
    # -------------------------------------------------------------------------
    def get_events_for_venue(
        self,
        venue_id: str,
        year: int | None = None,
        sport: str | None = None,
        event_name_query: str | None = None,
    ) -> ToolExecutionResult:
        """Traverse Venue <- reverse_HELD_AT <- Event with deterministic filtering."""
        trace = self.recorder.start_trace(
            "get_events_for_venue",
            {
                "venue_id": venue_id,
                "year": year,
                "sport": sport,
                "event_name_query": event_name_query,
            },
        )
        edges = self._get_edges("Venue", venue_id, "reverse_HELD_AT")
        event_ids = [e.get("to_id", "") for e in edges if e.get("to_id")]
        event_ids = event_ids[: self.bounds.max_events]

        if not event_ids:
            trace.finish(
                success=True,
                result_summary={"count": 0, "event_ids": []},
                provenance=[],
            )
            return ToolExecutionResult(
                success=True,
                data=[],
                provenance=[],
                latency_ms=trace.latency_ms,
                trace=trace,
            )

        filter_res = self.filter_events(
            event_ids=event_ids,
            year=year,
            sport=sport,
            venue_id=venue_id,
            event_name_query=event_name_query,
        )
        trace.finish(
            success=filter_res.success,
            result_summary={
                "initial_count": len(event_ids),
                "filtered_count": len(filter_res.data),
            },
            provenance=filter_res.provenance,
            error=filter_res.error,
        )
        return ToolExecutionResult(
            success=filter_res.success,
            data=filter_res.data,
            provenance=filter_res.provenance,
            latency_ms=trace.latency_ms,
            error=filter_res.error,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Primitive 5: DETERMINISTIC EVENT FILTERING
    # -------------------------------------------------------------------------
    def filter_events(
        self,
        event_ids: list[str],
        year: int | None = None,
        sport: str | None = None,
        venue_id: str | None = None,
        event_name_query: str | None = None,
        venue_prefiltered: bool = False,
    ) -> ToolExecutionResult:
        """Filter candidate event IDs against deterministic graph attributes."""
        trace = self.recorder.start_trace(
            "filter_events",
            {
                "input_count": len(event_ids),
                "year": year,
                "sport": sport,
                "venue_id": venue_id,
                "event_name_query": event_name_query,
                "venue_prefiltered": venue_prefiltered,
            },
        )
        target_ids = event_ids[: max(self.bounds.max_events, len(event_ids))]
        ev_map = self._get_vertices_by_id("Event", target_ids)

        norm_name_query = normalize_string(event_name_query) if event_name_query else ""
        norm_sport = normalize_string(sport) if sport else ""

        surviving: list[EventCandidate] = []
        provenances: list[str] = []

        # Determine if we need to inspect Event edges (for sport, or unverified venue)
        need_event_edges = (
            bool(norm_sport) or (bool(venue_id) and not venue_prefiltered)
        )

        for eid in target_ids:
            if eid not in ev_map:
                continue
            attrs = ev_map[eid].get("attributes", {})
            e_name = attrs.get("name", "")
            e_year = attrs.get("year", 0)
            e_desc = attrs.get("description", "")

            # 1. Year Filter
            if year is not None and e_year != year:
                continue

            # 2. Event Name Substring / Exact Filter
            if norm_name_query:
                norm_ev_name = normalize_string(e_name)
                norm_ev_desc = normalize_string(e_desc)
                if (
                    norm_name_query not in norm_ev_name
                    and norm_name_query not in norm_ev_desc
                ):
                    continue

            # 3. Sport and Venue Relationship Filters
            event_edges = self._get_edges("Event", eid) if need_event_edges else []
            prov_steps: list[str] = [f"(Event:{eid} year={e_year})"]

            if venue_id:
                if venue_prefiltered:
                    prov_steps.append(f"-[HELD_AT]-> (Venue:{venue_id})")
                else:
                    held_venues = [
                        e.get("to_id", "")
                        for e in event_edges
                        if e.get("e_type") == "HELD_AT" and e.get("to_type") == "Venue"
                    ]
                    if venue_id not in held_venues:
                        continue
                    prov_steps.append(f"-[HELD_AT]-> (Venue:{venue_id})")

            if norm_sport:
                sport_ids = [
                    e.get("to_id", "")
                    for e in event_edges
                    if e.get("e_type") == "BELONGS_TO" and e.get("to_type") == "Sport"
                ]
                if not sport_ids:
                    continue
                s_map = self._get_vertices_by_id("Sport", [sport_ids[0]])
                s_name = (
                    s_map.get(sport_ids[0], {}).get("attributes", {}).get("name", "")
                )
                norm_graph_sport = normalize_string(s_name)
                # Check match against sport name or sport_id slug
                if (
                    norm_sport not in norm_graph_sport
                    and norm_sport not in sport_ids[0].lower()
                ):
                    continue
                prov_steps.append(f"-[BELONGS_TO]-> (Sport:{sport_ids[0]})")

            full_prov = " ".join(prov_steps)
            surviving.append(
                EventCandidate(
                    event_id=eid,
                    name=e_name,
                    year=e_year,
                    description=e_desc,
                    provenance_path=full_prov,
                )
            )
            provenances.append(full_prov)

        trace.finish(
            success=True,
            result_summary={
                "input_count": len(target_ids),
                "surviving_count": len(surviving),
            },
            provenance=provenances,
        )
        return ToolExecutionResult(
            success=True,
            data=surviving,
            provenance=provenances,
            latency_ms=trace.latency_ms,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Primitive 6: PARTICIPANT -> COUNTRY AGGREGATION
    # -------------------------------------------------------------------------
    def aggregate_countries_for_participants(
        self,
        participants: list[dict[str, str]],  # [{"id": ..., "type": ...}]
    ) -> ToolExecutionResult:
        """Aggregate and deduplicate countries represented by a set of participants."""
        trace = self.recorder.start_trace(
            "aggregate_countries_for_participants",
            {"count": len(participants)},
        )
        distinct_country_ids: set[str] = set()
        provenances: list[str] = []

        bounded_parts = participants[: self.bounds.max_participants]
        for part in bounded_parts:
            ptype = part.get("type", "Person")
            pid = part.get("id", "")
            if not pid or ptype not in ("Person", "Team"):
                continue

            edges = self._get_edges(ptype, pid, "REPRESENTS")
            for e in edges:
                cid = e.get("to_id", "")
                if cid:
                    distinct_country_ids.add(cid)
                    provenances.append(
                        f"({ptype}:{pid}) -[REPRESENTS]-> (Country:{cid})"
                    )

        c_map = self._get_vertices_by_id("Country", list(distinct_country_ids))
        country_names = [
            c_map.get(cid, {}).get("attributes", {}).get("name", cid)
            for cid in sorted(distinct_country_ids)
        ]

        agg = CountryAggregation(
            distinct_count=len(distinct_country_ids),
            country_ids=sorted(distinct_country_ids),
            country_names=country_names,
            provenance=provenances[: self.bounds.max_provenance_paths],
        )
        trace.finish(
            success=True,
            result_summary={
                "distinct_count": agg.distinct_count,
                "countries": agg.country_names,
            },
            provenance=provenances,
        )
        return ToolExecutionResult(
            success=True,
            data=agg,
            provenance=provenances,
            latency_ms=trace.latency_ms,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Capability 7: COMPOSITE MULTI-HOP OPERATION
    # -------------------------------------------------------------------------
    def execute_composite_multihop(
        self,
        participant_type: str | None = None,
        participant_id: str | None = None,
        venue_id: str | None = None,
        year: int | None = None,
        sport: str | None = None,
        event_name_query: str | None = None,
        date_cue: str | None = None,
        include_participants: bool = True,
        include_countries: bool = True,
    ) -> ToolExecutionResult:
        """Perform multi-hop reasoning by intersecting constraints and getting context.

        Workflow:
        1. If participant_id: get participant events.
        2. If venue_id: get venue events.
        3. Intersect candidate event sets.
        4. Apply year/sport/event-name filters early.
        5. For surviving events, retrieve full context (participants, venue, sport).
        6. If include_countries: aggregate distinct countries across participants.
        """
        trace = self.recorder.start_trace(
            "execute_composite_multihop",
            {
                "participant_id": participant_id,
                "venue_id": venue_id,
                "year": year,
                "sport": sport,
                "event_name_query": event_name_query,
            },
        )

        candidate_event_ids: list[str] = []
        path_segments: list[str] = []

        # Step 1: Candidate Events from Participant
        if participant_id and participant_type:
            part_res = self.get_events_for_participant(participant_type, participant_id)
            if not part_res.success:
                trace.finish(success=False, error=part_res.error)
                return part_res
            part_eids = [c.event_id for c in part_res.data]
            candidate_event_ids = part_eids
            path_segments.extend(part_res.provenance)

        # Step 2: Candidate Events from Venue
        if venue_id:
            venue_edges = self._get_edges("Venue", venue_id, "reverse_HELD_AT")
            venue_eids = [e.get("to_id", "") for e in venue_edges if e.get("to_id")]
            for eid in venue_eids:
                path_segments.append(
                    f"(Venue:{venue_id}) <-[reverse_HELD_AT]- (Event:{eid})"
                )

            if candidate_event_ids:
                # Intersect candidate sets
                candidate_event_ids = [
                    eid for eid in candidate_event_ids if eid in set(venue_eids)
                ]
            else:
                candidate_event_ids = venue_eids

        # Step 3: Filter candidates early using Year, Sport, Venue, Event Name
        filter_res = self.filter_events(
            event_ids=candidate_event_ids,
            year=year,
            sport=sport,
            venue_id=venue_id if not participant_id else venue_id,
            event_name_query=event_name_query,
            venue_prefiltered=bool(venue_id),
        )
        if not filter_res.success:
            trace.finish(success=False, error=filter_res.error)
            return filter_res

        surviving_events: list[EventCandidate] = filter_res.data
        if not surviving_events:
            trace.finish(
                success=True,
                result_summary={
                    "events_found": 0,
                    "message": "No events survived filters",
                },
                provenance=path_segments,
            )
            return ToolExecutionResult(
                success=True,
                data={
                    "events": [],
                    "event_contexts": [],
                    "country_aggregation": None,
                },
                provenance=path_segments,
                latency_ms=trace.latency_ms,
                trace=trace,
            )

        if date_cue:
            chunk_ids = [f"{ev.event_id}#c0000" for ev in surviving_events]
            chunk_map = self._get_vertices_by_id("Chunk", chunk_ids)

            def _score_event(ev: EventCandidate) -> float:
                c_data = chunk_map.get(f"{ev.event_id}#c0000", {})
                c_text = c_data.get("attributes", {}).get("text", "")
                full_text = f"{ev.name} {ev.description} {c_text}"
                return compute_date_score(date_cue, full_text)

            surviving_events.sort(key=lambda ev: -_score_event(ev))

        # Step 4: Expand surviving events to full context
        contexts: list[EventContext] = []
        all_participants: list[dict[str, str]] = []

        max_expand = min(
            self.bounds.max_events, 10 if date_cue else self.bounds.max_events
        )
        for ev in surviving_events[:max_expand]:
            ctx_res = self.get_event_context(ev.event_id)
            if ctx_res.success and ctx_res.data:
                contexts.append(ctx_res.data)
                if include_participants:
                    all_participants.extend(ctx_res.data.participants)
                path_segments.extend(ctx_res.provenance)

        # Step 5: Country Aggregation
        country_agg: CountryAggregation | None = None
        if include_countries and all_participants:
            agg_res = self.aggregate_countries_for_participants(all_participants)
            if agg_res.success:
                country_agg = agg_res.data
                path_segments.extend(agg_res.provenance)

        composite_data = {
            "events": surviving_events,
            "event_contexts": contexts,
            "country_aggregation": country_agg,
        }

        trace.finish(
            success=True,
            result_summary={
                "surviving_events_count": len(surviving_events),
                "event_ids": [e.event_id for e in surviving_events],
                "participants_count": len(all_participants),
                "countries_count": country_agg.distinct_count if country_agg else 0,
            },
            provenance=path_segments[: self.bounds.max_provenance_paths],
        )

        return ToolExecutionResult(
            success=True,
            data=composite_data,
            provenance=path_segments[: self.bounds.max_provenance_paths],
            latency_ms=trace.latency_ms,
            trace=trace,
        )
