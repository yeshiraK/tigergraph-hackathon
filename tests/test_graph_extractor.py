"""Unit and integration tests for deterministic graph extraction layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tgh.ingestion.graph_extractor import (
    extract_graph_from_corpus,
    is_team_event,
    make_country_id,
    make_entity_id,
    make_event_id,
    make_person_id,
    make_sport_id,
    make_team_id,
    make_venue_id,
    slugify,
    validate_extracted_graph,
)

SAMPLE_INDIVIDUAL_DOC = {
    "doc_id": "Q25301483",
    "title": "Badminton at the 2016 Summer Olympics – Women's singles",
    "source": "wikipedia",
    "url": "https://example.org/badminton",
    "text": (
        "[Infobox Olympic event]\n"
        "  event: Women's singles\n"
        "  games: 2016 Summer\n"
        "  venue: Riocentro – Pavilion 4\n"
        "  date: 11–19 August\n"
        "  competitors: 40\n"
        "  nations: 35\n"
        "  gold: Carolina Marín\n"
        "  goldNOC: ESP\n"
        "  silver: P. V. Sindhu\n"
        "  silverNOC: IND\n"
        "  bronze: Nozomi Okuhara\n"
        "  bronzeNOC: JPN\n\n"
        "The women's singles badminton tournament was held in Rio."
    ),
}

SAMPLE_TEAM_DOC = {
    "doc_id": "Q2297633",
    "title": "Cycling at the 2012 Summer Olympics – Women's team pursuit",
    "source": "wikipedia",
    "url": "https://example.org/cycling",
    "text": (
        "[Infobox Olympic event]\n"
        "  event: Women's team pursuit\n"
        "  games: 2012 Summer\n"
        "  venue: London Velopark\n"
        "  date: 3 to 4 August\n"
        "  teams: 10\n"
        "  gold: Dani King, Laura Trott, Joanna Rowsell\n"
        "  goldNOC: GBR\n"
        "  silver: Sarah Hammer, Dotsie Bausch, Jennie Reed\n"
        "  silverNOC: USA\n"
        "  bronze: Tara Whitten, Gillian Carleton, Jasmin Glaesser\n"
        "  bronzeNOC: CAN\n\n"
        "Great Britain won the gold medal."
    ),
}

SAMPLE_MISSING_VENUE_AND_VACANT_DOC = {
    "doc_id": "Q999999",
    "title": "Weightlifting at the 2016 Summer Olympics – Men's 77 kg",
    "source": "wikipedia",
    "url": "https://example.org",
    "text": (
        "[Infobox Olympic event]\n"
        "  event: Men's 77 kg\n"
        "  games: 2016 Summer\n"
        "  gold: vacant\n"
        "  goldNOC: \n"
        "  silver: Nijat Rahimov\n"
        "  silverNOC: KAZ\n"
        "  bronze: Mohamed Ihab\n"
        "  bronzeNOC: EGY\n\n"
        "The gold medal was declared vacant due to doping disqualification."
    ),
}

SAMPLE_NON_OLYMPIC_DOC = {
    "doc_id": "Q1520721",
    "title": "Jab We Met",
    "source": "wikipedia",
    "url": "https://example.org/film",
    "text": (
        "[Infobox film]\n"
        "  name: Jab We Met\n"
        "  director: Imtiaz Ali\n\n"
        "Jab We Met is a 2007 Indian Hindi-language romantic comedy film."
    ),
}


class TestGraphExtractor(unittest.TestCase):
    """Test suite covering the deterministic graph extractor."""

    def test_slugify_and_id_determinism(self) -> None:
        """Slugify and ID generators must be 100% deterministic and normalized."""
        self.assertEqual(slugify("Riocentro – Pavilion 4"), "riocentro_pavilion_4")
        self.assertEqual(slugify("Riocentro - Pavilion 4"), "riocentro_pavilion_4")
        self.assertEqual(
            slugify("Estadi Olímpic de Montjuïc"), "estadi_olimpic_de_montjuic"
        )
        self.assertEqual(slugify("Val-d'Isère"), "val_disere")

        # IDs
        self.assertEqual(make_event_id("Q25301483"), "Q25301483")
        self.assertEqual(make_sport_id("Badminton"), "sport_badminton")
        self.assertEqual(
            make_venue_id("Richmond Olympic Oval"), "venue_richmond_olympic_oval"
        )
        self.assertEqual(make_country_id("esp"), "country_ESP")
        self.assertEqual(make_person_id("Carolina Marín"), "person_carolina_marin")
        self.assertEqual(
            make_team_id("Great Britain", "Q123", "gold"), "team_great_britain"
        )
        self.assertEqual(
            make_entity_id("sport", "sport_badminton"), "entity_sport_badminton"
        )

    def test_repeatability(self) -> None:
        """Running extraction twice produces identical vertices and edges."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(SAMPLE_INDIVIDUAL_DOC) + "\n")
            tmp.write(json.dumps(SAMPLE_TEAM_DOC) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res1 = extract_graph_from_corpus(tmp_path)
            res2 = extract_graph_from_corpus(tmp_path)

            self.assertEqual(res1.events, res2.events)
            self.assertEqual(res1.sports, res2.sports)
            self.assertEqual(res1.venues, res2.venues)
            self.assertEqual(res1.countries, res2.countries)
            self.assertEqual(res1.persons, res2.persons)
            self.assertEqual(res1.teams, res2.teams)
            self.assertEqual(res1.entities, res2.entities)
            self.assertEqual(res1.held_at_edges, res2.held_at_edges)
            self.assertEqual(res1.belongs_to_edges, res2.belongs_to_edges)
            self.assertEqual(res1.participated_in_edges, res2.participated_in_edges)
            self.assertEqual(res1.represents_edges, res2.represents_edges)
            self.assertEqual(res1.mentions_edges, res2.mentions_edges)
            self.assertEqual(res1.resolves_to_edges, res2.resolves_to_edges)
        finally:
            tmp_path.unlink()

    def test_valid_vertex_generation(self) -> None:
        """Extracts valid Event, Sport, Venue, Country, Person, Entity vertices."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(SAMPLE_INDIVIDUAL_DOC) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            self.assertIn("Q25301483", res.events)
            event = res.events["Q25301483"]
            self.assertEqual(event.event_id, "Q25301483")
            self.assertEqual(event.name, "Women's singles")
            self.assertEqual(event.year, 2016)

            self.assertIn("sport_badminton", res.sports)
            self.assertEqual(res.sports["sport_badminton"].name, "Badminton")

            self.assertIn("venue_riocentro_pavilion_4", res.venues)
            v_name = res.venues["venue_riocentro_pavilion_4"].name
            self.assertEqual(v_name, "Riocentro – Pavilion 4")

            self.assertIn("country_ESP", res.countries)
            self.assertEqual(res.countries["country_ESP"].code, "ESP")

            self.assertIn("person_carolina_marin", res.persons)
            p_name = res.persons["person_carolina_marin"].name
            self.assertEqual(p_name, "Carolina Marín")

            self.assertIn("entity_sport_badminton", res.entities)
            self.assertIn("entity_venue_riocentro_pavilion_4", res.entities)
            self.assertIn("entity_country_ESP", res.entities)
            self.assertIn("entity_person_carolina_marin", res.entities)

            # Integrity check
            errors = validate_extracted_graph(res)
            self.assertEqual(errors, [])
        finally:
            tmp_path.unlink()

    def test_valid_edge_generation(self) -> None:
        """Extracts valid directional and bidirectional edges with exact endpoints."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(SAMPLE_INDIVIDUAL_DOC) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)

            # BELONGS_TO: Event -> Sport
            self.assertIn(("Q25301483", "sport_badminton"), res.belongs_to_edges)

            # HELD_AT: Event -> Venue
            self.assertIn(
                ("Q25301483", "venue_riocentro_pavilion_4"), res.held_at_edges
            )

            # PARTICIPATED_IN: Person -> Event
            self.assertIn(
                ("Person", "person_carolina_marin", "Q25301483"),
                res.participated_in_edges,
            )

            # REPRESENTS: Person -> Country
            self.assertIn(
                ("Person", "person_carolina_marin", "country_ESP"),
                res.represents_edges,
            )

            # MENTIONS: Chunk <-> Entity
            self.assertIn(
                ("Q25301483#c0000", "entity_sport_badminton"), res.mentions_edges
            )
            self.assertIn(
                ("Q25301483#c0000", "entity_venue_riocentro_pavilion_4"),
                res.mentions_edges,
            )
            self.assertIn(
                ("Q25301483#c0000", "entity_person_carolina_marin"),
                res.mentions_edges,
            )

            # RESOLVES_TO: Entity -> TypedVertex
            self.assertIn(
                ("entity_sport_badminton", "Sport", "sport_badminton"),
                res.resolves_to_edges,
            )
            self.assertIn(
                (
                    "entity_venue_riocentro_pavilion_4",
                    "Venue",
                    "venue_riocentro_pavilion_4",
                ),
                res.resolves_to_edges,
            )
            self.assertIn(
                ("entity_person_carolina_marin", "Person", "person_carolina_marin"),
                res.resolves_to_edges,
            )
        finally:
            tmp_path.unlink()

    def test_unresolved_entity_handling(self) -> None:
        """Missing venues or vacant medals are tracked as unresolved."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(SAMPLE_MISSING_VENUE_AND_VACANT_DOC) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            self.assertEqual(res.stats.unresolved_venues, 1)
            self.assertEqual(res.stats.unresolved_medalists, 1)
            self.assertEqual(len(res.held_at_edges), 0)

            self.assertIn("person_nijat_rahimov", res.persons)
            self.assertIn("person_mohamed_ihab", res.persons)

            errors = validate_extracted_graph(res)
            self.assertEqual(errors, [])
        finally:
            tmp_path.unlink()

    def test_ambiguous_and_skipped_handling(self) -> None:
        """Non-Olympic documents are cleanly skipped without error."""
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(SAMPLE_NON_OLYMPIC_DOC) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            self.assertEqual(res.stats.skipped_non_olympic_docs, 1)
            self.assertEqual(len(res.events), 0)
            self.assertEqual(len(res.sports), 0)
            self.assertEqual(len(res.venues), 0)
        finally:
            tmp_path.unlink()

    def test_duplicate_prevention(self) -> None:
        """Duplicate athletes/venues across events merge into single vertices."""
        doc1 = json.loads(json.dumps(SAMPLE_INDIVIDUAL_DOC))
        doc2 = json.loads(json.dumps(SAMPLE_INDIVIDUAL_DOC))
        doc2["doc_id"] = "Q25301484"
        doc2["title"] = "Badminton at the 2016 Summer Olympics – Women's singles"
        doc2["text"] = doc2["text"].replace("Women's singles", "Women's singles")

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(doc1) + "\n")
            tmp.write(json.dumps(doc2) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            self.assertEqual(len(res.venues), 1)
            self.assertEqual(len(res.sports), 1)
            self.assertEqual(len(res.events), 2)
            self.assertEqual(len(res.held_at_edges), 2)
            errors = validate_extracted_graph(res)
            self.assertEqual(errors, [])
        finally:
            tmp_path.unlink()

    # --- Specific Tests Required for Correction 1 (Person vs Team) ---

    def test_case_1_solo_event(self) -> None:
        """Solo event: Synchronized swimming – Women's solo -> Person."""
        title = "Synchronized swimming at the 1992 Summer Olympics – Women's solo"
        event = "Women's solo"
        self.assertFalse(is_team_event(title, event, {"competitors": "53"}))

    def test_case_2_individual_event(self) -> None:
        """Individual event containing 'individual' -> Person."""
        title = (
            "Gymnastics at the 2016 Summer Olympics – "
            "Women's artistic individual all-around"
        )
        event = "Women's artistic individual all-around"
        self.assertFalse(is_team_event(title, event, {}))

    def test_case_3_sailing_49er(self) -> None:
        """Sailing 49er -> Team."""
        title = "Sailing at the 2016 Summer Olympics – 49er"
        event = "Men's 49er"
        self.assertTrue(is_team_event(title, event, {}))

    def test_case_4_sailing_470(self) -> None:
        """Sailing 470 -> Team."""
        title = "Sailing at the 2016 Summer Olympics – Men's 470"
        event = "Men's 470"
        self.assertTrue(is_team_event(title, event, {}))

    def test_case_5_nacra_17(self) -> None:
        """Sailing Nacra 17 -> Team."""
        title = "Sailing at the 2016 Summer Olympics – Nacra 17"
        event = "Nacra 17"
        self.assertTrue(is_team_event(title, event, {}))

    def test_case_6_yngling(self) -> None:
        """Sailing Yngling -> Team."""
        title = "Sailing at the 2008 Summer Olympics – Yngling"
        event = "Yngling"
        self.assertTrue(is_team_event(title, event, {}))

    def test_case_7_group_rhythmic_gymnastics(self) -> None:
        """Group rhythmic gymnastics -> Team."""
        title = (
            "Gymnastics at the 2016 Summer Olympics – "
            "Women's rhythmic group all-around"
        )
        event = "Women's rhythmic group all-around"
        self.assertTrue(is_team_event(title, event, {}))

    def test_case_8_normal_individual_event(self) -> None:
        """A normal individual event (Women's singles) -> Person."""
        title = "Badminton at the 2016 Summer Olympics – Women's singles"
        event = "Women's singles"
        self.assertFalse(is_team_event(title, event, {}))

    def test_case_9_normal_relay_team_event(self) -> None:
        """A normal relay/team event -> Team."""
        title = "Cycling at the 2012 Summer Olympics – Women's team pursuit"
        event = "Women's team pursuit"
        self.assertTrue(is_team_event(title, event, {}))

    # --- Specific Tests Required for Correction 2 (MENTIONS Provenance) ---

    def test_case_10_mentions_multi_chunk_provenance(self) -> None:
        """Entity in multiple chunks gets MENTIONS for each containing chunk."""
        # Create a document where Carolina Marín appears in chunk 0 and in chunk 1
        long_paragraph = "word " * 1200  # forces a chunk split at 768 tokens
        doc = {
            "doc_id": "Q_MULTI_TEST",
            "title": "Badminton at the 2016 Summer Olympics – Women's singles",
            "source": "wikipedia",
            "url": "https://example.org",
            "text": (
                "[Infobox Olympic event]\n"
                "  event: Women's singles\n"
                "  games: 2016 Summer\n"
                "  gold: Carolina Marín\n"
                "  goldNOC: ESP\n\n"
                "Carolina Marín played an outstanding first match. "
                f"{long_paragraph}\n\n"
                "In the decisive final, Carolina Marín secured the gold medal."
            ),
        }

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(doc) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            person_ent_id = "entity_person_carolina_marin"

            # Check that MENTIONS exists for chunk 0 and chunk 1
            chunks_with_person = [
                c_id for (c_id, ent_id) in res.mentions_edges if ent_id == person_ent_id
            ]
            self.assertIn("Q_MULTI_TEST#c0000", chunks_with_person)
            self.assertIn("Q_MULTI_TEST#c0001", chunks_with_person)
            self.assertGreaterEqual(len(chunks_with_person), 2)
        finally:
            tmp_path.unlink()

    def test_case_11_mentions_negative_case(self) -> None:
        """Chunk not containing the entity receives no MENTIONS edge."""
        long_paragraph = "word " * 1200  # forces a chunk split at 768 tokens
        doc = {
            "doc_id": "Q_NEG_TEST",
            "title": "Badminton at the 2016 Summer Olympics – Women's singles",
            "source": "wikipedia",
            "url": "https://example.org",
            "text": (
                "[Infobox Olympic event]\n"
                "  event: Women's singles\n"
                "  games: 2016 Summer\n"
                "  gold: Carolina Marín\n"
                "  goldNOC: ESP\n\n"
                "Body paragraph without name. "
                f"{long_paragraph}\n\n"
                "Final paragraph without name."
            ),
        }

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(doc) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            person_ent_id = "entity_person_carolina_marin"

            chunks_with_person = [
                c_id for (c_id, ent_id) in res.mentions_edges if ent_id == person_ent_id
            ]
            self.assertIn("Q_NEG_TEST#c0000", chunks_with_person)
            self.assertNotIn("Q_NEG_TEST#c0001", chunks_with_person)
            self.assertNotIn("Q_NEG_TEST#c0002", chunks_with_person)
        finally:
            tmp_path.unlink()

    def test_case_12_title_only_sport_no_fabricated_mention(self) -> None:
        """Sport absent from chunk text receives no fabricated MENTIONS edge."""
        doc = {
            "doc_id": "Q_TITLE_SPORT_TEST",
            "title": "Athletics at the 2008 Summer Olympics – Men's 110 metres hurdles",
            "source": "wikipedia",
            "url": "https://example.org",
            "text": (
                "[Infobox Olympic event]\n"
                "  event: Men's 110 metres hurdles\n"
                "  games: 2008 Summer\n"
                "  venue: National Stadium\n"
                "  gold: Dayron Robles\n"
                "  goldNOC: CUB\n\n"
                "The 110 metres hurdles competition took place over three days."
            ),
        }

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
            tmp.write(json.dumps(doc) + "\n")
            tmp_path = Path(tmp.name)

        try:
            res = extract_graph_from_corpus(tmp_path)
            sport_ent_id = "entity_sport_athletics"

            # Sport vertex and entity exist
            self.assertIn("sport_athletics", res.sports)
            self.assertIn(sport_ent_id, res.entities)

            # Edge BELONGS_TO exists: Event -> Sport
            self.assertIn(
                ("Q_TITLE_SPORT_TEST", "sport_athletics"), res.belongs_to_edges
            )
            # Edge RESOLVES_TO exists: Entity -> Sport
            self.assertIn(
                (sport_ent_id, "Sport", "sport_athletics"), res.resolves_to_edges
            )

            # ZERO chunk MENTIONS edges because "Athletics" is absent from text
            sport_mentions = [
                c_id for (c_id, ent_id) in res.mentions_edges if ent_id == sport_ent_id
            ]
            self.assertEqual(sport_mentions, [])
            self.assertEqual(res.stats.sport_title_only_no_mentions, 1)
        finally:
            tmp_path.unlink()

    def test_real_corpus_extraction_integrity(self) -> None:
        """Integration test on data/raw/corpus.jsonl verifying zero errors."""
        corpus_path = Path("/Users/yeshi/Desktop/tgh/data/raw/corpus.jsonl")
        self.assertTrue(corpus_path.is_file(), "corpus.jsonl must exist")

        res = extract_graph_from_corpus(corpus_path)
        self.assertEqual(res.stats.total_docs_processed, 2951)
        self.assertEqual(res.stats.olympic_event_docs, 2187)
        self.assertEqual(res.stats.skipped_non_olympic_docs, 764)
        self.assertEqual(res.stats.ambiguous_mappings, 0)
        self.assertEqual(len(res.events), 2187)
        self.assertEqual(len(res.sports), 42)
        self.assertEqual(res.stats.sport_title_only_no_mentions, 412)

        errors = validate_extracted_graph(res)
        self.assertEqual(errors, [], f"Referential errors: {errors[:10]}")


if __name__ == "__main__":
    unittest.main()
