"""Grounding, retrieval fallback, and share boundaries. No network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reminisce.agent import ABSTAIN, answer_question, clips_for_viewer, daily_digest, is_grounded, story
from reminisce.store import Clip, set_shared, upsert_clip
from reminisce.vectors import rank_clips, tag_match
from reminisce.vss_search import VssHit, VssSearchError, parse_search_payload


class ScriptedModel:
    def __init__(self, text: str) -> None:
        self.model = "scripted"
        self.text = text

    def complete(self, system: str, user: str) -> str:
        return self.text


def _clip(clip_id: str, description: str, tags: list[str], vector: list[float], shared: list[str] | None = None) -> Clip:
    return Clip(
        clip_id=clip_id,
        owner_id="user_123",
        recorded_at="2026-10-17T12:30:00+01:00",
        duration_s=6,
        cosmos_description=description,
        activity_tags=tags,
        embedding_id=f"emb_{clip_id}",
        embedding=vector,
        shared_with=list(shared or []),
        source_citation=clip_id,
        media_relpath=f"data/clips/{clip_id}.mp4",
        description_source="cosmos",
    )


class ReminisceTests(unittest.TestCase):
    def test_tag_boundary(self) -> None:
        self.assertTrue(tag_match("when did I last go for a walk?", ["walk"]))
        self.assertFalse(tag_match("walking the dog", ["walk"]))
        self.assertFalse(tag_match("did I have lunch today?", ["breakfast"]))

    def test_rank_prefers_matching_tag(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch", "soup"], [1.0, 0.0])
        breakfast = _clip("clip_005", "Tea and toast", ["breakfast"], [0.0, 1.0])
        hits = rank_clips("did I have lunch today?", [breakfast, lunch], [0.7071, 0.7071])
        self.assertEqual(hits[0].clip.clip_id, "clip_001")

    def test_bad_model_cannot_invent_a_memory(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])
        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("Mum ate in Paris. Source: clip_999."),
        )
        self.assertIn("Soup at the kitchen table", answer.text)
        self.assertIn("clip_001", answer.text)
        self.assertNotIn("Paris", answer.text)
        self.assertNotIn("clip_999", answer.text)
        self.assertTrue(answer.grounded)
        self.assertTrue(is_grounded(answer.text, answer.clips, {"clip_001"}))

    def test_unrelated_question_abstains(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])
        answer = answer_question(
            "did I fly a plane?",
            [lunch],
            [0.0, 1.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("NONE"),
        )
        self.assertEqual(answer.text, ABSTAIN)
        self.assertEqual(answer.citations, [])

    def test_carer_cannot_see_unshared_clips(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])
        visible = clips_for_viewer([lunch], role="carer", owner_id="user_123", carer_id="carer_daughter")
        self.assertEqual(visible, [])
        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="carer",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("clip_001"),
        )
        self.assertEqual(answer.text, ABSTAIN)
        self.assertNotIn("clip_001", answer.text)

    def test_carer_can_open_a_clip_that_was_sent(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0], ["carer_daughter"])
        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="carer",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("She ate in Paris. clip_999."),
        )
        self.assertIn("clip_001", answer.citations)
        self.assertIn("Soup at the kitchen table", answer.text)
        self.assertNotIn("Paris", answer.text)
        self.assertTrue(answer.grounded)

    def test_digest_uses_only_shared_clips(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0], ["carer_daughter"])
        walk = _clip("clip_003", "A walk outside", ["walk"], [0.0, 1.0])
        note = daily_digest([lunch, walk], owner_id="user_123", carer_id="carer_daughter", carer_name="my daughter")
        self.assertIn("clip_001", note.citations)
        self.assertNotIn("clip_003", note.citations)
        self.assertIn("Soup at the kitchen table", note.text)
        self.assertNotIn("A walk outside", note.text)
        self.assertTrue(note.grounded)

    def test_story_cites_every_clip_it_uses(self) -> None:
        clips = [
            _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0]),
            _clip("clip_003", "A walk outside", ["walk"], [0.0, 1.0]),
        ]
        told = story(clips, owner_id="user_123")
        self.assertEqual(told.citations, ["clip_001", "clip_003"])
        self.assertTrue(told.grounded)

    def test_share_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clips.json"
            clip = _clip("clip_003", "A walk outside", ["walk"], [0.0, 1.0])
            upsert_clip(clip, path)
            set_shared("clip_003", "carer_daughter", True, path)
            stored = set_shared("clip_003", "carer_daughter", False, path)
            self.assertEqual(stored.shared_with, [])
            set_shared("clip_003", "carer_daughter", True, path)
            from reminisce.store import load_clips

            again = load_clips(path)[0]
            self.assertEqual(again.shared_with, ["carer_daughter"])

    def test_search_payload_maps_clip_and_segment_ids(self) -> None:
        payload = {
            "results": [
                {
                    "clip_id": "clip_001",
                    "source": "s3://example/clip_001.mp4",
                    "similarity_score": 0.81,
                    "reasoning_content": "Soup at the kitchen table",
                },
                {
                    "id": "seg-77",
                    "source": "s3://example/dock.mp4",
                    "similarity_score": 0.44,
                    "reasoning_content": "A forklift near a dock",
                },
            ]
        }
        hits = parse_search_payload(payload)
        self.assertEqual(hits[0].clip_id, "clip_001")
        self.assertEqual(hits[0].similarity, 0.81)
        self.assertEqual(hits[1].citation_id, "seg-77")
        self.assertEqual(hits[1].clip_id, "")
        self.assertEqual(parse_search_payload({"results": []}), [])

    def test_vss_hit_on_a_journal_clip_is_the_search_path(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            return [
                VssHit(
                    citation_id="clip_001",
                    description="Soup at the kitchen table",
                    similarity=0.9,
                    source="s3://example/clip_001.mp4",
                    clip_id="clip_001",
                )
            ]

        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [0.0, 1.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("clip_001"),
            searcher=searcher,
        )
        self.assertEqual(answer.retrieval_path, "vss")
        self.assertIn("vss_search", answer.tools)
        self.assertNotIn("search_day", answer.tools)
        self.assertEqual(answer.citations, ["clip_001"])
        self.assertIn("Soup at the kitchen table", answer.text)
        self.assertTrue(answer.grounded)

    def test_empty_vss_falls_back_to_the_journal(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            return []

        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("NONE"),
            searcher=searcher,
        )
        self.assertEqual(answer.retrieval_path, "journal")
        self.assertIn("search_day", answer.tools)
        self.assertEqual(answer.citations, ["clip_001"])
        self.assertIn("Source: clip_001", answer.text)
        self.assertTrue(answer.grounded)

    def test_archive_hit_is_used_even_when_the_journal_has_clips(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            return [
                VssHit(
                    citation_id="seg-77",
                    description="A forklift near a dock",
                    similarity=0.55,
                    source="s3://example/dock.mp4",
                )
            ]

        answer = answer_question(
            "was there a forklift?",
            [lunch],
            [1.0, 0.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("hit_1"),
            searcher=searcher,
            fetch=lambda _source: None,
        )
        self.assertEqual(answer.retrieval_path, "vss")
        self.assertEqual(answer.citations, ["seg-77"])
        self.assertIn("A forklift near a dock", answer.text)
        self.assertNotIn("Soup", answer.text)
        self.assertTrue(answer.grounded)

    def test_model_picks_a_search_hit_by_its_short_label(self) -> None:
        def searcher(_query: str):
            return [
                VssHit("seg-1", "A car is driving on the road.", 0.20, source="s3://example/a.mp4"),
                VssHit("seg-2", "A person is walking a small white dog.", 0.18, source="s3://example/b.mp4"),
            ]

        answer = answer_question(
            "did I walk the dog?",
            [],
            [],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("hit_2"),
            searcher=searcher,
            fetch=lambda _source: None,
        )
        self.assertEqual(answer.selection, "model")
        self.assertEqual(answer.citations, ["seg-2"])
        self.assertIn("walking a small white dog", answer.text)
        self.assertNotIn("car is driving", answer.text)
        self.assertTrue(answer.grounded)

    def test_invented_label_falls_back_to_the_top_search_hit(self) -> None:
        def searcher(_query: str):
            return [VssHit("seg-1", "A car is driving on the road.", 0.20, source="s3://example/a.mp4")]

        answer = answer_question(
            "did I fly a plane?",
            [],
            [],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("hit_9. She flew to Paris."),
            searcher=searcher,
            fetch=lambda _source: None,
        )
        self.assertEqual(answer.citations, ["seg-1"])
        self.assertNotIn("Paris", answer.text)
        self.assertTrue(answer.grounded)

    def test_vss_error_uses_the_journal_and_keeps_the_note_off_the_sentence(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            raise VssSearchError("search HTTP 503")

        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("clip_001"),
            searcher=searcher,
        )
        self.assertEqual(answer.retrieval_path, "journal")
        self.assertEqual(answer.retrieval_note, "search HTTP 503")
        self.assertNotIn("503", answer.text)
        self.assertIn("clip_001", answer.citations)

    def test_segment_id_is_cited_when_the_journal_is_empty(self) -> None:
        def searcher(_query: str):
            return [
                VssHit(
                    citation_id="seg-77",
                    description="A forklift near a dock",
                    similarity=0.6,
                    source="s3://example/dock.mp4",
                )
            ]

        answer = answer_question(
            "was there a forklift?",
            [],
            [],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("seg-77"),
            searcher=searcher,
            fetch=lambda _source: Path("/tmp/dock.mp4"),
        )
        self.assertEqual(answer.retrieval_path, "vss")
        self.assertEqual(answer.citations, ["seg-77"])
        self.assertIn("A forklift near a dock", answer.text)
        self.assertIn("Source: seg-77", answer.text)
        self.assertEqual(answer.clips[0].media_relpath, "/tmp/dock.mp4")
        self.assertTrue(answer.grounded)
        self.assertTrue(is_grounded(answer.text, answer.clips, set()))

    def test_carer_does_not_receive_an_unshared_vss_clip(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            return [VssHit("clip_001", "Soup at the kitchen table", 0.9, clip_id="clip_001")]

        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="carer",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("clip_001"),
            searcher=searcher,
        )
        self.assertEqual(answer.text, ABSTAIN)
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.retrieval_path, "journal")


if __name__ == "__main__":
    unittest.main()
