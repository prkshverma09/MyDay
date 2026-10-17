"""Grounding, retrieval fallback, and share boundaries. No network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reminisce.agent import ABSTAIN, answer_question, clips_for_viewer, daily_digest, is_grounded, story
from reminisce.store import Clip, set_shared, upsert_clip
from reminisce.vectors import rank_clips, tag_match


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


if __name__ == "__main__":
    unittest.main()
