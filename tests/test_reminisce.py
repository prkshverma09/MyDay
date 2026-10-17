"""Grounding, retrieval fallback, and share boundaries. No network."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from reminisce.agent import (
    ABSTAIN,
    answer_question,
    clips_for_viewer,
    clips_since,
    daily_digest,
    find_clips,
    is_grounded,
    story,
)
from reminisce.store import Clip, set_shared, upsert_clip
from reminisce.vectors import rank_clips, tag_match
from reminisce.vss_search import (
    SearchFilters,
    VssHit,
    VssSearchError,
    parse_moments,
    parse_query,
    parse_search_payload,
    parse_summary,
)


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

    def test_matching_journal_clip_beats_an_archive_street_scene(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])

        def searcher(_query: str):
            return [VssHit("seg-1", "A car is driving on the road.", 0.55, source="s3://example/a.mp4")]

        answer = answer_question(
            "did I have lunch today?",
            [lunch],
            [1.0, 0.0],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("clip_001"),
            searcher=searcher,
            fetch=lambda _source: None,
        )
        self.assertEqual(answer.retrieval_path, "journal")
        self.assertEqual(answer.citations, ["clip_001"])
        self.assertNotIn("car is driving", answer.text)
        self.assertTrue(answer.grounded)

    def test_archive_hit_answers_when_the_journal_has_nothing(self) -> None:
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [0.0, 1.0])

        def searcher(_query: str):
            return [VssHit("seg-77", "A forklift near a dock", 0.55, source="s3://example/dock.mp4")]

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

    def test_weak_archive_hit_is_not_served_as_a_memory(self) -> None:
        def searcher(_query: str):
            return [VssHit("seg-1", "The scene is a daytime view from a vehicle.", 0.19, source="s3://example/a.mp4")]

        answer = answer_question(
            "did I have lunch today?",
            [],
            [],
            role="owner",
            owner_id="user_123",
            carer_id="carer_daughter",
            llm=ScriptedModel("hit_1"),
            searcher=searcher,
            fetch=lambda _source: None,
        )
        self.assertEqual(answer.text, ABSTAIN)
        self.assertEqual(answer.citations, [])

    def test_model_picks_a_search_hit_by_its_short_label(self) -> None:
        def searcher(_query: str):
            return [
                VssHit("seg-1", "A car is driving on the road.", 0.45, source="s3://example/a.mp4"),
                VssHit("seg-2", "A person is walking a small white dog.", 0.40, source="s3://example/b.mp4"),
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
            return [VssHit("seg-1", "A car is driving on the road.", 0.40, source="s3://example/a.mp4")]

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

    def test_query_lifts_out_time_and_place(self) -> None:
        options = {"location": ["new_york", "san_francisco", "toronto"]}
        words, filters = parse_query("a person walking a dog in New York this week", options)
        self.assertEqual(words, "a person walking a dog")
        self.assertEqual(filters.time_window, "7d")
        self.assertEqual(filters.metadata, {"location": "new_york"})
        words, filters = parse_query("traffic light", options)
        self.assertEqual(words, "traffic light")
        self.assertFalse(filters.active())
        words, filters = parse_query("in the last hour", options)
        self.assertEqual(words, "in the last hour")
        self.assertEqual(filters.time_window, "1h")

    def test_filters_build_the_search_request(self) -> None:
        self.assertEqual(SearchFilters().request_fields(), {})
        fields = SearchFilters("24h", {"camera_id": "pie_cam-3", "location": "", "bogus": "x"}).request_fields()
        self.assertEqual(fields, {"time_filter": "24h", "metadata_filters": {"camera_id": "pie_cam-3"}})
        month = SearchFilters("30d").request_fields(now=datetime(2026, 10, 17, 12, 0, 0))
        self.assertEqual(month["time_filter"], "custom")
        self.assertEqual(month["custom_start_date"], "2026-09-17T12:00:00")
        self.assertEqual(month["custom_end_date"], "2026-10-17T12:00:00")

    def test_moments_and_summary_come_from_the_response(self) -> None:
        payload = {
            "chunk_results": [
                {
                    "preview_source": "s3://bucket/segments/cam_chunk_0020_segment_006_of_006.mp4",
                    "original_video": "s3://bucket/cam_chunk_0020.mp4",
                    "reasoning_content": "A car is driving on the road.",
                    "similarity_score": 0.29,
                    "best_match_start_sec": 25.0,
                    "best_match_end_sec": 30.0,
                    "matched_segment_count": 3,
                    "location": "new_york",
                    "camera_id": "nyc_streets_cam-1",
                },
                {"preview_source": "", "reasoning_content": "no source"},
            ],
            "llm_synthesis": {"response": "Cars drive past.", "model": "cosmos"},
        }
        moments = parse_moments(payload)
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].citation_id, "cam_chunk_0020_segment_006_of_006.mp4")
        self.assertEqual((moments[0].start_sec, moments[0].end_sec), (25.0, 30.0))
        self.assertEqual(moments[0].place, "new york")
        self.assertEqual(moments[0].matched_segments, 3)
        self.assertEqual(parse_summary(payload), ("Cars drive past.", "cosmos"))
        self.assertEqual(parse_summary({"llm_synthesis": {"response": "x", "error": "timeout"}}), ("", ""))

    def test_journal_search_keeps_matches_inside_the_window(self) -> None:
        now = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)
        lunch = _clip("clip_001", "Soup at the kitchen table", ["lunch"], [1.0, 0.0])
        lunch.recorded_at = "2026-10-17T12:30:00+00:00"
        old_walk = _clip("clip_003", "A walk outside", ["walk"], [0.0, 1.0])
        old_walk.recorded_at = "2026-10-01T09:00:00+00:00"
        self.assertEqual([clip.clip_id for clip in clips_since([lunch, old_walk], "24h", now)], ["clip_001"])
        self.assertEqual(len(clips_since([lunch, old_walk], "all", now)), 2)
        found = find_clips("soup", [lunch, old_walk], [1.0, 0.0], owner_id="user_123")
        self.assertEqual([hit.clip.clip_id for hit in found], ["clip_001"])
        self.assertEqual(find_clips("soup", [lunch], [1.0, 0.0], owner_id="someone_else"), [])
        walking = find_clips("a person walking a dog", [lunch, old_walk], [0.0, 0.0], owner_id="user_123")
        self.assertEqual([hit.clip.clip_id for hit in walking], ["clip_003"])

    def test_journal_search_ignores_a_clip_that_is_close_to_every_query(self) -> None:
        visit = _clip("clip_002", "My daughter visits with flowers", ["visit", "daughter"], [1.0, 0.0])
        self.assertEqual(find_clips("a forklift", [visit], [1.0, 0.0], owner_id="user_123"), [])


if __name__ == "__main__":
    unittest.main()
