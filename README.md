# MyDay — Ask Your Day

**A personal episodic memory for elderly people with early memory loss who live alone.**

Your day is recorded. You can ask it anything — and every answer comes with the actual moment as proof.

Built by **Team 20** at the Real-Time Video Agents Hack — LDN (17 Oct 2026).

## The problem

Repetitive questioning is one of the earliest and most frequent behaviours in Alzheimer's and early memory loss: *"Has Sarah been today?" "Where are my keys?" "Did I lock the door?"* — asked again and again, driven by genuine anxiety and a need for reassurance. Family carers answer patiently… until they can't. Existing assistants (e.g. smart-glasses companions) focus on prompting what to do **next**. Nobody gives back what **already happened** — with evidence.

## The niche

- **Who:** elderly people with early memory loss, living alone at home
- **Second user / buyer:** their adult children, who live elsewhere and currently field the same worried phone calls all day
- **Surface:** not a general assistant — five reassurance questions, answered calmly, identically, the 1st time and the 40th:
  1. "Has Sarah been today?" *(people)*
  2. "Where are my keys / glasses / hearing aid?" *(lost objects)*
  3. "Did I lock the door?" *(safety checks)*
  4. "Is the stove off?" *(safety checks)*
  5. "When is my appointment?" *(what people told her)*
- **Interface:** the TV — voice questions, huge captions (hearing loss), nothing new to learn. The app is for the family: a nightly recap and summary.

Explicitly **not**: care homes, late-stage care, medication management, an all-purpose companion, or a medical device.

## How it works

1. **Capture** — a POV phone and fixed door/kitchen cameras record the day in short chunks (10–30s).
2. **Understand** — every chunk is captioned, transcribed and structured into a *memory entry*: time, place, people, action, objects, one plain statement of what happened.
3. **Ask** — the user asks in plain language. The agent retrieves the matching moments and answers in one short sentence **with the source clip attached**.
4. **Abstain** — if it isn't recorded, it says so: *"I don't have that recorded."* The agent never invents a memory — for this user, a confident hallucination is the worst possible failure.
5. **Act** — appointments mentioned in conversation ("Dentist, Thursday 10am") become reminders; a stove left on in an empty kitchen triggers a spoken alert.
6. **Recap** — every evening, a short "Your Day" film replays the day's key moments for the user (reviewing the day is the ritual shown to support autobiographical recall) and a summary goes to the family.

## Architecture

```
POV phone + fixed door/kitchen cameras
        │  (10–30s chunks, live)
        ▼
VAST AI OS ── ingestion, one unified timeline, vector index, pipeline orchestration
        │
        ├──► YOLO ───────────────── per-frame detection/tracking: people, keys, glasses,
        │                            hearing aid, cat, mug → "object last-seen" records
        ├──► NVIDIA Cosmos Reason ─ per-chunk understanding: scene, activity, people,
        │                            why it might matter → structured memory entries
        └──► Speech-to-text ──────── transcripts ("Dentist, Thursday 10am")
        │
        ▼  memory entries + transcripts embedded → semantically searchable in VAST
User asks (voice, on TV)
        ▼
W&B agent ── plans the query → semantic search retrieves moments → composes ONE
             short grounded answer + the clip as proof — or abstains.
             Side-jobs: reminder extraction · stove watch-list alerts ·
             nightly recap + family summary. Every Q&A logged/traced in W&B.
        ▼
CoreWeave GPUs run the heavy inference (Cosmos + batch YOLO) underneath.
```

**Per-tech one-liners:** VAST = the memory's filing system and plumbing · Cosmos = the eyes that understand · YOLO = the tracker that never loses the keys · semantic search = how Margaret asks · W&B = the brain that answers, acts, and keeps receipts · CoreWeave = the GPU muscle · Cursor = build tooling.

### Core data contracts

```jsonc
// Segment        { clip_id, camera, place, t_start, t_end, file }
// Memory entry   { clip_id, t_start, t_end, place, people[], action, objects[],
//                  statement, transcript, answers[], importance }
// Answer         { text, clip_id, t_start, confidence, abstain }
```

## Build pipelines (parallel workstreams)

| Pipeline | Scope | Owner |
|---|---|---|
| 0 — Data collection | Film & label the day's clips | Team |
| 1 — Video data storage | Ingestion, storage, unified timeline (VAST) | Prakash Verma |
| 2 — Video → YOLO → text | Object/person detection & last-seen records *(optional layer)* | — |
| 3 — Video → Cosmos → text | Per-chunk captions & understanding | Alec Y |
| 4 — Text → LLM → Memory | Text + transcripts → structured memory entries, last-seen & coverage stores | Rosemary Ng + Prakash Verma |
| 5 — Query LLM → Memory | Question answering over the memory store, with proof & abstention | F L |

Every pipeline builds against the contracts above with mock data, so nobody blocks anybody; integration is by checkpoint, not at the deadline.

## Demo (3 minutes)

Meet Margaret. Her morning was recorded — now she asks it questions.

- **Ask** — "Where are my keys?" → *"In the blue bowl, 9:12"* + the clip plays
- **Ask + act** — "Who came this morning?" → *"Sarah, your daughter"* + clip, and the **Thursday 10am dentist reminder** appears, extracted from what Sarah said
- **Search** — "Show me everything with the cat" → the timeline filters to the moment
- **Ask again** — "Has Sarah been today?" — the same calm answer, the same clip. As many times as she needs
- **Honesty** — "Did I lock the front door?" → *"I don't have that recorded."*
- **Spot + act** — kitchen feed: stove on, nobody there → a spoken reminder fires
- **Recap** — the "Your Day" film plays; the family summary card is shown

> Wispy reminds you what's next. **MyDay gives you back what already happened — with proof.**

## Trust & safety

- **Grounded answers only** — every answer carries its source clip and timestamp
- **Abstention by design** — low retrieval confidence → "I don't have that recorded"
- **A memory aid, not a medical device** — no diagnosis, no medication claims
- **Privacy** — home footage stays the family's; recap sharing is opt-in

## Status

Hackathon build — work in progress (demos 16:30, 17 Oct 2026).

## Perception payload

`pipeline/video_to_text.py` splits a video into 5-second windows and runs Cosmos Reason and YOLO11 in parallel. It writes JSONL whose `llm_input` field is the text a later memory LLM should read. This repo does not call that LLM. `library/example-descriptions/annotations.jsonl` is a 100-window sample of that contract, taken from footage already in the index.

## Getting started

```bash
git clone https://github.com/prkshverma09/MyDay.git
cd MyDay
```

Setup and run instructions will be added once the project has code.

## Contributing

1. Create a branch from `main`.
2. Make your changes and commit them with a clear message.
3. Open a pull request against `main`.
