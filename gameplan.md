# Gameplan — Integrating shorts_analyzer into shorts-auto-editor

## Context

This project (shorts-auto-editor) cuts down raw screen recordings into finished
shorts. The sibling project `shorts_analyzer` (port 9021) studies *published*
YouTube shorts to figure out what works on a per-channel basis. We want to
feed that "what works" signal back into shorts-auto-editor so the cuts come out
better and ship with titles that match the channel's winning patterns.

Most of the time, both apps run inside `youtube_hub` (which already supervises
both api_servers — see `youtube_hub/service_defs.py`). The hub orchestrates;
shorts-auto-editor is reached at `http://localhost:9020`, the analyzer at
`http://localhost:9021`. The integration should assume the hub is running both
services, but shorts-auto-editor must still work standalone if 9021 is down.

## The fundamental mismatch

- shorts_analyzer is keyed on **published `video_id`s**.
- shorts-auto-editor operates on **raw recordings** that have no `video_id` yet.

So the integration is *not* "look up this clip" — it's **"pull channel-level
guidance and apply it as priors at cut time."** The link between a raw
recording and a `video_id` only exists *after* publish, which is what makes
the feedback loop (Phase 4 below) interesting.

---

## Phase 0 — Bridge first (do this before anything else)

The point of Phase 0 is to make shorts-auto-editor *able to talk to* the analyzer
at all, with no behavior change yet. Everything later depends on this working
cleanly with the hub up.

### 0.1 Analyzer client module

Create `analyzer_client.py` at the project root. Thin wrapper over
`http://localhost:9021` exposing only what we'll actually call:

- `health() -> bool` — `GET /health`, swallow connection errors, return False
- `list_results() -> list[dict]` — `GET /results`
- `read_result(name: str) -> dict` — `GET /results/read?name=...`
- `get_videos(output: str) -> dict` — `GET /videos?output=...`

Constructor takes `base_url` (default `http://localhost:9021`) and a short
timeout (2s for health, 10s for reads). Every method must degrade gracefully
when the analyzer is down: log a warning, return `None`/`[]`, never raise into
the caller's pipeline.

### 0.2 Channel resolution

The analyzer's output files are keyed by channel handle (e.g.
`PeepingOtter.json`). shorts-auto-editor needs to know *which* channel a given
batch belongs to. Two ways to wire it:

1. **Hub-driven (preferred):** the hub already knows the active channel from
   `hub_settings.json`. Add a `channel_handle` field to the shorts-auto-editor
   `/process` request body in `api_server.py`, pass it through to
   `VideoProcessor.process_single_video`.
2. **Standalone fallback:** add a `channel_handle` field to the GUI
   (`ui/ui_components.py`) so manual runs still work.

Resolution rule: `analyzer_output_filename = f"{channel_handle}.json"`. If
`channel_handle` is None, skip all analyzer calls and run the legacy
pipeline. **No analyzer dependency is ever required for cuts to succeed.**

### 0.3 Local cache

The analyzer's synthesis/tailwind data changes slowly (operator-triggered
reruns only — see `shorts_analyzer/API.md`). Cache reads in
`~/.shorts-auto-editor/cache/` keyed by `{channel_handle}.{filetype}.json` with
the file's `modified` timestamp from `GET /results`. This keeps batch
processing fast and survives the analyzer being briefly down.

### 0.4 Hub plumbing

Update `youtube_hub/subtitler_routes.py` (or wherever the hub posts to 9020)
to forward `channel_handle` from `hub_settings.json` into the shorts-auto-editor
`/process` call. Add a hub setting if one doesn't already exist.

**Done when:** from inside the hub, hitting "process" on a video logs
`[analyzer] connected, channel=PeepingOtter, synthesis fresh` (or
`[analyzer] unavailable, proceeding without guidance`) and the cut still
completes either way.

---

## Phase 1 — YouTube metadata title generation

The README's open To-Do. The title is the YouTube metadata title only — the
text that goes under the video on YouTube. **Nothing gets burned into the
video pixels.** Subtitles and onomatopoeia stay as-is.

The publisher (`youtube_shorts_publisher`) is explicitly out of scope for
this phase. We treat `shorts_data/shorts_metadata_N.json` as the contract
surface — when shorts-auto-editor writes the title into that file, Phase 1 is
done. Hooking it up to the publisher is a later, separate piece of work.

### 1.1 Title generator

New module `title_generator.py`. Inputs:

- The dialogue transcript from Phase 1 of `core/video_processor.py` (already
  produced — currently just used for subtitles)
- The trim summary / final duration
- The channel's `synthesis.json` content via `analyzer_client`

Calls `llm/gemini_text_generator.py` with a prompt that includes:
- Recent winning title formulas from synthesis (PeepingOtter-specific)
- Hook patterns the channel's top performers use
- The transcript and trim summary for the current cut

Returns a single chosen title. The prompt itself is where the actual quality
work lives — once the wiring is proven end-to-end, we'll iterate on the
prompt with real cuts to see what the model produces and tune from there.
Cache the synthesis block across the batch (it's identical for every video
in a single run).

### 1.2 Wire into the pipeline

In `core/video_processor.py`, after Phase 1 (transcription) and before
trim execution, call `title_generator.generate(...)`. Write the chosen
title into the per-video entry of `shorts_data/shorts_metadata_N.json`
under a new `title` field, alongside `file_info`. Also surface it in the
hub UI's file row (the `FileEntry.title` field already exists in
[subtitler-page.component.ts](../youtube_hub/src/app/components/subtitler-page/subtitler-page.component.ts)
and renders as `→ "title"` next to the filename — it's just never set today).

### 1.3 Failure modes — strict no-fallback

Title generation is strictly best-effort. **If anything goes wrong, the
title field is omitted from the metadata entirely.** No fallback to
transcript-only, no placeholder string, no "Untitled". The operator can
look at the missing field and decide what to do.

Specifically: omit the title field when —
- The analyzer is unreachable, or
- The channel's `synthesis.json` is missing on the analyzer, or
- The Gemini title call fails or returns nothing usable, or
- The transcript is empty.

In every case the cut itself still ships normally. Log a single warning per
video so the operator can see *why* the title is missing, but never raise
into the pipeline.

### Done when

A finished cut has its YouTube metadata title written into
`shorts_data/shorts_metadata_N.json` when the analyzer is up and producing
useful output, no `title` field at all when it isn't, and the hub's file
row reflects whichever case applies.

---

## Phase 2 — Per-clip hook scoring (analyzer-side endpoint)

Once titles work, bias the cut itself toward openings that match the
channel's winning hook patterns.

### 2.1 New analyzer endpoint

Add to `shorts_analyzer/api_server.py`:

```
POST /score-hook
{
  "output": "PeepingOtter.json",
  "transcript": [{ "start": 0.0, "end": 1.2, "text": "..." }, ...],
  "candidate_starts": [0.0, 1.5, 3.2]
}
→
{
  "ranked": [
    { "start": 1.5, "score": 0.81, "why": "matches 'question hook' pattern" },
    { "start": 0.0, "score": 0.42, "why": "..." },
    ...
  ]
}
```

Text-only Gemini call, reuses the channel's tailwind/synthesis context as
system prompt. Cheap. Synchronous (no job slot needed — different from the
expensive `/rerun/*` endpoints).

### 2.2 shorts-auto-editor consumption

In `clip_editor/intelligent_trimmer.py`, after generating candidate trim
points, call `analyzer_client.score_hook(...)`. Use the score as one input
among the existing heuristics — **do not let it fully override** the
intelligent trimmer; treat it as a tiebreaker / re-ranking signal. Log which
candidate was picked and why.

**Done when:** the trim log shows hook scores next to each candidate and the
selected start matches the highest-ranked candidate (with the existing
trimmer constraints respected).

---

## Phase 3 — Close the loop after publish

This is what makes the system *learn*. shorts-auto-editor's outputs eventually
become the analyzer's training corpus.

### 3.1 Stamp video_id back

After `youtube_shorts_publisher` (sibling project) uploads a cut, write the
resulting `video_id` into the corresponding entry of
`shorts_data/shorts_metadata_N.json`. Likely lives in the publisher or in
hub pipeline glue, not here — but shorts-auto-editor needs to make the metadata
record findable (stable filename + a unique `cut_id` field would help).

### 3.2 Targeted ingest endpoint

Add to `shorts_analyzer/api_server.py`:

```
POST /ingest
{ "channel_url": "...", "video_id": "abc123" }
```

Runs Phase 1 + Phase 2 just for that single ID, skipping the full channel
scrape. Reuses the existing job slot and progress reporting.

### 3.3 Hub trigger

In `youtube_hub/pipeline/`, after a successful publish, fire `POST /ingest`.
Once analytics have had ~24h to accumulate, fire `POST /rerun/analytics`
for that `video_id`. Now the next batch's title generation (Phase 1) and
hook scoring (Phase 2) sees the new short's performance data.

**Done when:** publishing a short results in it appearing in the analyzer's
`/videos` listing within minutes, with analytics filling in over the next
day.

---

## What stays out of scope

- **Title A/B testing or thumbnail generation** — separate problem, separate
  tools. Possibly belongs in `youtube_shorts_publisher` if anywhere.
- **Replacing the intelligent trimmer with the analyzer.** The trimmer makes
  decisions from the raw video; the analyzer reasons about published
  performance. They're complementary, not substitutes. Phase 2 is a
  re-ranking layer, not a replacement.
- **Auto-running expensive analyzer reruns from shorts-auto-editor.** All
  `/rerun/analysis` and similar are operator-triggered (per the analyzer's
  API contract). Don't let cut-time code start them.

---

## Sequencing summary

| Phase | What | Where the work lives |
|------|------|---------------------|
| 0 | Bridge: client, channel handle, cache, fallback | this repo + `youtube_hub` |
| 1 | Title generation + overlay | this repo |
| 2 | Hook scoring | `shorts_analyzer` (new endpoint) + this repo (consumer) |
| 3 | Publish-time feedback loop | `youtube_hub` + `shorts_analyzer` (new endpoint) |

Phase 0 is the only one that's strictly required as a prerequisite for
everything else. Phase 1 ships the visible win. Phases 2 and 3 are
independent of each other and can be tackled in either order once 1 is solid.
