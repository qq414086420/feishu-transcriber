"""Align ASR segments with speaker diarization segments."""

from audio_transcribe.diarizer import DiarizationSegment


def _overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration between two time intervals.

    Returns the duration of intersection, or 0 if intervals do not overlap.
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)


def align(
    asr_segments: list[dict],
    diarization_segments: list[DiarizationSegment],
) -> list[dict]:
    """Assign speaker labels to ASR segments based on diarization overlap.

    For each ASR segment, finds the diarization segment with maximum time
    overlap and assigns that speaker. Segments with None timestamps or
    no overlapping diarization get "UNKNOWN".

    Does NOT mutate input dicts — creates new dicts with speaker added.
    """
    if not asr_segments:
        return []

    results: list[dict] = []
    for seg in asr_segments:
        start = seg.get("start")
        end = seg.get("end")

        if start is None or end is None:
            results.append({**seg, "speaker": "UNKNOWN"})
            continue

        if not diarization_segments:
            results.append({**seg, "speaker": "UNKNOWN"})
            continue

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for dia in diarization_segments:
            ov = _overlap(start, end, dia.start, dia.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dia.speaker

        results.append({**seg, "speaker": best_speaker})

    return results


def build_verbatim(segments: list[dict]) -> str:
    """Format aligned segments as a verbatim transcript.

    Each segment is formatted as ``[speaker]: text\\n`` and all lines
    are joined. Empty input returns an empty string.
    """
    if not segments:
        return ""
    return "".join(f"[{seg['speaker']}]: {seg['text']}\n" for seg in segments)
