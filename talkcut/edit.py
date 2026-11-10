"""Deleted words -> cut ranges -> rendered audio.

All cut decisions live here. The editor only sends the indices of deleted words, so preview
and export can never disagree about what gets removed.
"""

import bisect
import os
import re
import subprocess
import tempfile

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "hmm", "mm", "mhm"}
PAD = 0.05  # seconds of silence kept next to a surviving word, so cuts don't clip it
FADE = 0.01  # fade at each joint; hides the click a hard cut makes
FRAME = 0.01  # seconds per loudness frame used for snapping


def find_fillers(words):
    """Indices of words that are filler sounds, ignoring case and punctuation."""
    return [i for i, w in enumerate(words) if re.sub(r"[^\w]", "", w["text"].lower()) in FILLERS]


def plan_cuts(words, deleted, duration, pad=PAD, max_pause=None):
    """Turn deleted word indices into (start, end) ranges to remove, in seconds.

    Each run of adjacent deleted words becomes one cut, which also takes the silence around
    it. `pad` seconds of that silence are left next to the surviving neighbours, but the pad
    never reaches into a deleted word: if the gap is tighter than the pad, the cut starts or
    ends exactly at the deleted word.
    """
    deleted = set(deleted)
    cuts = []
    i, n = 0, len(words)
    while i < n:
        if i not in deleted:
            if max_pause is not None and i > 0 and (i - 1) not in deleted:
                prev_end = words[i - 1]["end"]
                curr_start = words[i]["start"]
                if curr_start - prev_end > max_pause:
                    start = prev_end + max_pause
                    end = curr_start
                    if end > start:
                        cuts.append((round(start, 3), round(end, 3)))
            i += 1
            continue
        j = i
        while j < n and j in deleted:
            j += 1
        start = min(words[i - 1]["end"] + pad, words[i]["start"]) if i > 0 else 0.0
        end = max(words[j]["start"] - pad, words[j - 1]["end"]) if j < n else duration
        if end > start:
            cuts.append((round(start, 3), round(end, 3)))
        i = j
    return cuts


def keep_ranges(cuts, duration):
    """Inverse of cuts: the audio intervals to PRESERVE, spanning [0, duration]."""
    keeps = []
    t = 0.0
    for start, end in cuts:
        if start > t:
            keeps.append((t, start))
        t = max(t, end)
    if t < duration:
        keeps.append((t, duration))
    return keeps


def snap_to_silence(cuts, audio_data, sample_rate, window_ms=30):
    """Nudge cut boundaries to the quietest point within +/- window_ms.

    Refines Whisper word boundaries using actual audio levels.
    """
    if not cuts or audio_data is None or len(audio_data) == 0:
        return cuts

    window_samples = int(sample_rate * (window_ms / 1000.0))
    n_samples = len(audio_data)

    snapped = []
    for start, end in cuts:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # Snap start boundary
        best_start = start_sample
        min_energy = float("inf")
        search_from = max(0, start_sample - window_samples)
        search_to = min(n_samples - 1, start_sample + window_samples)
        if search_to > search_from:
            for i in range(search_from, search_to):
                energy = abs(audio_data[i])
                if energy < min_energy:
                    min_energy = energy
                    best_start = i

        # Snap end boundary
        best_end = end_sample
        min_energy = float("inf")
        search_from = max(0, end_sample - window_samples)
        search_to = min(n_samples - 1, end_sample + window_samples)
        if search_to > search_from:
            for i in range(search_from, search_to):
                energy = abs(audio_data[i])
                if energy < min_energy:
                    min_energy = energy
                    best_end = i

        snapped.append((round(best_start / sample_rate, 3), round(best_end / sample_rate, 3)))

    return snapped


def edited_path(input_path):
    base, ext = os.path.splitext(input_path)
    return f"{base}.edited{ext}"


def final_cuts(project, levels=None):
    """The cuts preview and export both use: planned from words, snapped to pauses."""
    words = project["words"]
    cuts = plan_cuts(words, project["deleted"], project["duration"], max_pause=project.get("max_pause"))
    return snap_to_silence(cuts, levels, words) if levels is not None else cuts


def export(project, input_path, output_path):
    cuts = final_cuts(project)
    render(input_path, output_path, cuts)


def render(input_path, output_path, cuts):
    """Concat audio segments between cuts using ffmpeg."""
    duration = probe_duration(input_path)
    keeps = keep_ranges(cuts, duration)

    if not keeps:
        return

    filter_complex = ""
    for i, (start, end) in enumerate(keeps):
        filter_complex += f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
    for i in range(len(keeps)):
        filter_complex += f"[a{i}]"
    filter_complex += f"concat=n={len(keeps)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex, "-map", "[out]", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def probe_duration(input_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())
