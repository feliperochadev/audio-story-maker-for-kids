from __future__ import annotations
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

def _reduce_silence(
    audio: AudioSegment,
    min_silence_len_ms: int,
    keep_silence_ms: int,
    silence_thresh_db: float,
    fade_ms: int,
) -> AudioSegment:
    if min_silence_len_ms <= 0:
        return audio

    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
    )
    if not silences:
        return audio

    segments = []
    last_end = 0
    for start_ms, end_ms in silences:
        if start_ms > last_end:
            segments.append(audio[last_end:start_ms])
        last_end = end_ms
    if last_end < len(audio):
        segments.append(audio[last_end:])
    if not segments:
        return audio

    output = AudioSegment.empty()
    last_idx = len(segments) - 1
    for idx, segment in enumerate(segments):
        if fade_ms > 0:
            if idx > 0:
                segment = segment.fade_in(fade_ms)
            if idx < last_idx:
                segment = segment.fade_out(fade_ms)
        output += segment
        if idx < last_idx and keep_silence_ms > 0:
            output += AudioSegment.silent(duration=keep_silence_ms)
    return output


def wav_to_mp3(
    in_wav: str | Path,
    out_mp3: str | Path,
    bitrate: str = "192k",
    gain_db: float = -3.0,
    silence_min_len_ms: int = 0,
    silence_keep_ms: int = 0,
    silence_thresh_db: float = -40.0,
    silence_fade_ms: int = 0,
) -> Path:
    in_wav = Path(in_wav).expanduser().resolve()
    out_mp3 = Path(out_mp3).expanduser().resolve()
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(in_wav))
    audio = _reduce_silence(
        audio,
        silence_min_len_ms,
        silence_keep_ms,
        silence_thresh_db,
        silence_fade_ms,
    )
    if gain_db != 0:
        audio = audio.apply_gain(gain_db)

    audio.export(str(out_mp3), format="mp3", bitrate=bitrate)
    return out_mp3
