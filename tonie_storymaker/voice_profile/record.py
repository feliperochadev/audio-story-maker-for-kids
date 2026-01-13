from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

@dataclass
class RecordOptions:
    seconds: int = 15
    samplerate: int = 22050
    channels: int = 1

def record_wav(out_path: str | Path, seconds: int = 15, samplerate: int = 22050, channels: int = 1) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = int(seconds * samplerate)
    print(f"Recording {seconds}s at {samplerate} Hz... Speak clearly and avoid background noise.")
    audio = sd.rec(frames, samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()

    # flatten to mono if needed
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio.reshape(-1)

    sf.write(str(out_path), audio, samplerate)
    print(f"Saved recording to: {out_path}")
    return out_path

def trim_silence(wav_path: str | Path, out_path: str | Path, threshold: float = 0.015, pad_ms: int = 150) -> Path:
    """
    Simple silence trimming by amplitude threshold. Keeps a small pad around speech.
    """
    wav_path = Path(wav_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    audio, sr = sf.read(str(wav_path), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    abs_audio = np.abs(audio)
    idx = np.where(abs_audio > threshold)[0]
    if len(idx) == 0:
        # nothing detected; copy as-is
        sf.write(str(out_path), audio, sr)
        return out_path

    pad = int((pad_ms / 1000.0) * sr)
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, len(audio) - 1)

    trimmed = audio[start:end+1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), trimmed, sr)
    return out_path

def normalize_peak(wav_path: str | Path, out_path: str | Path, target_peak: float = 0.9) -> Path:
    """
    Peak-normalize audio to target_peak (0..1 float).
    """
    wav_path = Path(wav_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    audio, sr = sf.read(str(wav_path), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        gain = target_peak / peak
        audio = np.clip(audio * gain, -1.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    return out_path
