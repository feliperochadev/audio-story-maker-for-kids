from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from pydub import AudioSegment

from .text_utils import split_into_chapters, split_for_tts, clean_text, normalize_for_tts
from .audio_utils import wav_to_mp3

_WORKER_TTS = None
_WORKER_CONFIG = None


def _batch_chunks(chunks: List[str], max_chars: int) -> List[str]:
    if max_chars <= 0:
        return chunks
    batches: List[str] = []
    current: List[str] = []
    current_len = 0
    for chunk in chunks:
        chunk_len = len(chunk)
        if not current:
            current = [chunk]
            current_len = chunk_len
            continue
        if current_len + 1 + chunk_len <= max_chars:
            current.append(chunk)
            current_len += 1 + chunk_len
        else:
            batches.append(" ".join(current).strip())
            current = [chunk]
            current_len = chunk_len
    if current:
        batches.append(" ".join(current).strip())
    return batches


def _cache_speaker_embedding(tts, speaker_wav: Path) -> None:
    manager = getattr(tts.synthesizer.tts_model, "speaker_manager", None)
    if not manager or not hasattr(manager, "compute_embedding_from_clip"):
        return

    original_fn = manager.compute_embedding_from_clip
    cache = {}

    def _key(value):
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        return str(value)

    key = _key(speaker_wav)
    cache[key] = original_fn(speaker_wav)

    def _cached_compute_embedding_from_clip(value):
        k = _key(value)
        if k not in cache:
            cache[k] = original_fn(value)
        return cache[k]

    manager.compute_embedding_from_clip = _cached_compute_embedding_from_clip


def _synthesize_with_tts(
    tts,
    index: int,
    chunk: str,
    title: str,
    speaker_wav: Path,
    language: str,
    out_dir: Path,
    mp3_bitrate: str,
    output_gain_db: float,
    silence_min_len_ms: int,
    silence_keep_ms: int,
    silence_thresh_db: float,
    silence_fade_ms: int,
) -> Path:
    wav_path = out_dir / f"{title} - Part {index:02d}.wav"
    mp3_path = out_dir / f"{title} - Part {index:02d}.mp3"

    tts.tts_to_file(
        text=chunk,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(wav_path),
    )

    wav_to_mp3(
        wav_path,
        mp3_path,
        bitrate=mp3_bitrate,
        gain_db=output_gain_db,
        silence_min_len_ms=silence_min_len_ms,
        silence_keep_ms=silence_keep_ms,
        silence_thresh_db=silence_thresh_db,
        silence_fade_ms=silence_fade_ms,
    )

    try:
        wav_path.unlink(missing_ok=True)
    except Exception:
        pass

    return mp3_path


def _prepare_torch_safe_globals():
    try:
        from torch.serialization import add_safe_globals, safe_globals
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig
        from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig, BaseTrainingConfig
    except Exception:
        return None

    safe_globals_list = []
    for candidate in (XttsConfig, XttsAudioConfig, BaseDatasetConfig, BaseAudioConfig, BaseTrainingConfig):
        if candidate:
            safe_globals_list.append(candidate)

    if safe_globals_list:
        add_safe_globals(safe_globals_list)
        return safe_globals(safe_globals_list)
    return None


def _worker_init(config: dict) -> None:
    global _WORKER_TTS, _WORKER_CONFIG
    _WORKER_CONFIG = config
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from TTS.api import TTS

    safe_ctx = _prepare_torch_safe_globals()
    if safe_ctx:
        with safe_ctx:
            _WORKER_TTS = TTS(config["tts_model"], gpu=config["tts_use_gpu"], progress_bar=False)
    else:
        _WORKER_TTS = TTS(config["tts_model"], gpu=config["tts_use_gpu"], progress_bar=False)


def _synthesize_chunk(args: tuple[int, str, str]) -> tuple[int, str]:
    index, chunk, title = args
    s = _WORKER_CONFIG

    out_dir = Path(s["out_dir"]).expanduser().resolve()
    mp3_path = _synthesize_with_tts(
        _WORKER_TTS,
        index=index,
        chunk=chunk,
        title=title,
        speaker_wav=Path(s["speaker_wav"]),
        language=s["language"],
        out_dir=out_dir,
        mp3_bitrate=s["mp3_bitrate"],
        output_gain_db=s["output_gain_db"],
        silence_min_len_ms=s["silence_min_len_ms"],
        silence_keep_ms=s["silence_keep_ms"],
        silence_thresh_db=s["silence_thresh_db"],
        silence_fade_ms=s["silence_fade_ms"],
    )

    return index, str(mp3_path)
from ..config import get_settings

def synthesize_story_to_mp3s(
    story_text: str,
    title: str,
    speaker_wav: str | Path,
    language: Optional[str] = None,
    out_dir: str | Path = "output",
) -> List[Path]:
    """
    Voice-clone a story into multiple MP3 chapter files.
    Returns list of created MP3 paths.
    """
    # Ensure torch loads are not forced into weights_only mode (PyTorch 2.6+).
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from TTS.api import TTS

    s = get_settings()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    language = (language or s.default_language).strip().lower()
    story_text = clean_text(story_text)

    chapters = split_into_chapters(
        story_text,
        max_chars=s.chapter_max_chars,
        chapter_keywords=s.chapter_split_keywords,
    )
    if s.tts_strip_punctuation:
        chapters = [normalize_for_tts(clean_text(ch)) for ch in chapters]
    if not chapters:
        raise ValueError("Story text is empty after cleaning.")

    safe_globals_context = _prepare_torch_safe_globals()

    # Load model once
    if safe_globals_context:
        with safe_globals_context:
            tts = TTS(s.tts_model, gpu=s.tts_use_gpu, progress_bar=False)
    else:
        tts = TTS(s.tts_model, gpu=s.tts_use_gpu, progress_bar=False)
    print(f"TTS device: {'cuda' if s.tts_use_gpu else 'cpu'}")

    speaker_wav = Path(speaker_wav).expanduser().resolve()
    if not speaker_wav.exists():
        raise FileNotFoundError(f"Speaker wav not found: {speaker_wav}")
    _cache_speaker_embedding(tts, speaker_wav)

    created: List[Path] = []
    total = len(chapters)
    overall_bar = tqdm(total=total, desc="Chapters", unit="chapter")

    if s.tts_workers <= 0:
        cpu_count = os.cpu_count() or 1
        workers = max(1, cpu_count - 1)
    else:
        workers = max(1, s.tts_workers)
    if s.tts_use_gpu and workers > 1:
        print("GPU detected; forcing TTS_WORKERS=1 to avoid contention.")
        workers = 1

    if workers == 1:
        for i, chapter_text in enumerate(chapters, start=1):
            stop_event = threading.Event()
            chapter_start = time.monotonic()
            chapter_bar = tqdm(
                total=0,
                desc=f"Chapter {i} elapsed",
                unit="s",
                leave=False,
                bar_format="{desc}: {n_fmt}s",
            )

            def tick_progress() -> None:
                while not stop_event.wait(1):
                    elapsed = int(time.monotonic() - chapter_start)
                    chapter_bar.n = elapsed
                    chapter_bar.refresh()

            progress_thread = threading.Thread(target=tick_progress, daemon=True)
            progress_thread.start()

            chapter_chunks = split_for_tts(chapter_text, max_chars=s.tts_max_chars)
            combined = AudioSegment.empty()
            for chunk in chapter_chunks:
                wav_path = out_dir / f"{title} - Chapter {i:02d}.wav"
                tts.tts_to_file(
                    text=chunk,
                    speaker_wav=str(speaker_wav),
                    language=language,
                    file_path=str(wav_path),
                    split_sentences=False,
                )
                combined += AudioSegment.from_file(str(wav_path))
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

            chapter_wav = out_dir / f"{title} - Chapter {i:02d}.wav"
            combined.export(str(chapter_wav), format="wav")
            mp3_path = out_dir / f"{title} - Chapter {i:02d}.mp3"
            wav_to_mp3(
                chapter_wav,
                mp3_path,
                bitrate=s.mp3_bitrate,
                gain_db=s.output_gain_db,
                silence_min_len_ms=s.silence_min_len_ms,
                silence_keep_ms=s.silence_keep_ms,
                silence_thresh_db=s.silence_thresh_db,
                silence_fade_ms=s.silence_fade_ms,
            )
            try:
                chapter_wav.unlink(missing_ok=True)
            except Exception:
                pass

            created.append(mp3_path)

            stop_event.set()
            progress_thread.join(timeout=1)
            chapter_bar.close()
            overall_bar.update(1)
    else:
        worker_config = {
            "tts_model": s.tts_model,
            "tts_use_gpu": s.tts_use_gpu,
            "speaker_wav": str(speaker_wav),
            "language": language,
            "out_dir": str(out_dir),
            "mp3_bitrate": s.mp3_bitrate,
            "output_gain_db": s.output_gain_db,
            "silence_min_len_ms": s.silence_min_len_ms,
            "silence_keep_ms": s.silence_keep_ms,
            "silence_thresh_db": s.silence_thresh_db,
            "silence_fade_ms": s.silence_fade_ms,
        }
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(worker_config,)) as executor:
            futures = [
                executor.submit(_synthesize_chunk, (i, chunk, title))
                for i, chunk in enumerate(chapters, start=1)
            ]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                overall_bar.update(1)
            results.sort(key=lambda item: item[0])
            created = [Path(p) for _, p in results]

    overall_bar.close()

    return created
