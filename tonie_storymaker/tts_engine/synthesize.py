from __future__ import annotations

import gc
import inspect
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import torch
import torchaudio as ta
from tqdm import tqdm

from .audio_utils import wav_to_mp3
from .text_utils import split_into_chapters, split_for_tts, clean_text
from ..config import get_settings

def _log_gpu_stats(label: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(
            f"{label} | cuda mem free={free // (1024 ** 2)}MB "
            f"total={total // (1024 ** 2)}MB allocated={allocated // (1024 ** 2)}MB "
            f"reserved={reserved // (1024 ** 2)}MB"
        )
    except Exception:
        pass

_TONE_TAG_PATTERN = re.compile(r"\[tone\s*:\s*([^\]]+)\]", re.IGNORECASE)

_TONE_PRESETS = {
    "neutral": {"cfg_weight": 0.5, "exaggeration": 0.5},
    "calm": {"cfg_weight": 0.4, "exaggeration": 0.4},
    "gentle": {"cfg_weight": 0.35, "exaggeration": 0.6},
    "toddler": {"cfg_weight": 0.3, "exaggeration": 0.7},
    "storybook": {"cfg_weight": 0.3, "exaggeration": 0.7},
    "dramatic": {"cfg_weight": 0.3, "exaggeration": 0.8},
}


def _merge_preface_with_first_chapter(chapters: List[str], chapter_keywords: List[str]) -> List[str]:
    if len(chapters) < 2:
        return chapters
    keywords = [kw.strip() for kw in chapter_keywords if kw.strip()]
    if not keywords:
        return chapters
    pattern = re.compile(rf"(?i)^({'|'.join(re.escape(k) for k in keywords)})\b")
    first = chapters[0].lstrip()
    second = chapters[1].lstrip()
    if pattern.match(first):
        return chapters
    if not pattern.match(second):
        return chapters
    merged = f"{chapters[0].strip()} ... {chapters[1].strip()}"
    return [merged] + chapters[2:]


def _load_chatterbox_model(variant: str, device: str):
    if variant == "turbo":
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except Exception as exc:
            raise RuntimeError(
                "Chatterbox Turbo is not available in this installation. "
                "Install from the GitHub repo to get chatterbox.tts_turbo."
            ) from exc
        return ChatterboxTurboTTS.from_pretrained(device=device), "turbo"
    if variant == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(device=device), "multilingual"
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=device), "english"

def _gpu_meets_vram_requirement(min_vram_gb: float) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        return total_gb >= min_vram_gb
    except Exception:
        return False


from contextlib import contextmanager


@contextmanager
def _force_cpu():
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_is_available = torch.cuda.is_available
    original_device_count = torch.cuda.device_count
    original_torch_load = torch.load

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _false():
        return False

    def _zero():
        return 0

    torch.cuda.is_available = _false
    torch.cuda.device_count = _zero

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = _cpu_load

    try:
        yield
    finally:
        if original_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
        torch.cuda.is_available = original_is_available
        torch.cuda.device_count = original_device_count
        torch.load = original_torch_load

def _resolve_tone_settings(tone: str, settings) -> dict:
    tone_key = (tone or "").strip().lower()
    values = _TONE_PRESETS.get(tone_key, _TONE_PRESETS["neutral"]).copy()
    if settings.tts_tone_cfg_weight is not None:
        values["cfg_weight"] = settings.tts_tone_cfg_weight
    if settings.tts_tone_exaggeration is not None:
        values["exaggeration"] = settings.tts_tone_exaggeration
    return values


def _parse_tone_segments(text: str, default_tone: str) -> List[tuple[str, str]]:
    segments: List[tuple[str, str]] = []
    current_tone = default_tone
    cursor = 0
    for match in _TONE_TAG_PATTERN.finditer(text):
        before = text[cursor:match.start()].strip()
        if before:
            segments.append((current_tone, before))
        current_tone = match.group(1).strip().lower()
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        segments.append((current_tone, tail))
    return segments


def _generate_audio(model, text: str, audio_prompt_path: str | None, language_id: str | None, tone_settings: dict):
    kwargs = {}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path
    kwargs.update(tone_settings)

    signature = inspect.signature(model.generate)
    if "language_id" in signature.parameters:
        if language_id is None and signature.parameters["language_id"].default is inspect._empty:
            raise ValueError("language_id is required for this model; pass --language or set DEFAULT_LANGUAGE.")
        if language_id is not None:
            kwargs["language_id"] = language_id
    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return model.generate(text, **filtered)


def synthesize_story_to_mp3s(
    story_text: str,
    title: str,
    speaker_wav: str | Path,
    language: Optional[str] = None,
    out_dir: str | Path = "output",
    tone: Optional[str] = None,
) -> List[Path]:
    """
    Voice-clone a story into multiple MP3 chapter files.
    Returns list of created MP3 paths.
    """
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_TOKEN", hf_token)
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
    chapters = _merge_preface_with_first_chapter(chapters, s.chapter_split_keywords)
    if not chapters:
        raise ValueError("Story text is empty after cleaning.")

    speaker_wav = Path(speaker_wav).expanduser().resolve()
    if not speaker_wav.exists():
        raise FileNotFoundError(f"Speaker wav not found: {speaker_wav}")
    def is_oom_error(exc: Exception) -> bool:
        return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def run_synthesis(model, max_chars: int, variant: str) -> List[Path]:
        created: List[Path] = []
        total = len(chapters)
        overall_bar = tqdm(total=total, desc="Chapters", unit="chapter")
        tone_default = (tone or s.tts_tone).strip().lower()
        use_language_id = language if variant in {"multilingual", "turbo"} else None

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

            try:
                segments = _parse_tone_segments(chapter_text, tone_default)
                combined_text = " ".join(segment for _, segment in segments).strip()
                combined_text = clean_text(combined_text)
                if not combined_text:
                    raise ValueError("Chapter text is empty after cleaning.")
                print(f"chunk text (chapter {i:02d}): {combined_text}")
                segment_settings = _resolve_tone_settings(tone_default, s)
                with torch.inference_mode():
                    try:
                        _log_gpu_stats("before generate")
                        wav = _generate_audio(
                            model,
                            text=combined_text,
                            audio_prompt_path=str(speaker_wav),
                            language_id=use_language_id,
                            tone_settings=segment_settings,
                        )
                    except Exception as exc:
                        chunk_len = len(combined_text)
                        ref_size = speaker_wav.stat().st_size if speaker_wav.exists() else 0
                        print(
                            "Chatterbox generate failed. "
                            f"chunk_len={chunk_len} max_chars={max_chars} "
                            f"ref_bytes={ref_size} tone={tone_default} variant={variant}"
                        )
                        _log_gpu_stats("after failure")
                        print(traceback.format_exc())
                        raise exc
                    if wav.dim() > 1:
                        wav = wav.squeeze(0)
                    wav = wav.detach().cpu().float()
                    model_device = getattr(model, "device", None)
                    if model_device is not None and str(model_device).startswith("cuda"):
                        torch.cuda.empty_cache()
                        gc.collect()
            except Exception as exc:
                stop_event.set()
                progress_thread.join(timeout=1)
                chapter_bar.close()
                overall_bar.close()
                raise exc

            chapter_wav = out_dir / f"{title} - Chapter {i:02d}.wav"
            ta.save(str(chapter_wav), wav.unsqueeze(0), model.sr)

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

        overall_bar.close()
        return created

    if s.tts_use_gpu and _gpu_meets_vram_requirement(s.tts_min_gpu_vram_gb):
        device = "cuda"
    else:
        device = "cpu"
        if s.tts_use_gpu:
            print(
                f"GPU VRAM below {s.tts_min_gpu_vram_gb}GB; forcing CPU. "
                "Increase TTS_MIN_GPU_VRAM_GB or set TTS_USE_GPU=false to silence this."
            )
    if s.tts_chatterbox_variant == "turbo":
        attempts = [
            ("turbo", device, s.tts_max_chars),
            ("turbo", "cpu", min(s.tts_max_chars, 160)),
        ]
    else:
        attempts = [
            (s.tts_chatterbox_variant, device, s.tts_max_chars),
            (s.tts_chatterbox_variant, "cpu", min(s.tts_max_chars, 160)),
        ]

    seen = set()
    for variant, attempt_device, max_chars in attempts:
        key = (variant, attempt_device, max_chars)
        if key in seen:
            continue
        seen.add(key)
        model = None
        try:
            if attempt_device == "cuda":
                print(f"Loading model (variant={variant}, device=cuda, max_chars={max_chars})")
                model, variant_used = _load_chatterbox_model(variant, device=attempt_device)
                print(f"TTS device: {attempt_device} (variant: {variant_used})")
                return run_synthesis(model, max_chars=max_chars, variant=variant_used)

            with _force_cpu():
                print(f"Loading model (variant={variant}, device=cpu, max_chars={max_chars})")
                model, variant_used = _load_chatterbox_model(variant, device="cpu")
                print(f"TTS device: cpu (variant: {variant_used})")
                return run_synthesis(model, max_chars=max_chars, variant=variant_used)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            if attempt_device == "cuda":
                torch.cuda.empty_cache()
            if model is not None:
                del model
            gc.collect()
            print(f"CUDA OOM detected. Retrying with lower settings (variant={variant}, device={attempt_device}).")

    raise RuntimeError("Chatterbox failed due to CUDA out of memory even after fallback attempts.")
