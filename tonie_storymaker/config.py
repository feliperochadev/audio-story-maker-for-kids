import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

@dataclass(frozen=True)
class Settings:
    app_data_dir: Path

    # export
    mp3_bitrate: str
    chapter_max_chars: int
    output_gain_db: float
    silence_min_len_ms: int
    silence_keep_ms: int
    silence_thresh_db: float
    silence_fade_ms: int

    # tts
    tts_model: str
    default_language: str
    tts_use_gpu: bool
    tts_strip_punctuation: bool
    tts_max_chars: int
    tts_workers: int
    tts_gpu_queue_share: float
    tts_gpu_min_chars: int
    tts_batch_max_chars: int
    chapter_split_keywords: List[str]

    # tonies / playwright
    tonies_base_url: str
    playwright_storage_state: str
    playwright_headless: bool

    upload_on_duplicate: str
    duplicate_suffix_format: str

    # selectors
    sel_nav_creative_tonies_text: str
    sel_edit_content_text: str
    sel_upload_audio_text: str
    sel_track_title_locator: str
    sel_file_input: str

def get_settings() -> Settings:
    app_data_dir = Path(os.getenv("APP_DATA_DIR", ".tonie_storymaker")).expanduser().resolve()
    app_data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_data_dir=app_data_dir,

        mp3_bitrate=os.getenv("MP3_BITRATE", "192k"),
        chapter_max_chars=int(os.getenv("CHAPTER_MAX_CHARS", "2500")),
        output_gain_db=float(os.getenv("OUTPUT_GAIN_DB", "-3.0")),
        silence_min_len_ms=int(os.getenv("SILENCE_MIN_LEN_MS", "600")),
        silence_keep_ms=int(os.getenv("SILENCE_KEEP_MS", "150")),
        silence_thresh_db=float(os.getenv("SILENCE_THRESH_DB", "-40.0")),
        silence_fade_ms=int(os.getenv("SILENCE_FADE_MS", "40")),

        tts_model=os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
        default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        tts_use_gpu=_get_bool("TTS_USE_GPU", False),
        tts_strip_punctuation=_get_bool("TTS_STRIP_PUNCTUATION", True),
        tts_max_chars=int(os.getenv("TTS_MAX_CHARS", "200")),
        tts_workers=int(os.getenv("TTS_WORKERS", "1")),
        tts_gpu_queue_share=float(os.getenv("TTS_GPU_QUEUE_SHARE", "0.5")),
        tts_gpu_min_chars=int(os.getenv("TTS_GPU_MIN_CHARS", "160")),
        tts_batch_max_chars=int(os.getenv("TTS_BATCH_MAX_CHARS", "400")),
        chapter_split_keywords=[
            kw.strip()
            for kw in os.getenv(
                "CHAPTER_SPLIT_KEYWORDS",
                "chapter,capitulo,capítulo,chapitre,kapitel,capitolo,hoofdstuk,глава,章,章节",
            ).split(",")
            if kw.strip()
        ],

        tonies_base_url=os.getenv("TONIES_BASE_URL", "https://my.tonies.com/"),
        playwright_storage_state=os.getenv("PLAYWRIGHT_STORAGE_STATE", "auth_state.json"),
        playwright_headless=_get_bool("PLAYWRIGHT_HEADLESS", False),

        upload_on_duplicate=os.getenv("UPLOAD_ON_DUPLICATE", "skip").strip().lower(),
        duplicate_suffix_format=os.getenv("DUPLICATE_SUFFIX_FORMAT", " ({n})"),

        sel_nav_creative_tonies_text=os.getenv("SEL_NAV_CREATIVE_TONIES_TEXT", "Creative Tonies"),
        sel_edit_content_text=os.getenv("SEL_EDIT_CONTENT_TEXT", "Edit Content"),
        sel_upload_audio_text=os.getenv("SEL_UPLOAD_AUDIO_TEXT", "Upload Audio"),
        sel_track_title_locator=os.getenv("SEL_TRACK_TITLE_LOCATOR", 'css=[data-testid="track-title"]'),
        sel_file_input=os.getenv("SEL_FILE_INPUT", 'css=input[type="file"]'),
    )
