from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from playwright.sync_api import sync_playwright, Page

from ..config import get_settings

def _ensure_state_path() -> Path:
    s = get_settings()
    state_path = s.app_data_dir / s.playwright_storage_state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return state_path

def login_and_save_state() -> Path:
    """
    Opens Tonies site in a visible browser; you log in manually (email/password + any MFA),
    then press Enter in terminal. Saves storage state for future automated runs.
    """
    s = get_settings()
    state_path = _ensure_state_path()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(s.tonies_base_url)
        print("\nLog in manually in the opened browser window.")
        print("When you are fully logged in and can see your account, return here and press Enter.")
        input()
        ctx.storage_state(path=str(state_path))
        browser.close()

    print(f"Saved Playwright session to: {state_path}")
    return state_path

def _get_track_names(page: Page) -> Set[str]:
    s = get_settings()
    # Best-guess locator; update SEL_TRACK_TITLE_LOCATOR if needed.
    loc = page.locator(s.sel_track_title_locator)
    names = set()
    count = loc.count()
    for i in range(count):
        t = loc.nth(i).inner_text().strip()
        if t:
            names.add(t)
    return names

def _navigate_to_creative(page: Page, creative_name: str):
    s = get_settings()
    # Go to Creative Tonies section and click the desired item by visible text.
    page.get_by_text(s.sel_nav_creative_tonies_text, exact=False).click()
    page.get_by_text(creative_name, exact=False).click()
    page.get_by_text(s.sel_edit_content_text, exact=False).click()

def _upload_one(page: Page, mp3_path: Path, track_name: str) -> None:
    s = get_settings()
    page.get_by_text(s.sel_upload_audio_text, exact=False).click()
    page.set_input_files(s.sel_file_input, str(mp3_path))

    # Many sites show an inline editor after upload; we don't assume structure.
    # If you want auto-rename, you'll likely need to adjust selectors for the title field.

def upload_files_to_creative_tonie(
    creative_name: str,
    files: Iterable[str | Path],
) -> List[str]:
    """
    Upload mp3 files to the selected Creative-Tonie.
    Duplicate handling is controlled by env:
      UPLOAD_ON_DUPLICATE=skip | rename
    Track name defaults to filename stem.
    Returns list of uploaded track names.
    """
    s = get_settings()
    state_path = _ensure_state_path()
    if not state_path.exists():
        raise FileNotFoundError(
            f"Playwright session not found at {state_path}. Run: tonie-storymaker login"
        )

    file_paths = [Path(f).expanduser().resolve() for f in files]
    for fp in file_paths:
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        if fp.suffix.lower() != ".mp3":
            raise ValueError(f"Only .mp3 supported by this uploader right now: {fp}")

    uploaded: List[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=s.playwright_headless)
        ctx = browser.new_context(storage_state=str(state_path))
        page = ctx.new_page()
        page.goto(s.tonies_base_url)

        _navigate_to_creative(page, creative_name)

        existing = _get_track_names(page)

        for fp in file_paths:
            base_name = fp.stem.strip()
            track_name = base_name

            if track_name in existing:
                if s.upload_on_duplicate == "skip":
                    print(f"Skipping duplicate track name: {track_name}")
                    continue
                elif s.upload_on_duplicate == "rename":
                    n = 2
                    while True:
                        candidate = track_name + s.duplicate_suffix_format.format(n=n)
                        if candidate not in existing:
                            track_name = candidate
                            break
                        n += 1
                    print(f"Renaming duplicate '{base_name}' -> '{track_name}'")
                else:
                    raise ValueError("UPLOAD_ON_DUPLICATE must be 'skip' or 'rename'")

            print(f"Uploading: {fp.name}")
            _upload_one(page, fp, track_name)

            # We *assume* the site creates a track and shows it; add to existing so duplicates are handled
            existing.add(track_name)
            uploaded.append(track_name)

        browser.close()

    return uploaded
