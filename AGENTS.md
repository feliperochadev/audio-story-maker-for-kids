# Repository Guidelines

## Project Structure & Module Organization
- `tonie_storymaker/` holds the Python package and CLI entrypoint (`tonie_storymaker/cli.py`).
- `tonie_storymaker/tts_engine/` contains text/audio utilities and synthesis logic.
- `tonie_storymaker/voice_profile/` handles voice sample recording.
- `tonie_storymaker/tonies_uploader/` encapsulates the Playwright-based upload flow.
- Root files: `README.md` for usage, `pyproject.toml` for packaging, `requirements.txt` for deps.

## Build, Test, and Development Commands
- `python -m venv .venv` and `source .venv/bin/activate`: create/activate a local venv.
- `pip install -r requirements.txt`: install runtime dependencies.
- `playwright install`: install browser binaries for uploads.
- `tonie-storymaker record-sample --out speaker.wav --seconds 15`: capture a reference voice sample.
- `tonie-storymaker synthesize --story-file story.txt --title "Bedtime Story" --speaker-wav speaker.wav --language en --out-dir output`: generate chaptered MP3s.
- `tonie-storymaker login` / `tonie-storymaker upload --creative-name "My Creative Tonie" --files output/*.mp3`: optional web upload flow.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation; keep modules small and focused by feature area.
- Prefer descriptive, snake_case function and variable names; modules follow package naming.
- No formatter or linter is configured; keep changes consistent with existing style.

## Testing Guidelines
- No test framework or `tests/` directory is present yet.
- If adding tests, place them under `tests/` and mirror module names (e.g., `tests/test_text_utils.py`).
- Add a short note in `README.md` describing how to run new tests.

## Architecture Overview
- CLI entrypoint in `tonie_storymaker/cli.py` orchestrates recording, synthesis, login, and upload commands.
- Text and audio processing live in `tonie_storymaker/tts_engine/`, which converts story files into chaptered MP3s.
- Upload automation is isolated to `tonie_storymaker/tonies_uploader/` to keep Playwright concerns separate.

## Commit & Pull Request Guidelines
- No Git history is available in this workspace, so follow the repo owner’s conventions if provided.
- Otherwise, use short, imperative commit subjects (e.g., "Add uploader retry logic").
- PRs should summarize behavior changes, include CLI examples, and note any new dependencies.

## Configuration Tips
- Some commands require system dependencies like `ffmpeg` and PortAudio dev libraries.
- Playwright automation may break on UI changes; keep selectors scoped and document updates.
