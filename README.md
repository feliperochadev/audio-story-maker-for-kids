# Tonie Storymaker (Linux)

A small Python app that:
- records a **voice sample** (speaker reference WAV)
- **voice-clones** using Chatterbox TTS (Turbo or multilingual)
- generates **chaptered audio** from pasted/story text
- exports **MP3** (Tonies-compatible) and optionally automates upload with **Playwright**

> Notes
> - This project uses Chatterbox TTS models. Ensure your usage complies with the model license.
> - Website automation can be fragile; UI changes may break selectors.

## Quick start

### 1) System deps (Linux)
- `ffmpeg`
- PortAudio dev libs (for `sounddevice`)
  - Debian/Ubuntu: `sudo apt-get install -y ffmpeg portaudio19-dev`
  - Fedora: `sudo dnf install -y ffmpeg portaudio-devel`

### 2) Install Python deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

For Chatterbox Turbo, this repo installs from GitHub. You also need a Hugging Face token to download model weights.

```bash
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

### 3) Configure environment
Copy `.env.example` to `.env` and edit:
```bash
cp .env.example .env
```

### 4) Record a voice sample
```bash
tonie-storymaker record-sample --out speaker.wav --seconds 15
```

### 5) Create MP3 chapters from a story text file
```bash
tonie-storymaker synthesize \
  --story-file story.txt \
  --title "Bedtime Story" \
  --speaker-wav speaker.wav \
  --language en \
  --out-dir output
```

### Tone control
Set a global tone in `.env`:
```
TTS_TONE=toddler
```
Override per run:
```bash
tonie-storymaker synthesize --tone gentle ...
```
Override per segment in text:
```
[tone: calm] Once upon a time...
[tone: dramatic] Suddenly, a wolf appeared.
```

### 6) (Optional) Save login session for my.tonies.com
Opens a browser; log in manually, then press Enter in terminal.
```bash
tonie-storymaker login
```

### 7) (Optional) Upload all generated chapters if track names don't already exist
```bash
tonie-storymaker upload \
  --creative-name "My Creative Tonie" \
  --files output/*.mp3
```

## Track naming & duplicates
Uploads use the filename (without extension) as the track name by default.
If a track name already exists, the uploader can either:
- skip (default)
- or append a suffix like " (2)" (set `UPLOAD_ON_DUPLICATE=rename`)

See `.env.example` for config.
