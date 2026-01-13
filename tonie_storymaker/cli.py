from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_settings
from .voice_profile.record import record_wav, trim_silence, normalize_peak
from .tts_engine.synthesize import synthesize_story_to_mp3s
from .tonies_uploader.uploader import login_and_save_state, upload_files_to_creative_tonie

def cmd_record_sample(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    tmp = out
    if args.trim or args.normalize:
        tmp = out.with_suffix(".raw.wav")

    record_wav(tmp, seconds=args.seconds, samplerate=args.samplerate)

    processed = tmp
    if args.trim:
        trimmed = out.with_suffix(".trim.wav")
        trim_silence(processed, trimmed, threshold=args.trim_threshold, pad_ms=args.trim_pad_ms)
        processed = trimmed

    if args.normalize:
        normed = out
        normalize_peak(processed, normed, target_peak=args.normalize_peak)
        processed = normed

    # cleanup intermediates
    for p in [out.with_suffix(".raw.wav"), out.with_suffix(".trim.wav")]:
        if p.exists() and p != out:
            try:
                p.unlink()
            except Exception:
                pass

    print(f"Speaker sample ready: {out}")
    return 0

def cmd_synthesize(args: argparse.Namespace) -> int:
    story_file = Path(args.story_file).expanduser().resolve()
    text = story_file.read_text(encoding="utf-8")

    created = synthesize_story_to_mp3s(
        story_text=text,
        title=args.title,
        speaker_wav=args.speaker_wav,
        language=args.language,
        out_dir=args.out_dir,
    )
    print("\nCreated MP3 files:")
    for p in created:
        print(" -", p)
    return 0

def cmd_login(args: argparse.Namespace) -> int:
    login_and_save_state()
    return 0

def cmd_upload(args: argparse.Namespace) -> int:
    uploaded = upload_files_to_creative_tonie(args.creative_name, args.files)
    print("\nUploaded tracks:")
    for name in uploaded:
        print(" -", name)
    return 0

def main() -> int:
    s = get_settings()

    ap = argparse.ArgumentParser(prog="tonie-storymaker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_rec = sub.add_parser("record-sample", help="Record a speaker reference WAV for voice cloning")
    ap_rec.add_argument("--out", required=True, help="Output wav path, e.g. speaker.wav")
    ap_rec.add_argument("--seconds", type=int, default=15)
    ap_rec.add_argument("--samplerate", type=int, default=22050)
    ap_rec.add_argument("--trim", action="store_true", help="Trim silence")
    ap_rec.add_argument("--trim-threshold", type=float, default=0.015)
    ap_rec.add_argument("--trim-pad-ms", type=int, default=150)
    ap_rec.add_argument("--normalize", action="store_true", help="Peak normalize")
    ap_rec.add_argument("--normalize-peak", type=float, default=0.9)
    ap_rec.set_defaults(func=cmd_record_sample)

    ap_syn = sub.add_parser("synthesize", help="Synthesize story text into MP3 chapters using voice cloning")
    ap_syn.add_argument("--story-file", required=True)
    ap_syn.add_argument("--title", required=True)
    ap_syn.add_argument("--speaker-wav", required=True)
    ap_syn.add_argument("--language", default=None, help="e.g. en, fr, es (default from env DEFAULT_LANGUAGE)")
    ap_syn.add_argument("--out-dir", default="output")
    ap_syn.set_defaults(func=cmd_synthesize)

    ap_log = sub.add_parser("login", help="Login to my.tonies.com manually and save a Playwright session")
    ap_log.set_defaults(func=cmd_login)

    ap_up = sub.add_parser("upload", help="Upload MP3 files to a Creative-Tonie (Playwright automation)")
    ap_up.add_argument("--creative-name", required=True, help="Visible name of your Creative-Tonie in my.tonies.com")
    ap_up.add_argument("--files", nargs="+", required=True, help="MP3 files to upload (globs expanded by your shell)")
    ap_up.set_defaults(func=cmd_upload)

    args = ap.parse_args()
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
