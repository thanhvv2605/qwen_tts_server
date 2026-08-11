"""Register every {name}.wav + {name}.txt pair in a directory as a voice.

Usage:
    python scripts/register_voices.py /path/to/voice_dir [base_url]
"""

import sys
from pathlib import Path

import httpx


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    voice_dir = Path(sys.argv[1])
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    pairs = sorted(voice_dir.glob("*.wav"))
    if not pairs:
        print(f"no .wav files found in {voice_dir}")
        sys.exit(1)

    ok = failed = 0
    for wav_path in pairs:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"SKIP {wav_path.name}: no matching .txt")
            failed += 1
            continue
        name = wav_path.stem.lower()
        resp = httpx.post(
            f"{base_url}/v1/voices",
            data={"name": name, "ref_text": txt_path.read_text(encoding="utf-8")},
            files={"ref_audio": (wav_path.name, wav_path.read_bytes(), "audio/wav")},
            timeout=60,
        )
        if resp.status_code == 201:
            print(f"OK   {name} ({resp.json()['duration_s']}s)")
            ok += 1
        elif resp.status_code == 409:
            print(f"SKIP {name}: already registered")
            ok += 1
        else:
            print(f"FAIL {name}: {resp.status_code} {resp.text}")
            failed += 1

    print(f"\n{ok} registered/existing, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
