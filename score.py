#!/usr/bin/env python3
"""
Score creatives against Meta's TRIBE V2 brain response model.

Usage:
    python score.py video.mp4
    python score.py creatives/
    python score.py creatives/ -o results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

SUPPORTED = {
    ".mp4", ".mov", ".webm", ".avi", ".mkv",
    ".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
}


def find_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED else []
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED)
    return []


def main():
    parser = argparse.ArgumentParser(description="Score creatives with TRIBE V2")
    parser.add_argument("input", help="Video/image file or directory")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    args = parser.parse_args()

    files = find_files(args.input)
    if not files:
        print(f"No supported files found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTribe Scorer — {len(files)} creative(s)\n")

    import modal

    Predictor = modal.Cls.from_name("tribe-scorer", "Predictor")
    predictor = Predictor()

    results = []
    for i, filepath in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {filepath.name}", end="", flush=True)
        t0 = time.time()

        with open(filepath, "rb") as f:
            data = f.read()

        try:
            result = predictor.score.remote(data, filepath.name)
            elapsed = time.time() - t0
            overall = result.get("overall_score", "?")
            print(f" → {overall} ({elapsed:.0f}s)")
            results.append(result)
        except Exception as e:
            print(f" → error: {e}")

    if not results:
        print("\nNo creatives scored successfully.")
        sys.exit(1)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    else:
        print(f"\n{json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()
