"""
CLI entry point for the upload transcription pipeline.

Usage:
    python -m scripts.run_pipeline --in path/to/audio.wav --out data/output
"""
import argparse
from pathlib import Path

from app.pipeline.run import run_pipeline, RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Notecryption transcription pipeline")
    parser.add_argument("--in", dest="input_path", required=True, help="Input audio (wav/mp3)")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = run_pipeline(args.input_path, str(out_dir), RunConfig())

    print(f"Transcription complete: {len(notes)} notes")
    for ext in ("notes.json", "out.musicxml", "out.mid"):
        p = out_dir / ext
        if p.exists():
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
