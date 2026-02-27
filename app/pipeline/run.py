# app/pipeline/run.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.audio.preprocess import ensure_wav_mono
from app.postprocess.cleanup import cleanup_notes, quantize_notes
from app.transcription.basic_pitch_poly import transcribe_with_basic_pitch
from app.transcription.guitar_mapper import map_notes_to_guitar
from app.transcription.articulation import annotate_articulations

# Optional (only if you added the exporter file I gave you)
try:
    from app.export.musicxml_export import export_notes_to_musicxml, MusicXmlConfig
except Exception:
    export_notes_to_musicxml = None
    MusicXmlConfig = None


@dataclass
class RunConfig:
    # Transcription
    target_sr: int = 22050

    # Cleanup / filtering
    min_dur_s: float = 0.12
    dedupe_onset_tol_s: float = 0.04
    pitch_range: Tuple[int, int] = (40, 88)  # guitar-ish

    # Quantization (optional but helps readability)
    enable_quantize: bool = True
    quantize_bpm: int = 112
    quantize_grid: str = "1/16"

    # Guitar mapping
    max_fret: int = 24

    # Articulation annotation (upload only)
    enable_articulation: bool = True

    # MusicXML export
    export_musicxml: bool = True
    musicxml_title: str = "Transcription"
    musicxml_bpm: int = 112
    musicxml_key_fifths: int = 2  # D major-ish
    musicxml_filename: str = "out.musicxml"


def _load_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    # safety: remove NaN/inf
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return audio, int(sr)


def run_pipeline(audio_path: str, out_dir: Optional[str] = None, cfg: Optional[RunConfig] = None) -> List[Dict]:
    """
    End-to-end upload pipeline:
      input (mp3/wav) -> wav mono -> BasicPitch -> cleanup -> quantize -> map to guitar
      -> articulation annotate (optional) -> MusicXML export (optional)

    Returns: list[dict] items:
      {
        "pitch": int,
        "onset_s": float,
        "dur_s": float,
        "string": int,
        "fret": int,
        "tech": optional dict
      }
    """
    cfg = cfg or RunConfig()

    # Prepare output directory
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    # Convert to wav mono at target_sr
    wav_path = ensure_wav_mono(audio_path, target_sr=cfg.target_sr)
    _wav_is_temp = (wav_path != audio_path)

    # Load mono audio (for articulation contour analysis)
    audio, sr = _load_mono_float32(wav_path)

    # 1) Polyphonic transcription
    raw_notes = transcribe_with_basic_pitch(wav_path)  # expected list of NoteEvent-like objects

    # Clean up temp wav now that both consumers have read it
    if _wav_is_temp:
        try:
            import os as _os
            _os.remove(wav_path)
        except Exception:
            pass

    # 2) Cleanup
    raw_notes = cleanup_notes(
        raw_notes,
        min_dur=cfg.min_dur_s,
        dedupe_onset_tol=cfg.dedupe_onset_tol_s,
        pitch_range=cfg.pitch_range,
    )

    # 3) Quantize (for readability)
    if cfg.enable_quantize:
        raw_notes = quantize_notes(raw_notes, bpm=cfg.quantize_bpm, grid=cfg.quantize_grid)

    # 4) Map to guitar string/fret
    mapped = map_notes_to_guitar(raw_notes, max_fret=cfg.max_fret)

    out: List[Dict] = []
    for n in mapped:
        # mapped object should have: pitch, onset, duration, string, fret
        if getattr(n, "string", None) is None or getattr(n, "fret", None) is None:
            continue
        out.append(
            {
                "pitch": int(n.pitch),
                "onset_s": float(n.onset),
                "dur_s": float(n.duration),
                "string": int(n.string),
                "fret": int(n.fret),
            }
        )

    out.sort(key=lambda r: (float(r["onset_s"]), int(r["string"]), int(r["pitch"])))

    # 5) Articulation annotation (bend/slide/hammer/pull) for uploads
    if cfg.enable_articulation and out:
        out = annotate_articulations(audio=audio, sr=sr, notes=out)

    # 6) MusicXML export (so it can look like your screenshot)
    if (
        cfg.export_musicxml
        and out_path is not None
        and export_notes_to_musicxml is not None
        and MusicXmlConfig is not None
        and out
    ):
        xml_file = out_path / cfg.musicxml_filename
        export_notes_to_musicxml(
            out,
            str(xml_file),
            MusicXmlConfig(
                title=cfg.musicxml_title,
                bpm=cfg.musicxml_bpm,
                key_fifths=cfg.musicxml_key_fifths,
                time_sig=(4, 4),
                divisions=960,
                instrument_name="Guitar",
            ),
        )

    # 7) MIDI export
    if out_path is not None and out:
        try:
            from app.export.midi_export import export_midi_from_dicts
            export_midi_from_dicts(out, str(out_path / "out.mid"), bpm=cfg.musicxml_bpm)
        except Exception:
            pass

    # 8) JSON export
    if out_path is not None and out:
        try:
            import json as _json
            (out_path / "notes.json").write_text(
                _json.dumps(out, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    return out


# Optional: quick CLI run (you can ignore if you already use scripts.run_pipeline)
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input audio (mp3/wav)")
    ap.add_argument("--out", dest="out", required=True, help="Output directory")
    args = ap.parse_args()

    notes = run_pipeline(args.inp, args.out, RunConfig())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print(f"OK: {len(notes)} notes -> {out_dir/'notes.json'}")
    if (out_dir / "out.musicxml").exists():
        print(f"MusicXML: {out_dir/'out.musicxml'}") 