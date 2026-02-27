from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import xml.etree.ElementTree as ET


@dataclass
class MusicXmlConfig:
    title: str = "Transcription"
    bpm: int = 112
    time_sig: Tuple[int, int] = (4, 4)  # (beats, beat_type)
    key_fifths: int = 0  # 0=C, 2=D, -1=F etc.
    divisions: int = 960  # ticks per quarter note (high resolution)
    instrument_name: str = "Guitar"
    add_chord_symbols: bool = False  # keep false unless you actually compute chords


# ---------- Pitch helpers ----------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_step_alter_oct(midi: int) -> Tuple[str, int, int]:
    pc = midi % 12
    octv = midi // 12 - 1
    name = _NOTE_NAMES[pc]
    if len(name) == 1:
        return name, 0, octv
    return name[0], 1, octv  # only sharps; good enough for MVP


def seconds_to_ticks(sec: float, bpm: int, divisions: int) -> int:
    # quarter note duration = 60/bpm seconds
    q = 60.0 / float(bpm)
    return int(round((sec / q) * divisions))


def _ticks_to_note_type(dur_ticks: int, divisions: int) -> str:
    """Map tick duration to MusicXML note type name."""
    quarter = divisions
    ratios = [
        (4.0, "whole"),
        (3.0, "half"),       # dotted half
        (2.0, "half"),
        (1.5, "quarter"),    # dotted quarter
        (1.0, "quarter"),
        (0.75, "eighth"),    # dotted eighth
        (0.5, "eighth"),
        (0.25, "16th"),
        (0.125, "32nd"),
    ]
    ratio = dur_ticks / float(quarter) if quarter > 0 else 1.0
    best = "quarter"
    best_diff = 999.0
    for r, name in ratios:
        diff = abs(ratio - r)
        if diff < best_diff:
            best_diff = diff
            best = name
    return best


# ---------- MusicXML builders ----------

def _sub(parent: ET.Element, tag: str, text: Optional[str] = None, **attrib) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def _note_el(
    *,
    pitch_midi: int,
    dur_ticks: int,
    voice: int,
    staff: int,
    string: int,
    fret: int,
    tech: Optional[Dict] = None,
    tie_start: bool = False,
    tie_stop: bool = False,
) -> ET.Element:
    n = ET.Element("note")

    # pitch
    pitch = _sub(n, "pitch")
    step, alter, octv = midi_to_step_alter_oct(int(pitch_midi))
    _sub(pitch, "step", step)
    if alter != 0:
        _sub(pitch, "alter", str(alter))
    _sub(pitch, "octave", str(octv))

    # duration
    _sub(n, "duration", str(int(dur_ticks)))
    _sub(n, "voice", str(int(voice)))
    _sub(n, "type", _ticks_to_note_type(dur_ticks, 960))
    _sub(n, "staff", str(int(staff)))

    if tie_start:
        _sub(n, "tie", type="start")
    if tie_stop:
        _sub(n, "tie", type="stop")

    notations = None
    if tech or tie_start or tie_stop:
        notations = _sub(n, "notations")

    if tie_start:
        tied = _sub(notations, "tied", type="start")
    if tie_stop:
        tied = _sub(notations, "tied", type="stop")

    # TAB info is stored under <technical> in <notations>
    technical = None
    if tech is not None or True:
        if notations is None:
            notations = _sub(n, "notations")
        technical = _sub(notations, "technical")
        _sub(technical, "string", str(int(string)))
        _sub(technical, "fret", str(int(fret)))

    # Techniques (MVP):
    # bend: {"type":"bend","semitones":1.0}
    # slide: {"type":"slide","to": ...}
    # hammer/pull: {"type":"hammer"/"pull"}
    if tech and technical is not None:
        t = tech.get("type")

        if t == "bend":
            bend = _sub(technical, "bend")
            # MusicXML bend-alter is in semitones (can be float).
            semis = float(tech.get("semitones", 1.0))
            _sub(bend, "bend-alter", f"{semis:.3f}".rstrip("0").rstrip("."))

        elif t == "slide":
            # slide is a notation element, not "technical" in some renderers,
            # but MuseScore accepts it in technical too. We also add <slide>.
            if notations is None:
                notations = _sub(n, "notations")
            _sub(notations, "slide", type="start")

        elif t == "hammer":
            if notations is None:
                notations = _sub(n, "notations")
            _sub(notations, "hammer-on", type="start")

        elif t == "pull":
            if notations is None:
                notations = _sub(n, "notations")
            _sub(notations, "pull-off", type="start")

    return n


def export_notes_to_musicxml(
    notes: List[Dict],
    out_path: str,
    cfg: MusicXmlConfig = MusicXmlConfig(),
) -> None:
    """
    Writes a 2-staff MusicXML Part (standard + TAB).
    notes: list of dicts with keys pitch,onset_s,dur_s,string,fret, optional tech
    """
    notes = [dict(n) for n in notes]
    notes.sort(key=lambda r: (float(r["onset_s"]), int(r["pitch"])))

    score = ET.Element("score-partwise", version="3.1")

    # work title
    work = _sub(score, "work")
    _sub(work, "work-title", cfg.title)

    # part list
    part_list = _sub(score, "part-list")
    score_part = _sub(part_list, "score-part", id="P1")
    _sub(score_part, "part-name", cfg.instrument_name)

    # part
    part = _sub(score, "part", id="P1")

    beats, beat_type = cfg.time_sig
    ticks_per_measure = int(cfg.divisions * beats * (4 / beat_type))

    # group notes into measures by onset ticks
    def measure_index(on_ticks: int) -> int:
        return int(on_ticks // ticks_per_measure) + 1

    # build measure dict
    measures: Dict[int, List[Dict]] = {}
    for n in notes:
        on = seconds_to_ticks(float(n["onset_s"]), cfg.bpm, cfg.divisions)
        dur = max(1, seconds_to_ticks(float(n["dur_s"]), cfg.bpm, cfg.divisions))
        n["_on"] = on
        n["_dur"] = dur
        mi = measure_index(on)
        measures.setdefault(mi, []).append(n)

    max_m = max(measures.keys()) if measures else 1

    # emit measures
    for mi in range(1, max_m + 1):
        meas = _sub(part, "measure", number=str(mi))

        # attributes only in first measure
        if mi == 1:
            attr = _sub(meas, "attributes")
            _sub(attr, "divisions", str(cfg.divisions))

            key = _sub(attr, "key")
            _sub(key, "fifths", str(cfg.key_fifths))

            ts = _sub(attr, "time")
            _sub(ts, "beats", str(beats))
            _sub(ts, "beat-type", str(beat_type))

            # two staves: 1 standard, 2 TAB
            _sub(attr, "staves", "2")

            # staff 1 clef (treble)
            clef1 = _sub(attr, "clef", number="1")
            _sub(clef1, "sign", "G")
            _sub(clef1, "line", "2")

            # staff 2 clef (TAB)
            clef2 = _sub(attr, "clef", number="2")
            _sub(clef2, "sign", "TAB")
            _sub(clef2, "line", "5")

            sd = _sub(attr, "staff-details", number="2")
            _sub(sd, "staff-lines", "6")

            # tempo
            direction = _sub(meas, "direction", placement="above")
            dir_type = _sub(direction, "direction-type")
            met = _sub(dir_type, "metronome")
            _sub(met, "beat-unit", "quarter")
            _sub(met, "per-minute", str(cfg.bpm))
            _sub(direction, "sound", tempo=str(cfg.bpm))

        # measure contents timeline (simple: insert notes + rests)
        items = measures.get(mi, [])
        items.sort(key=lambda r: (int(r["_on"]), int(r["pitch"])))

        cursor = (mi - 1) * ticks_per_measure

        def add_rest(dur_ticks: int, staff: int):
            if dur_ticks <= 0:
                return
            rn = _sub(meas, "note")
            _sub(rn, "rest")
            _sub(rn, "duration", str(int(dur_ticks)))
            _sub(rn, "voice", "1")
            _sub(rn, "type", _ticks_to_note_type(dur_ticks, cfg.divisions))
            _sub(rn, "staff", str(int(staff)))

        for n in items:
            on = int(n["_on"])
            dur = int(n["_dur"])
            if on > cursor:
                # add rests in BOTH staves to keep alignment
                add_rest(on - cursor, staff=1)
                add_rest(on - cursor, staff=2)
                cursor = on

            # Add same note twice: staff 1 + staff 2 (TAB)
            tech = n.get("tech")
            pitch = int(n["pitch"])
            string = int(n["string"])
            fret = int(n["fret"])

            meas.append(_note_el(
                pitch_midi=pitch,
                dur_ticks=dur,
                voice=1,
                staff=1,
                string=string,
                fret=fret,
                tech=tech,
            ))
            meas.append(_note_el(
                pitch_midi=pitch,
                dur_ticks=dur,
                voice=1,
                staff=2,
                string=string,
                fret=fret,
                tech=tech,
            ))

            cursor = on + dur

        # fill end-of-measure rests
        end = mi * ticks_per_measure
        if cursor < end:
            add_rest(end - cursor, staff=1)
            add_rest(end - cursor, staff=2)

        if mi == max_m:
            bar = _sub(meas, "barline", location="right")
            _sub(bar, "bar-style", "light-heavy")

    tree = ET.ElementTree(score)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)