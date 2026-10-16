#!/usr/bin/env python3
"""Prepare authentic Project Aria VRS/MP4, gaze and narration manifest.

No missing modality is fabricated. Official vrs_to_mp4 makes a preview MP4,
not a calibration-equivalent CV export. Preserve the original VRS/timestamps.
"""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess
from pathlib import Path

def prepare(vrs,mp4,gaze_csv,transcript,out,demo_id,convert=False,asr=False):
    if not (vrs or mp4):
        raise ValueError("Provide --vrs or --mp4 from an actual Aria demonstration")
    if vrs and not vrs.is_file(): raise FileNotFoundError(vrs)
    if mp4 and not mp4.is_file() and not (convert and vrs): raise FileNotFoundError(mp4)
    if gaze_csv and not gaze_csv.is_file(): raise FileNotFoundError(gaze_csv)
    if transcript and not transcript.is_file(): raise FileNotFoundError(transcript)
    if not demo_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in demo_id):
        raise ValueError("Invalid demo ID")
    out.mkdir(parents=True,exist_ok=True)
    if convert:
        if not vrs: raise ValueError("--convert requires --vrs")
        if shutil.which("vrs_to_mp4") is None:
            raise RuntimeError("Install official Project Aria vrs_to_mp4 (projectaria_tools + moviepy) or supply --mp4")
        mp4=mp4 or out / (demo_id+".mp4")
        subprocess.run(["vrs_to_mp4","--vrs",str(vrs),"--output_video",str(mp4)],check=True)
        if not mp4.is_file(): raise RuntimeError("Aria conversion produced no MP4")
    transcript_source="user_transcript" if transcript else None
    if asr:
        if not mp4 or not mp4.is_file(): raise ValueError("--asr requires an MP4")
        try: from faster_whisper import WhisperModel
        except ImportError as exc: raise RuntimeError("Install faster-whisper to use --asr") from exc
        model=WhisperModel("small",device="cpu",compute_type="int8")
        segments,_=model.transcribe(str(mp4),vad_filter=True)
        transcript=out/(demo_id+"_asr.json")
        transcript.write_text(json.dumps({"segments":[{"start_s":float(s.start),"end_s":float(s.end),"text":s.text.strip()} for s in segments]},indent=2))
        transcript_source="faster_whisper_small"
    samples=0
    if gaze_csv:
        with gaze_csv.open(newline="") as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or not any(
                "gaze" in name.lower() or name.lower() in
                {"x","y","gaze_x","gaze_y","frame","frame_idx","frame_index",
                 "timestamp_s","timestamp_ns","tracking_timestamp_us"}
                for name in reader.fieldnames
            ):
                raise ValueError(
                    "CSV does not resemble a projected gaze file. "
                    "Inspect header with transfer_aria_gaze.py inspect and "
                    "supply correct file, not an annotation CSV."
                )
            samples=sum(1 for _ in reader)
            if not samples: raise ValueError("Gaze CSV is empty")
    manifest={
        "demo_id":demo_id,
        "recording_file":str((vrs or mp4).resolve()),
        "vrs":str(vrs.resolve()) if vrs else None,
        "mp4":str(mp4.resolve()) if mp4 and mp4.is_file() else None,
        "gaze_csv":str(gaze_csv.resolve()) if gaze_csv else None,
        "gaze_metadata":{"available":bool(gaze_csv),"samples":samples},
        "transcript":str(transcript.resolve()) if transcript else None,
        "transcript_source":transcript_source,
        "eye_gaze_projection":"NOT_PROVIDED",
        "notes":["Gaze must be calibrated and projected separately, using the established Aria/DINO correspondence method.",
                 "Review MP4 is not a calibrated substitute for raw VRS timestamps."]
    }
    output=out/(demo_id+"_manifest.json")
    output.write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vrs",type=Path)
    p.add_argument("--mp4",type=Path)
    p.add_argument("--gaze-csv",type=Path)
    p.add_argument("--transcript",type=Path)
    p.add_argument("--demo-id",required=True)
    p.add_argument("--out",type=Path,default=Path("data/processed"))
    p.add_argument("--convert",action="store_true")
    p.add_argument("--asr",action="store_true")
    a=p.parse_args()
    print(json.dumps(prepare(a.vrs,a.mp4,a.gaze_csv,a.transcript,a.out,a.demo_id,
                 convert=a.convert,asr=a.asr),indent=2))

if __name__=="__main__":
    main()
