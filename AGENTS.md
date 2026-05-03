# AGENTS.md

## Setup

```bash
pip install -r requirements.txt
```

Download the trained model to `models/best.pt` and a sample video to `input_videos/match.mp4` (Google Drive links in `README.md`).

## Project overview

Pipeline: **object detection → tracking → camera compensation → field line detection → perspective transform → speed/distance → team assignment → ball possession → annotated output video**.

- Entry point: `main.py`
- Top-level utility scripts: `yolo_inference.py` (standalone YOLO test)
- Notebooks: `training/football_training_yolo_v5.ipynb` (training), `development_and_analysis/color_assignement.ipynb` (color clustering analysis).

## Commands

```bash
python main.py          # full pipeline
python yolo_inference.py # raw YOLO inference on match.mp4
```

No lint, typecheck, or test commands exist. There is no CI.

## Architecture notes

### Stub/cache pattern
Tracking and camera movement results are cached as pickle files in `stubs/` (gitignored). Set `read_from_stub=False` in `main.py` to force recomputation. First run creates the stubs; subsequent runs skip detection.

### Fixed ball track ID
The ball always uses track ID `1` (hardcoded in `trackers/tracker.py:98`). Code that accesses ball tracks assumes `tracks["ball"][frame][1]`.

### Goalkeeper class merging
YOLO class "goalkeeper" is merged into "player" during tracking (`trackers/tracker.py:72-73`).

### View transformer coordinate system
The perspective transform maps pixel positions to a 35m × 68m court coordinate system. A static fallback trapezoid is used when field line detection fails (`view_transformer/view_transformer.py:11-16`).

### Speed/distance depends on perspective transform
Speed/distance estimation reads `position_transformed` from tracks, which is only set after the view transformer runs. The pipeline order in `main.py` matters: view transformer must run before speed/distance estimator.
