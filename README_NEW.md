# Football Analysis — Dynamic Field & Offside Detection

## Overview

This project performs **automatic football match video analysis**: it detects and tracks players, referees, and the ball, assigns teams based on jersey colours, compensates for camera motion and zoom, computes real-world positions in metres, and — as its most advanced feature — **detects offside violations** in real time.

The core pipeline is built on top of the open-source football-analysis project by [@abhisheksakapal](https://github.com/abhisheksakapal/football-analysis). The sections below detail what the original project provided and what extensions were added, with a focus on the **dynamic field area detection** and **offside calculation** that form the novel contribution of this work.

---

## Table of Contents

1. [Original Project](#original-project)
2. [My Extensions](#my-extensions)
   - [Dynamic Field Area Detection](#1-dynamic-field-area-detection-fieldlinesdetector)
   - [Camera Zoom Compensation](#2-camera-zoom-compensation)
   - [Offside Detection](#3-offside-detection-offsidedetector)
   - [Pipeline Integration](#4-pipeline-integration)
3. [Complete Pipeline (Current State)](#complete-pipeline-current-state)
4. [Project Architecture](#project-architecture)
5. [Data Structure](#data-structure)
6. [Setup & Usage](#setup--usage)
7. [Module Reference](#module-reference)
8. [Limitations & Future Work](#limitations--future-work)

---

## Original Project

The starting point was [Abhishek Sakapal's football-analysis](https://github.com/abhisheksakapal/football-analysis) repository. Its pipeline consisted of the following modules:

| Step | Module | Technology | Output |
|------|--------|------------|--------|
| 1 | Object Detection & Tracking | YOLOv8 + ByteTrack | Bounding boxes with unique track IDs per frame |
| 2 | Camera Movement Estimation | Lucas-Kanade Optical Flow (translation only) | Per-frame `[Δx, Δy]` camera displacement |
| 3 | View Transformation | Static perspective transform (OpenCV) | Pixel positions mapped to a fixed 35m × 68m coordinate system |
| 4 | Ball Interpolation | Pandas linear interpolation | Gap-filled ball trajectory |
| 5 | Speed & Distance | Euclidean distance over frame windows | Speed (km/h) and cumulative distance (m) per player |
| 6 | Team Assignment | K-Means clustering on jersey colours | Team 1 / Team 2 classification |
| 7 | Ball Possession | Nearest-player distance threshold | Which player controls the ball |
| 8 | Annotation & Export | OpenCV drawing | Annotated output video |

### Key limitations of the original project

- **Static field area**: the perspective transform used a single hardcoded trapezoid of pixel coordinates for the entire video. This meant that any camera pan, tilt, or zoom would cause incorrect position mapping outside the initial frame.
- **No zoom detection**: camera movement was estimated as pure translation. Zooming in/out was ignored, causing inaccurate position adjustment when the camera zoomed toward the penalty area or pulled back for a wide shot.
- **No tactical analysis**: the pipeline stopped at physical metrics (speed, distance) without any rule-based analysis like offside detection.

---

## My Extensions

The goal was to enable **accurate offside detection**. This required solving two foundational problems first: 

1) knowing *where the field is* in every frame, dynamically 
2) knowing *where each player stands* in real-world metres, compensating for camera movement and perspective distortion. Only then could the offside rule be applied reliably.

### 1. Dynamic Field Area Detection (`FieldLinesDetector`)

The core challenge: the camera moves (pan, tilt, zoom) throughout a match. A fixed trapezoid for perspective transformation fails as soon as the camera shifts. The solution: **detect the visible field area in every frame independently**, then compute a frame-specific perspective transform.

#### How it works

The detector runs **two parallel pipelines** per frame, then intersects their results to form a trapezoid:

##### a) White Boundary Line Pipeline

The football pitch is delimited by white boundary lines (the two long touchlines visible in most broadcast angles).

- **HSV masking** isolates white pixels in a narrow colour range (high V, low S), with a secondary mask for brighter whites near the advertising boards.
- **Dynamic ROI mask** restricts detection to the actual field area, excluding the crowd, sky, and advertising boards. The ROI is computed per frame using the anchor-based method described below.
- **Canny edge detection + HoughLinesP** extracts horizontal line segments.
- **Far/near separation**: lines are classified by their Y coordinate. The topmost (far) line and bottommost (near) line are identified — these are the two horizontal field boundaries visible in the camera shot.
- **EMA stabilizer** (`_LineStabilizer`): the Y coordinates of both lines are smoothed across frames using exponential moving average (α = 0.35), rejecting jumps larger than 80 px as outliers.

##### b) Grass Mowing Pattern Pipeline

Broadcast cameras make the grass mowing pattern visible — alternating light/dark green vertical stripes. These stripes form a set of parallel lines converging toward a vanishing point.

- **Dual green masks**: light green (high V) and dark green (low V) are extracted separately via HSV thresholding, then combined.
- **Morphological operations**: vertical closing (kernel 3×45) connects broken vertical stripe edges; opening (7×7) removes noise.
- **DBSCAN clustering** groups line candidates by their bottom X coordinate (eps = 60 px). Clusters with 2+ lines agreeing on their top X (within 50 px) are kept — this filters out spurious detections from player movement or shadows.
- **Leftmost/rightmost extreme lines** are selected. These two lines define the lateral boundaries of the visible field portion.
- **Vanishing point** is computed via least-squares intersection of all stable grass lines.
- **Adaptive tracking**: extreme lines are tracked across frames. A line is accepted as "valid detection" if its angle and position change is small (smooth tracking), or if the change is very large (indicating a genuine camera cut or rapid pan). Otherwise, the previous line is compensated using the camera movement deltas and optionally adjusted toward the vanishing point.
- **Line numbering**: each grass stripe is numbered (leftmost visible stripe → rightmost). The distance between the two extreme lines is computed as `line_number_right − line_number_left`, which represents how many grass stripes (and thus how many metres) are visible.

##### c) Trapezoid Computation

The four trapezoid vertices are computed by intersecting the far/near horizontal boundary lines with the left/right extreme grass lines:

```
vertices = [
    far_line ∩ line_left,      # top-left
    far_line ∩ line_right,     # top-right
    near_line ∩ line_right,    # bottom-right
    near_line ∩ line_left,     # bottom-left
]
```

When horizontal lines are missing (e.g., camera too zoomed in), the trapezoid falls back to the frame edges at the appropriate Y positions.

##### d) Dynamic ROI & Anchor-Based Top Border Detection

A critical sub-problem: the advertising boards above the field create white edge noise that tricks the horizontal line detector. The solution is an **adaptive top boundary** that tracks the bottom edge of the advertising boards:

- **Search region**: the top 10%–35% of the frame.
- **White line detection**: HoughLinesP identifies candidate horizontal white segments.
- **Grass verification**: each candidate must have green grass directly below it (at least 45% of the area in a 30 px strip). This distinguishes the board/field boundary from white lines on the boards themselves.
- **Selection**: among valid candidates, the highest (smallest Y) line with length ≥ 70% of the longest candidate is chosen.
- **EMA smoothing** (α = 0.10): the anchor Y is smoothed, rejecting jumps > 25 px.
- **Camera compensation**: when detection misses (up to 8 consecutive frames tolerated), the anchor Y is shifted by the camera Y movement.

The ROI bottom boundary is also dynamic: as the camera **zooms in** (cumulative zoom < 1.0), the bottom edge progressively moves upward to exclude the increasingly blurry near-field area.

##### Tuning constants

All detection thresholds are exposed as module-level constants (lines 7–40 of `detect_lines.py`), making the system re-tunable for different stadiums, lighting conditions, and broadcast styles.

### 2. Camera Zoom Compensation

The original optical flow estimated only X/Y translation by tracking features on the far-left and far-right edges of the frame. This ignored zoom.

#### How it works

The enhanced `CameraMovementEstimator` computes **two independent optical flows** per frame:

- **Translation flow**: features extracted from vertical strips at the extreme left (20 px) and right (150 px) edges. The maximum displacement vector among tracked features is taken as the camera translation `[Δx, Δy]`.

- **Zoom flow**: features extracted from a **3×3 grid** of 15%×15% regions distributed across the frame (9 regions, padded 10% from edges). These points are spaced far apart, making them sensitive to scale changes.

- **Zoom factor calculation** (`get_zoom_factor`): for every pair of successfully tracked points, the ratio `new_distance / old_distance` is computed. The **median of all ratios** is taken as the zoom factor. A median is used instead of mean for robustness against poorly tracked outliers. Ratios are only computed for point pairs with distance > 5 px (to avoid division-by-zero noise).

- **Thresholds**: zoom changes smaller than 0.004 are treated as no-zoom. Translation movements smaller than 5 px are treated as no-movement. When neither changes significantly, feature points are **not refreshed** — this prevents feature drift when the camera is static.

- **Output**: `camera_movement_per_frame` is now a list of `[Δx, Δy, zoom_factor]` triples instead of the original `[Δx, Δy]` pairs.

### 3. Offside Detection (`OffsideDetector`)

With accurate real-world positions available (via the dynamic trapezoid → perspective transform → metre coordinates), the offside rule can now be applied per frame.

#### Football offside rule (simplified)

A player is in an **offside position** if they are:
1. In the opposing team's half of the field, AND
2. Closer to the opponent's goal line than **both** the ball and the **second-to-last opponent** (including the goalkeeper).

If the goalkeeper is not in frame, the **last outfield defender** becomes the offside reference.

#### Algorithm (per frame)

1. **Goalkeeper detection**: for each team, check if any tracked player has `is_goalkeeper=True` in the current frame (even if their transformed position is unavailable — the goalkeeper may be outside the trapezoid).

2. **Player collection**: only players with valid `position_transformed` (inside the current trapezoid) are considered. Their X coordinate in the metre-based coordinate system is extracted.

3. **Attacking team identification**: the team whose player has `has_ball=True` is the attacking team. The other team is defending. (Fallback: if no player has the ball, the previous frame's attacking team is used.)

4. **Offside line calculation**:
   - Defenders are sorted by X coordinate according to the attack direction.
   - **With goalkeeper visible**: the offside line is the X coordinate of the **penultimate defender** (index 1 after sorting). Standard FIFA rule.
   - **Without goalkeeper visible** (< 11 players in frame): the offside line is the X coordinate of the **last defender** (index 0). This handles situations where the goalkeeper is off-screen.

5. **Offside marking**: each attacking player whose X coordinate is *beyond* the offside line (greater if attacking right, less if attacking left) is flagged as `offside=True`.

6. **Attacking direction**: by default, Team 1 attacks toward the right (+X) and Team 2 toward the left (−X). A dynamic detection mode (`_detect_attacking_direction`) can auto-determine this from mean X positions in early frames (useful since K-Means team labelling is non-deterministic across runs).

#### Visualisation

The `draw_offside` method renders:
- A **thick red rectangle** (3 px) around any offside player's bounding box.
- A **filled red label** reading "OFFSIDE" above the bounding box, with white text.

### 4. Pipeline Integration

All new modules are integrated into `main.py` in the correct dependency order:

```
Video Reading
  ↓
Tracker (YOLO + ByteTrack)
  ↓
CameraMovementEstimator (translation + zoom)
  ↓
FieldLinesDetector (per-frame trapezoids)
  ↓
ViewTransformer (dynamic perspective transform using trapezoids)
  ↓
Ball Interpolation
  ↓
SpeedAndDistanceEstimator
  ↓
TeamAssigner (K-Means jersey colours)
  ↓
PlayerBallAssigner (ball possession)
  ↓
OffsideDetector (offside rule application)
  ↓
Annotation & Video Export
```

The **trapezoid dependency chain** is critical:
```
FieldLinesDetector → trapezoids → ViewTransformer → position_transformed → OffsideDetector
                                                                           → Speed/Distance
```

Both the offside detector and the speed/distance estimator depend on `position_transformed`, which only exists after the view transformer processes the trapezoids. The pipeline order in `main.py` guarantees this.

---

## Complete Pipeline (Current State)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        main.py pipeline                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1.  read_video()                    → List[np.ndarray]              │
│  2.  Tracker.get_object_tracks()     → tracks dict (bbox + track_id) │
│  3.  Tracker.add_position_to_tracks() → position (feet anchor)       │
│  4.  CameraMovementEstimator          → [Δx, Δy, zoom] per frame     │
│  5.  CameraMovementEstimator          → position_adjusted            │
│  6.  FieldLinesDetector.process_video() → trapezoids per frame       │
│  7.  ViewTransformer                  → position_transformed (metres)│
│  8.  Tracker.interpolate_ball_positions() → gap-filled ball path     │
│  9.  SpeedAndDistance_Estimator       → speed (km/h), distance (m)   │
│ 10.  TeamAssigner                     → team, team_color per player  │
│ 11.  PlayerBallAssigner               → has_ball per player          │
│ 12.  OffsideDetector                  → offside flag per player      │
│ 13.  Drawing + save_video()           → output_video.avi             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Architecture

```
Tracking_football_players/
│
├── main.py                              # Pipeline orchestrator (entry point)
├── yolo_inference.py                    # Quick YOLO-only test script
├── requirements.txt                     # Python dependencies
├── README.md                            # Original project description
├── README_NEW.md                        # This file
├── AGENTS.md                            # Technical reference for AI agents
│
├── trackers/
│   └── tracker.py                       # YOLOv8 + ByteTrack detection & tracking
│                                        #   get_object_tracks(), add_position_to_tracks(),
│                                        #   interpolate_ball_positions(), draw_annotations()
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py     # Dual optical flow (translation + zoom)
│                                        #   get_camera_movement(), get_zoom_factor(),
│                                        #   add_adjust_positions_to_tracks(), draw_camera_movement()
│
├── field_lines_detector/
│   └── detect_lines.py                  # DYNAMIC FIELD AREA (my addition)
│                                        #   Pitch boundary line detection
│                                        #   Grass mowing pattern line detection
│                                        #   Trapezoid computation
│                                        #   Dynamic ROI with anchor-based top border
│                                        #   EMA line stabilization
│
├── view_transformer/
│   └── view_transformer.py              # Perspective transformation (pixels → metres)
│                                        #   Dynamic trapezoid support, fallback static trapezoid
│
├── offside_detector/                    # OFFSIDE DETECTION (my addition)
│   └── offside_detector.py              #   add_offside_to_tracks(), _get_offside_line(),
│                                        #   _detect_attacking_direction(), draw_offside()
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Speed (km/h) & total distance (m) per player
│
├── team_assigner/
│   └── team_assigner.py                 # K-Means jersey colour clustering → Team 1 / Team 2
│
├── player_ball_assigner/
│   └── player_ball_assigner.py          # Nearest-player ball possession assignment
│
├── utils/
│   ├── bbox_utils.py                    # Bounding box geometry (centre, width, foot position)
│   ├── geometric_utils.py               # Line extrapolation, intersection, trapezoid construction
│   └── video_utils.py                   # Video I/O (read, write, frame export)
│
├── development_and_analysis/
│   └── color_assignement.ipynb          # Jupyter notebook for team colour analysis
│
├── training/
│   └── football_training_yolo_v5.ipynb  # Jupyter notebook for YOLOv5 model training
│
├── models/                              # Trained YOLO weights (gitignored)
├── stubs/                               # Pickle caches (gitignored)
├── input_videos/                        # Input match videos (gitignored)
└── output_videos/                       # Processed output (gitignored)
```

---

## Data Structure

The central data structure is the `tracks` dictionary, which flows through every module in the pipeline:

```python
tracks = {
    "players":  [frame_dict, frame_dict, ...],   # one per frame
    "referees": [frame_dict, frame_dict, ...],
    "ball":     [frame_dict, frame_dict, ...],
}
```

Each `frame_dict` maps track IDs to a dictionary of attributes:

```python
tracks["players"][frame_num][track_id] = {
    "bbox":                  [x1, y1, x2, y2],   # bounding box in pixels
    "position":              (cx, y2),            # feet anchor point (pixels)
    "position_adjusted":     (cx', y2'),          # after camera motion compensation
    "position_transformed":  [x_m, y_m] | None,   # real-world metres (None if outside field)
    "speed":                 float,               # km/h (over 5-frame window)
    "distance":              float,               # cumulative metres
    "team":                  1 | 2,               # assigned team
    "team_color":            (R, G, B),           # jersey colour
    "has_ball":              bool,                # ball possession flag
    "offside":               bool,                # offside flag (my addition)
    "is_goalkeeper":         bool,                # goalkeeper flag
}
```

### Coordinate system

The `position_transformed` field uses a **right-handed coordinate system**:
- **X axis (0–35 m)**: along the pitch length (goal-line to goal-line). X increases toward the right side of the camera view.
- **Y axis (0–68 m)**: across the pitch width (touchline to touchline). Y increases toward the bottom of the camera view.

Players whose feet are **outside** the current trapezoid receive `position_transformed = None` and are excluded from offside calculation and speed/distance estimation.

The actual X span per frame is **dynamic**: it equals `(line_number_right − line_number_left) × 5.84 m`, where `5.84 m` is the standard distance between adjacent grass mowing stripes on a football pitch. This means the perspective transform adapts to how much of the field is currently visible.

---

## Setup & Usage

### Prerequisites

- Python 3.x
- Required packages (install with `pip install -r requirements.txt`):
  - `ultralytics` — YOLOv8
  - `supervision` — ByteTrack tracker
  - `opencv-python` — image processing, optical flow, perspective transform
  - `numpy` — array operations
  - `matplotlib` — (used in notebooks)
  - `pandas` — ball position interpolation
  - `shapely` — point-in-polygon test
  - `scikit-learn` — DBSCAN clustering, K-Means

### Download models and sample video

1. Download the trained YOLO model (`best.pt`) from [Google Drive](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing) and place it in `models/best.pt`.
2. Download the [sample match video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing) and place it in `input_videos/match.mp4`.

### Running

```bash
python main.py
```

This processes the first 300 frames of `input_videos/match.mp4` through the full pipeline and saves the annotated output to `output_videos/output_video.avi`.

- **First run** (or when `stubs/` is empty): all detection results (tracking, camera movement) are computed from scratch and cached as pickle files.
- **Subsequent runs**: cached results are loaded from `stubs/`, skipping re-computation. Set `read_from_stub=False` in `main.py` to force recomputation.

### Quick YOLO test

```bash
python yolo_inference.py
```

Runs raw YOLOv8 inference on the sample video without any post-processing.

---

## Module Reference

### `FieldLinesDetector` (field_lines_detector/detect_lines.py)

The most complex module in the project (808 lines). It contains:

| Method | Purpose |
|--------|---------|
| `process_video()` | Main entry point: returns `(output_frames, trapezoids)` |
| `_process_white_lines()` | White boundary line pipeline (HSV → ROI → Canny → Hough → stabilizer) |
| `_process_grass_lines()` | Grass mowing pattern pipeline (green masks → DBSCAN → extreme lines) |
| `_get_boundary_lines_separated()` | Classifies Hough lines into far/near by Y position |
| `_get_stable_lines()` | DBSCAN clustering of grass line candidates |
| `_adjust_extreme_grass_lines()` | Temporal tracking of left/right extreme lines with camera compensation |
| `_detect_top_border_y()` | Anchor-based advertising board bottom edge detection |
| `_get_field_roi_mask_strict()` | Dynamic ROI polygon using anchor + zoom-progressive bottom |
| `_LineStabilizer` (inner class) | EMA stabilizer for horizontal white lines |

**Key design decisions**:
- The white line and grass line pipelines run independently, then **intersect** to form the trapezoid. This makes the system robust: if white lines are occluded (players, shadows), the grass lines alone can define the field boundaries.
- The line numbering system (tracking which numbered grass stripe is visible) enables **sub-field precision**: knowing there are 6 visible stripes means 6 × 5.84 = 35 m of field width is visible, not the full 68 m.
- The anchor-based top border detection handles the most common failure mode — advertising boards being mistaken for field lines.

### `OffsideDetector` (offside_detector/offside_detector.py)

| Method | Purpose |
|--------|---------|
| `add_offside_to_tracks()` | Main per-frame offside logic |
| `_get_offside_line()` | Computes the offside line X coordinate from defender positions |
| `_detect_attacking_direction()` | Auto-detects which team attacks which direction |
| `draw_offside()` | Renders red border + "OFFSIDE" label on flagged players |

**Key design decisions**:
- The offside line uses the **penultimate defender** rule (not just the last defender), as per official FIFA Laws of the Game.
- Goalkeeper presence is checked **independently of position validity** — even if the GK is outside the trapezoid (e.g., rushing out), their presence still affects the offside line.
- Team ball possession from the `PlayerBallAssigner` is reused rather than recomputed, maintaining consistency.
- A player who is offside is flagged in the tracks dictionary, which any downstream consumer can query.

### `CameraMovementEstimator` (camera_movement_estimator/camera_movement_estimator.py)

Enhancements over the original:

| Feature | Original | Enhanced |
|---------|----------|----------|
| Translation | Lucas-Kanade on edge features | Same, unchanged |
| Zoom | Not detected | Lucas-Kanade on 3×3 grid features, median pairwise distance ratio |
| Feature refresh | On every frame | Only when movement exceeds thresholds (5 px translation, 0.004 zoom ratio) |
| Output format | `[Δx, Δy]` | `[Δx, Δy, zoom]` |

### `ViewTransformer` (view_transformer/view_transformer.py)

Adapted to accept per-frame trapezoids:

| Feature | Original | Enhanced |
|---------|----------|----------|
| Input | Static hardcoded trapezoid | Per-frame trapezoid from `FieldLinesDetector` |
| Target X span | Fixed `COURT_LENGTH = 35.0 m` | Dynamic: `distance_between_extreme_lines × 5.84 m` |
| Point validation | None | Shapely polygon containment test (points outside trapezoid → `None`) |

The static `FALLBACK_PIXELS` trapezoid is used when field line detection fails (e.g., a single frame with too much noise).

### `SpeedAndDistance_Estimator` (speed_and_distance_estimator/speed_and_distance_estimator.py)

Uses a **5-frame sliding window** (at 24 fps = 0.21 s) to compute instantaneous speed. The speed value is written to all frames in the window, giving smooth speed updates even if intermediate positions are missing.

### `TeamAssigner` (team_assigner/team_assigner.py)

- First frame: extracts jersey colours from all non-goalkeeper players.
- Runs 2-cluster K-Means on the collected RGB colours → Team 1 centre (R₁, G₁, B₁) and Team 2 centre (R₂, G₂, B₂).
- Subsequent frames: each player's jersey colour is classified by nearest cluster centre.
- Goalkeeper assignments are cached by ID for consistency.

### `PlayerBallAssigner` (player_ball_assigner/player_ball_assigner.py)

Computes the minimum distance from the ball centre to either the left or right bottom corner of each player's bounding box. Returns the ID of the player within 70 px of the ball, or `-1` if no player qualifies.

---

## Limitations & Future Work

### Current limitations

1. **Hardcoded goalkeeper IDs**: These are valid only for the sample match video and must be re-tuned for other matches.
2. **Static attacking direction**: the offside detector currently uses a hardcoded assumption that Team 1 attacks right. The dynamic direction detection method exists but is not used.
3. **Single-camera broadcast angle**: the system is designed exclusively for the standard TV broadcast camera — the elevated sideline view typically used in professional football telecasts. This specific angle is assumed throughout the pipeline, so footage from any other perspective (behind-goal, tactical cam, drone, etc.) would not work correctly without significant rework.
4. **No ball-offside check**: the current offside is a semplified version. The logic only checks player positions relative to the second-to-last defender. The ball position is not compared (a player level with or behind the ball cannot be offside, even if beyond the last defender).
5. **No half-way line awareness**: a player in their own half cannot be offside, but this is not currently checked.
6. **No error handling**: the pipeline has no try/except blocks. A single frame failure in any module propagates to a crash.

### Potential improvements

- **Ball-offside integration**: add the ball's `position_transformed` to the offside check. A player behind the ball is never offside.
- **Half-way line detection**: detect the centre line from the field lines module to enforce the "own half" exception.
- **Automated goalkeeper detection**: instead of hardcoded IDs, detect the goalkeeper by distinctive jersey colour (usually different from outfield players) or position (closest to own goal).
- **Multi-match configurability**: expose tuning constants (HSV ranges, Hough thresholds, DBSCAN eps) as a per-match configuration file.
- **Active offside rule**: distinguish between "offside position" and "offside offence". The latter requires the player to be involved in active play (interfering with play, interfering with an opponent, gaining an advantage).
- **Referees**: referees are currently tracked but not used. Their positions could assist in detecting set-piece situations where offside rules differ (e.g., goal kicks, throw-ins, corner kicks).
- **GPU acceleration**: the grass line DBSCAN clustering and per-frame HSV operations could be batched for GPU processing.
- **End-to-end offside visualisation**: draw the offside line itself on the output video for debugging and demonstration.

---

## Credits

- **Original football-analysis project**: [Abhishek Sakapal](https://github.com/abhisheksakapal/football-analysis)
- **Dynamic field detection & offside module**: implemented on top of the original pipeline
- **YOLOv8**: Ultralytics
- **ByteTrack**: `supervision` library

---

## License

This project inherits the license of the original [football-analysis](https://github.com/abhisheksakapal/football-analysis) repository.
