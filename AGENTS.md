# AGENTS.md

## Setup

```bash
pip install -r requirements.txt
```

Download the trained model to `models/best.pt` and a sample video to `input_videos/match.mp4` (Google Drive links in `README.md`).

## Project overview

### Pipeline
1. **Video Reading** (`main.py`) → FConverts raw video streams into NumPy arrays for frame-by-frame processing.
2. **Tracker** (`tracker.py`) → Leverages YOLOv8 and ByteTrack to detect and maintain unique IDs for players, referees, and the ball.
3. **Camera Movement** (`camera_movement_estimator.py`) → Uses Optical Flow to track background motion, allowing the system to compensate for camera pans, tilts, and zooms.
4. **FieldLinesDetector** (`detect_lines.py`)Detects and stabilizes geometric field lines, distinguishing between white boundary lines and grass mowing patterns.
Output: Trapezoids (a list of 4 coordinates per frame representing the intersection of boundary and grass lines), defining the strictly visible field area.
4. **View Transformer** (`view_transformer.py`)Applies a perspective transformation matrix to map coordinates from image pixels to real-world meters.
5. **Ball Interpolation** (`tracker.py`)Uses Pandas to fill detection gaps for the ball via linear interpolation, ensuring a smooth trajectory.
6. **Speed/Distance** (`speed_and_distance_estimator.py`) Calculates player metrics, such as instantaneous speed (km/h) and total distance covered (m).
7. **Team Assigner** (`team_assigner.py`)Employs K-Means clustering on jersey colors to automatically classify players into Team 1 or Team 2.
8. **Ball Assigner** (`player_ball_assigner.py`) Measures the spherical distance between the ball and players' feet to determine real-time possession.
9. **Draw + Save** Annotates the frames with bounding boxes, traces, and statistics, then exports the final processed video.

### Struttura Dati (Tracks)

```python
tracks = {
    "players": [{frame_dict}, ...],  # Lista di frame
    "referees": [...],
    "ball": [...]
}

# Frame dict: {track_id: {bbox, position, position_adjusted, 
#                         position_transformed, speed, distance, 
#                         team, team_color, has_ball}}
```

### Funzioni Chiave

| Modulo | Funzione | Output |
|---|---|---|
| Tracker | `get_object_tracks()` | BGenerates Bounding Boxes + Track IDs per frame. |
| Tracker | `add_position_to_tracks()` | Computes the anchor point (center for ball, feet for players). |
| Camera | `get_camera_movement()` | Calculates [x, y] translation and zoom factor per frame. |
| Camera | `add_adjust_positions_to_tracks()` | `position_adjusted` added|
| FieldLinesDetector |  `_get_boundary_lines_separated` | Isolates the long horizontal pitch boundary lines.
| FieldLinesDetector |  `_get_stable_lines()` | Extracts structurally significant vertical grass lines.
| ViewTransformer | `transform_point()` | Maps a single pixel point to a meter-based coordinate. |
| ViewTransformer | `add_transformed_position_to_tracks()` | `position_transformed` added |
| Speed | `add_speed_and_distance_to_tracks()` | Add `speed`, `distance` |
| TeamAssigner | `assign_team_color()` | Add `team`, `team_color` |
| BallAssigner | `assign_ball_to_player()` | Add `has_ball` to player |

### Tech Stack

- **YOLO v8**: State-of-the-art object detection.
- **ByteTrack**: Robust multi-object tracking.
- **OpenCV**: Core library for Optical Flow, perspective warps, and Video I/O.
- **Scikit-Learn**: K-Means clustering
- **Pandas**: Data manipulation and missing value interpolation.

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
