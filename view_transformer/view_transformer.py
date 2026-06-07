import numpy as np
import cv2
from shapely.geometry import Polygon, Point

class ViewTransformer():
    COURT_WIDTH  = 68.0
    COURT_LENGTH = 35.0
    COURT_SEGMENT = 5.84

    # Trapezio statico di fallback 
    FALLBACK_PIXELS = np.array([
        [110,  1035],
        [265,   275],
        [910,   260],
        [1640,  915],
    ], dtype=np.float32)

    def __init__(self):
        self._target_vertices = np.array([
            [0,                  0               ],
            [self.COURT_LENGTH,  0               ],
            [self.COURT_LENGTH,  self.COURT_WIDTH],
            [0,                  self.COURT_WIDTH],
        ], dtype=np.float32)

        # trasformazione corrente (aggiornata per frame)
        self._current_matrix   = None
        self._current_vertices = None
        self._set_pixel_vertices(self.FALLBACK_PIXELS, 0)

    # ── API pubblica ─────────────────────────────────────────────────────────

    def set_trapezoid(self, pixel_vertices: np.ndarray | None, distance_between_extreme_lines):
        """Aggiorna il trapezio per il frame corrente."""
        if pixel_vertices is not None:
            self._set_pixel_vertices(pixel_vertices, distance_between_extreme_lines)

    def transform_point(self, point, tid, frame):
        p = (int(point[0]), int(point[1]))
        
        is_inside = self._current_polygon.contains(Point(p))
        
        if not is_inside:
            return None
        reshaped = point.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self._current_matrix)
        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks, trapezoids):
        """
        trapezoids: list[np.ndarray | None], uno per frame.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # aggiorna il trapezio per questo frame
                trap, distance_between_extreme_lines = trapezoids[frame_num] if frame_num < len(trapezoids) else None
                self.set_trapezoid(trap, distance_between_extreme_lines)

                for track_id, track_info in track.items():
                    position = np.array(track_info['position_adjusted'])
                    pos_transformed = self.transform_point(position, track_id, frame_num) #track_id e frame_num aggiunti solo per DEBUG
                    if pos_transformed is not None:
                        pos_transformed = pos_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = pos_transformed

    # ── Privato ──────────────────────────────────────────────────────────────

    def order_vertices_ccw(self, vertices):
        """Ordina i vertici in senso counter-clockwise."""
        centroid = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - centroid[1], 
                            vertices[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        return vertices[sorted_indices]

    def _set_pixel_vertices(self, vertices: np.ndarray, distance_between_extreme_lines):
        #self._current_vertices = vertices.astype(np.float32)
        self._current_vertices = self.order_vertices_ccw(vertices.astype(np.float32))
        self._current_polygon = Polygon(self._current_vertices)
        if distance_between_extreme_lines!=0:
            self._target_vertices = np.array([
                [0,                  0],                                                  # TL
                [distance_between_extreme_lines*self.COURT_SEGMENT,   0],                 # TR
                [distance_between_extreme_lines*self.COURT_SEGMENT,   self.COURT_WIDTH],  # BR
                [0,                  self.COURT_WIDTH],                                   # BL
            ], dtype=np.float32)
        self._current_matrix   = cv2.getPerspectiveTransform(
            self._current_vertices, self._target_vertices
        )
        
    