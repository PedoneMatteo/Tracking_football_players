import numpy as np
import cv2

class ViewTransformer():
    COURT_WIDTH  = 68.0
    COURT_LENGTH = 35.0
    COURT_SEGMENT = 5.84

    # Trapezio statico di fallback (il tuo originale)
    FALLBACK_PIXELS = np.array([
        [110,  1035],
        [265,   275],
        [910,   260],
        [1640,  915],
    ], dtype=np.float32)

    def __init__(self):
        self._target_vertices = np.array([
            [0,                  self.COURT_WIDTH],
            [0,                  0               ],
            [self.COURT_LENGTH,  0               ],
            [self.COURT_LENGTH,  self.COURT_WIDTH],
        ], dtype=np.float32)

        # trasformazione corrente (aggiornata per frame)
        self._current_matrix   = None
        self._current_vertices = None
        self._set_pixel_vertices(self.FALLBACK_PIXELS, 0)

    # ── API pubblica ─────────────────────────────────────────────────────────

    def set_trapezoid(self, pixel_vertices: np.ndarray | None, distance_between_extreme_lines):
        """Aggiorna il trapezio per il frame corrente."""
        if pixel_vertices is not None:
            #print(f"Trapezoid updated with vertices: {pixel_vertices}")
            #print("Trapezoid updated with new vertices.")
            self._set_pixel_vertices(pixel_vertices, distance_between_extreme_lines)
        #else:
            #print("Trapezoid is None, using fallback.") 
        # se None, mantiene l'ultimo trapezio valido

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self._current_vertices, p, False) >= 0
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
                print("frame_num:", frame_num)
                self.set_trapezoid(trap, distance_between_extreme_lines)
                #print(" ")

                for track_id, track_info in track.items():
                    position = np.array(track_info['position_adjusted'])
                    pos_transformed = self.transform_point(position)
                    if pos_transformed is not None:
                        pos_transformed = pos_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = pos_transformed

    # ── Privato ──────────────────────────────────────────────────────────────

    def _set_pixel_vertices(self, vertices: np.ndarray, distance_between_extreme_lines):
        self._current_vertices = vertices.astype(np.float32)
        if distance_between_extreme_lines!=0:
            self._target_vertices = np.array([
                [0,                  self.COURT_WIDTH],
                [0,                  0               ],
                [distance_between_extreme_lines*self.COURT_SEGMENT,  0               ],
                [distance_between_extreme_lines*self.COURT_SEGMENT,  self.COURT_WIDTH],
            ], dtype=np.float32)
            print("_target_vertices:")
            print(self._target_vertices)
            print(" ")
        self._current_matrix   = cv2.getPerspectiveTransform(
            self._current_vertices, self._target_vertices
        )