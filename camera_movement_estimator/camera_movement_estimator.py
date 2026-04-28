import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance
from itertools import combinations


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.minimum_zoom_change = 0.004

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        h, w = frame.shape[:2]
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Maschera originale per la TRASLAZIONE (bordi, evita i giocatori)
        mask_translation = np.zeros_like(first_frame_grayscale)
        mask_translation[:, 0:20] = 1
        mask_translation[:, w-150:w] = 1

        # Maschera per lo ZOOM: griglia 3x3 distribuita sul frame
        # I punti devono essere lontani tra loro per rilevare cambi di scala
        mask_zoom = np.zeros_like(first_frame_grayscale)
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        region_w, region_h = int(w * 0.15), int(h * 0.15)
        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                cx = pad_x + col * (w - 2 * pad_x) // 2
                cy = pad_y + row * (h - 2 * pad_y) // 2
                x1, y1 = cx - region_w // 2, cy - region_h // 2
                mask_zoom[y1:y1+region_h, x1:x1+region_w] = 1

        self.features_translation = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_translation
        )

        self.features_zoom = dict(
            maxCorners=50,
            qualityLevel=0.3,
            minDistance=15,   # punti più distanti = rapporti di scala più stabili
            blockSize=7,
            mask=mask_zoom
        )

        # Tieni anche self.features per compatibilità
        self.features = self.features_translation
        def get_zoom_factor(self, old_points, new_points, status):
            """
            Stima il fattore di zoom confrontando le distanze reciproche
            tra feature points prima e dopo il frame.

            zoom > 1.0  →  la camera si avvicina (zoom in)
            zoom < 1.0  →  la camera si allontana (zoom out)
            zoom = 1.0  →  nessuno zoom

            Usa la mediana dei rapporti per robustezza agli outlier.
            Richiede almeno 2 punti validamente tracciati.
            """
            # Filtra solo i punti con tracking valido
            valid_mask = status.ravel() == 1
            old_pts = old_points[valid_mask].reshape(-1, 2)
            new_pts = new_points[valid_mask].reshape(-1, 2)

            if len(old_pts) < 2:
                return 1.0  # impossibile stimare, nessuno zoom

            ratios = []
            # Calcola il rapporto distanza_nuova / distanza_vecchia per ogni coppia
            for (i, j) in combinations(range(len(old_pts)), 2):
                old_dist = np.linalg.norm(old_pts[i] - old_pts[j])
                new_dist = np.linalg.norm(new_pts[i] - new_pts[j])

                if old_dist > 5.0:  # evita divisioni per distanze troppo piccole
                    ratios.append(new_dist / old_dist)

            if not ratios:
                return 1.0

            # La mediana è molto più robusta della media contro i punti mal tracciati
            zoom = float(np.median(ratios))

            # Se la variazione è trascurabile, trattala come assente
            if abs(zoom - 1.0) < self.minimum_zoom_change:
                zoom = 1.0

            return zoom

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Corregge le posizioni dei tracciati rimuovendo sia la traslazione
        della camera sia lo zoom accumulato fino a quel frame.
        """
        # Precalcola lo zoom cumulativo per ogni frame
        cumulative_zoom = 1.0
        zoom_per_frame = []
        for movement_data in camera_movement_per_frame:
            cumulative_zoom *= movement_data[2]  # indice 2 = zoom del frame
            zoom_per_frame.append(cumulative_zoom)

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0, 1.0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features_t = cv2.goodFeaturesToTrack(old_gray, **self.features_translation)
        old_features_z = cv2.goodFeaturesToTrack(old_gray, **self.features_zoom)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # --- Traslazione (feature sui bordi) ---
            new_features_t, status_t, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features_t, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            if new_features_t is not None and status_t is not None:
                for i, (new, old) in enumerate(zip(new_features_t, old_features_t)):
                    if status_t[i] == 0:
                        continue
                    new_pt = new.ravel()
                    old_pt = old.ravel()
                    distance = measure_distance(new_pt, old_pt)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(old_pt, new_pt)

            # --- Zoom (feature distribuite sul frame) ---
            zoom_factor = 1.0
            if old_features_z is not None:
                new_features_z, status_z, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features_z, None, **self.lk_params
                )
                if new_features_z is not None and status_z is not None:
                    zoom_factor = self.get_zoom_factor(old_features_z, new_features_z, status_z)

            if max_distance > self.minimum_distance or zoom_factor != 1.0:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y, zoom_factor]
                old_features_t = cv2.goodFeaturesToTrack(frame_gray, **self.features_translation)
                old_features_z = cv2.goodFeaturesToTrack(frame_gray, **self.features_zoom)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def get_zoom_factor(self, old_points, new_points, status):
        valid_mask = status.ravel() == 1
        old_pts = old_points[valid_mask].reshape(-1, 2)
        new_pts = new_points[valid_mask].reshape(-1, 2)
    
        if len(old_pts) < 2:
            return 1.0
    
        ratios = []
        for (i, j) in combinations(range(len(old_pts)), 2):
            old_dist = np.linalg.norm(old_pts[i] - old_pts[j])
            new_dist = np.linalg.norm(new_pts[i] - new_pts[j])
            if old_dist > 5.0:
                ratios.append(new_dist / old_dist)
    
        if not ratios:
            return 1.0
    
        zoom = float(np.median(ratios))
        if abs(zoom - 1.0) < self.minimum_zoom_change:
            zoom = 1.0
        return zoom

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        cumulative_zoom = 1.0

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 120), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement, zoom = camera_movement_per_frame[frame_num]
            cumulative_zoom *= zoom

            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Zoom: {zoom:.3f}x  (cum: {cumulative_zoom:.3f}x)",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Frame: {frame_num}",
                                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames