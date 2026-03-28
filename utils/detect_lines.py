import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# ── COSTANTI DI TUNING ──────────────────────────────────────────────────────
WHITE_V_MIN       = 160
WHITE_S_MAX       = 60
ROI_TOP_FRAC      = 0.24
ROI_BOTTOM_FRAC   = 0.89
HOUGH_THRESHOLD   = 60
HOUGH_MIN_LEN     = 200
HOUGH_MAX_GAP     = 25
ANGLE_TOLERANCE   = 20
FAR_LINE_MAX_Y    = 0.42
NEAR_LINE_MIN_Y   = 0.58
EMA_ALPHA         = 0.35
MAX_JUMP_PX       = 80
SENSITIVITY_TOP   = 8.5
SENSITIVITY_BOTTOM= 8.0
ADJUSTED_THRESHOLD= 35
# ────────────────────────────────────────────────────────────────────────────
# ── COSTANTI ANCHOR — versione finale semplificata ───────────────────────────
ANCHOR_SEARCH_TOP      = 0.10
ANCHOR_SEARCH_BOTTOM   = 0.35
ANCHOR_MIN_LINE_LEN    = 250
ANCHOR_MAX_ANGLE       = 5
ANCHOR_EMA_ALPHA       = 0.10
ANCHOR_MAX_JUMP        = 25
ANCHOR_MARGIN_ABOVE    = 0.01
ANCHOR_FALLBACK_FRAC   = 0.22
ANCHOR_GRASS_CHECK_PX  = 30    # striscia più ampia sotto → più affidabile
ANCHOR_GRASS_H_MIN     = 35
ANCHOR_GRASS_H_MAX     = 85
ANCHOR_GRASS_S_MIN     = 40
ANCHOR_GRASS_RATIO     = 0.45  # soglia più alta: deve essere chiaramente erba
ANCHOR_MAX_MISS        = 8

class FieldLinesDetector:
    """
    Rileva e disegna le linee del campo da calcio (fasce d'erba e bordi bianchi)
    su una sequenza di frame video.

    Uso:
        detector = FieldLinesDetector(first_frame)
        output_frames = detector.draw_field_lines_on_video(video_frames, camera_movement_per_frame, type=1)
    """

    def __init__(self, first_frame):
        # ── Stato pipeline ERBA (type=0) ────────────────────────────────────
        self.line_left        = []
        self.line_right       = []
        self.adjusted_left    = 0
        self.adjusted_right   = 0
        self.line_prev        = None
        self.angle_left_prev  = None
        self.top_x_prev       = None
        self.bottom_x_prev    = None
        self.v_flag           = 0

        # ── Stato pipeline LINEE BIANCHE (type=1) ───────────────────────────
        self._stabilizer = _LineStabilizer()

        # ── ROI dinamica ─────────────────────────────────────────────────────
        self.roi_top_curr    = ROI_TOP_FRAC
        self.roi_bottom_curr = ROI_BOTTOM_FRAC
        
        # ── Stato anchor bordo superiore ────────────────────────────────────────────
        self._anchor_y   = None   # y EMA-smoothed
        self._anchor_raw = None   # ultima y grezza accettata
        
        self.zoom_cum = 1.0

    # ════════════════════════════════════════════════════════════════════════
    #  ENTRY POINT
    # ════════════════════════════════════════════════════════════════════════

    def draw_field_lines_on_video(self, video_frames, camera_movement_per_frame, type):
        """
        Processa tutti i frame e ritorna la lista di frame annotati.
        type=0 → pipeline fasce d'erba
        type=1 → pipeline linee bianche di bordo campo
        """
        output_frames = []

        for frame_idx, frame in enumerate(video_frames):
            height, width = frame.shape[:2]
            output_frame  = frame.copy()
            preprocessed  = self._preprocess_image(frame)

            if type:
                output_frame = self._process_white_lines(
                    output_frame, preprocessed, height, width,
                    camera_movement_per_frame[frame_idx]
                )
            else:
                output_frame = self._process_grass_lines(
                    output_frame, preprocessed, height, width,
                    camera_movement_per_frame[frame_idx]
                )

            output_frames.append(output_frame)

        return output_frames

    # ════════════════════════════════════════════════════════════════════════
    #  PIPELINE LINEE BIANCHE  (type=1)
    # ════════════════════════════════════════════════════════════════════════

    def _process_white_lines(self, output_frame, preprocessed, height, width, cam_movement):
        zoom_factor = cam_movement[2]

        mask_white, _ = self._get_white_lines_mask(preprocessed)
        roi           = self._get_field_roi_mask_strict(preprocessed, height, width, zoom_factor, cam_movement)
        mask_roi      = cv2.bitwise_and(mask_white, roi)
        edges         = self._get_clean_edges(mask_roi)

        far_raw, near_raw     = self._get_boundary_lines_separated(edges, height, width)
        far_line, near_line   = self._stabilizer.update(far_raw, near_raw)

        lines_to_draw = [l for l in [far_line, near_line] if l is not None]
        return self._draw_field_lines(output_frame, lines_to_draw, roi_mask=roi)

    # ════════════════════════════════════════════════════════════════════════
    #  PIPELINE ERBA  (type=0)
    # ════════════════════════════════════════════════════════════════════════

    def _process_grass_lines(self, output_frame, preprocessed, height, width, cam_movement):
        grass_l, grass_d = self._get_grass_masks(preprocessed)
        field_roi        = self._get_field_roi_mask(grass_l, grass_d)

        edges_light    = self._get_clean_edges(grass_l)
        edges_dark     = self._get_clean_edges(grass_d)
        combined       = cv2.bitwise_or(edges_light, edges_dark)
        edges_combined = cv2.bitwise_and(combined, field_roi)

        grass_lines     = self._get_stable_lines(edges_combined, height, width)
        vanishing_point = self._compute_vanishing_point(grass_lines)

        line_left_curr, line_right_curr = self._get_extreme_lines(grass_lines, height, width)
        if line_left_curr is None or line_right_curr is None:
            return output_frame

        extreme_grass_lines = self._adjust_extreme_grass_lines(
            line_left_curr, line_right_curr, height, cam_movement, vanishing_point
        )
        self.v_flag = 0

        output_frame = self._draw_grass_lines(output_frame, grass_lines)
        output_frame = self._draw_extreme_grass_lines(output_frame, extreme_grass_lines)
        return output_frame

    # ════════════════════════════════════════════════════════════════════════
    #  PREPROCESSING
    # ════════════════════════════════════════════════════════════════════════

    def _preprocess_image(self, image):
        """CLAHE sul canale L per bilanciare luci/ombre."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_balanced = clahe.apply(l)
        lab_final = cv2.merge((l_balanced, a, b))
        return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    # ════════════════════════════════════════════════════════════════════════
    #  MASCHERA COLORE
    # ════════════════════════════════════════════════════════════════════════

    def _remove_noise_by_area(self, mask, min_area=1500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        return mask

    def _get_grass_masks(self, balanced_image):
        hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)

        lower_light = np.array([35, 90, 145])
        upper_light = np.array([55, 255, 255])
        mask_light  = cv2.inRange(hsv, lower_light, upper_light)

        lower_dark = np.array([35, 50, 20])
        upper_dark = np.array([55, 255, 100])
        mask_dark  = cv2.inRange(hsv, lower_dark, upper_dark)

        kernel_v    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 45))
        kernel_open = np.ones((7, 7), np.uint8)

        for m in [mask_light, mask_dark]:
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_v,    dst=m)
            cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel_open, dst=m)

        mask_light = self._remove_noise_by_area(mask_light, min_area=2000)
        mask_dark  = self._remove_noise_by_area(mask_dark,  min_area=2000)
        return mask_light, mask_dark

    def _get_white_lines_mask(self, balanced_image):
        hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv,
                            np.array([0,   0,   WHITE_V_MIN]),
                            np.array([180, WHITE_S_MAX, 255]))
        mask2 = cv2.inRange(hsv,
                            np.array([15, 0,  200]),
                            np.array([40, 40, 255]))
        mask_white = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        return mask_white, mask_white

    # ════════════════════════════════════════════════════════════════════════
    #  ROI
    # ════════════════════════════════════════════════════════════════════════
    
    def _get_field_roi_mask_strict(self, preprocessed, height, width, zoom_factor, cam_movement):
        # ── Aggiorna zoom_cum prima di tutto ─────────────────────────────────
        self.zoom_cum *= zoom_factor

        # ── Anchor superiore ─────────────────────────────────────────────────
        raw_y = self._detect_top_border_y(preprocessed, height, width)

        if raw_y is not None:
            jump_ok = (self._anchor_y is None
                       or abs(raw_y - self._anchor_y) < ANCHOR_MAX_JUMP)
            if jump_ok:
                self._anchor_raw  = raw_y
                self._anchor_miss = 0
            else:
                self._anchor_miss += 1
        else:
            self._anchor_miss += 1

        if self._anchor_miss <= ANCHOR_MAX_MISS and self._anchor_raw is not None:
            if self._anchor_y is None:
                self._anchor_y = self._anchor_raw
            else:
                self._anchor_y = ((1 - ANCHOR_EMA_ALPHA) * self._anchor_y
                                  + ANCHOR_EMA_ALPHA * self._anchor_raw)
        elif self._anchor_miss > 0 and self._anchor_y is not None:
            # detector missa: trasla l'anchor con il movimento camera
            dy = cam_movement[1]
            self._anchor_y = self._anchor_y - dy

        if self._anchor_y is None:
            top_frac = ANCHOR_FALLBACK_FRAC
        else:
            top_frac = (self._anchor_y / height) - ANCHOR_MARGIN_ABOVE

        top_frac = float(np.clip(top_frac, 0.05, 0.45))

        # ── Bordo inferiore (invariato) ───────────────────────────────────────
        zoom_progress = float(np.clip((1.0 - self.zoom_cum) / (1.0 - 0.726), 0.0, 1.0))
        target_bottom = float(np.clip(ROI_BOTTOM_FRAC - zoom_progress * 0.12, 0.0, 1.0))
        alpha = 0.055 * (0.6 + 0.4 * zoom_progress)
        self.roi_bottom_curr = ((1 - alpha) * self.roi_bottom_curr
                                + alpha * target_bottom)

        # ── Maschera ─────────────────────────────────────────────────────────
        roi = np.zeros((height, width), dtype=np.uint8)
        points = np.array([
            [int(width * 0.01), int(height * top_frac)],
            [int(width * 0.99), int(height * top_frac)],
            [int(width * 0.99), int(height * self.roi_bottom_curr)],
            [int(width * 0.01), int(height * self.roi_bottom_curr)],
        ], np.int32)
        cv2.fillPoly(roi, [points], 255)
        return roi
    
    def _detect_top_border_y(self, preprocessed, height, width):
        """
        Cerca il bordo inferiore dei tabelloni pubblicitari.
        Unico vincolo: sotto la linea ci deve essere chiaramente erba verde.
        Il colore sopra non viene controllato (tabelloni variabili).
        Tra tutti i candidati che passano il filtro erba, prende quello
        con la y più alta (più vicino al bordo campo reale) e più lungo.
        """
        y_top = int(height * ANCHOR_SEARCH_TOP)
        y_bot = int(height * ANCHOR_SEARCH_BOTTOM)
        strip = preprocessed[y_top:y_bot, :]

        hsv   = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv,
                            np.array([0,   0,   WHITE_V_MIN]),
                            np.array([180, WHITE_S_MAX, 255]))
        mask2 = cv2.inRange(hsv,
                            np.array([15,  0,   200]),
                            np.array([40,  40,  255]))
        mask  = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        edges  = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 0), 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=40,
            minLineLength=ANCHOR_MIN_LINE_LEN,
            maxLineGap=20
        )
        if lines is None:
            return None

        hsv_full    = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        grass_lower = np.array([ANCHOR_GRASS_H_MIN, ANCHOR_GRASS_S_MIN, 30])
        grass_upper = np.array([ANCHOR_GRASS_H_MAX, 255, 255])
        candidates  = []

        for line in [l[0] for l in lines]:
            x1, y1, x2, y2 = line
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ANCHOR_MAX_ANGLE < angle < (180 - ANCHOR_MAX_ANGLE):
                continue

            length   = np.hypot(x2 - x1, y2 - y1)
            y_center = (y1 + y2) / 2.0 + y_top

            xa = max(int(min(x1, x2)), 0)
            xb = min(int(max(x1, x2)), width - 1)
            if xa >= xb:
                continue

            # ── Controllo erba sotto ──────────────────────────────────────────
            y_below_start = int(y_center) + 5
            y_below_end   = min(y_below_start + ANCHOR_GRASS_CHECK_PX, height - 1)
            if y_below_start >= height:
                continue

            below = hsv_full[y_below_start:y_below_end, xa:xb]
            if below.size == 0:
                continue

            grass_mask  = cv2.inRange(below, grass_lower, grass_upper)
            grass_ratio = np.count_nonzero(grass_mask) / grass_mask.size
            if grass_ratio < ANCHOR_GRASS_RATIO:
                continue

            candidates.append((length, y_center))

        if not candidates:
            return None

        # Tra i candidati validi prendi il più in alto (y minima)
        # con lunghezza almeno il 70% del candidato più lungo —
        # evita di prendere un segmento corto casuale molto in alto
        max_len    = max(c[0] for c in candidates)
        min_len_th = max_len * 0.70
        filtered   = [c for c in candidates if c[0] >= min_len_th]
        filtered.sort(key=lambda c: c[1])   # ordina per y crescente (più in alto prima)
        return filtered[0][1]

    def _get_field_roi_mask(self, mask_light, mask_dark):
        """ROI per la pipeline erba (basata sulle maschere di colore)."""
        combined = cv2.bitwise_or(mask_light, mask_dark)
        kernel   = np.ones((50, 50), np.uint8)
        return cv2.dilate(combined, kernel, iterations=1)

    # ════════════════════════════════════════════════════════════════════════
    #  RILEVAMENTO BORDI
    # ════════════════════════════════════════════════════════════════════════

    def _get_clean_edges(self, mask):
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    # ════════════════════════════════════════════════════════════════════════
    #  RILEVAMENTO LINEE
    # ════════════════════════════════════════════════════════════════════════

    def _get_boundary_lines_separated(self, edges, height, width):
        hough = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LEN,
            maxLineGap=HOUGH_MAX_GAP
        )
        if hough is None:
            return None, None

        far_candidates  = []
        near_candidates = []

        for line in [l[0] for l in hough]:
            x1, y1, x2, y2 = line
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not (angle < ANGLE_TOLERANCE or angle > 180 - ANGLE_TOLERANCE):
                continue

            length   = np.hypot(x2 - x1, y2 - y1)
            y_center = (y1 + y2) / 2.0

            if y_center < height * FAR_LINE_MAX_Y:
                far_candidates.append((length, line))
            elif y_center > height * NEAR_LINE_MIN_Y:
                near_candidates.append((length, line))

        best_far  = max(far_candidates,  key=lambda x: x[0])[1] if far_candidates  else None
        best_near = max(near_candidates, key=lambda x: x[0])[1] if near_candidates else None
        return best_far, best_near

    def _get_stable_lines(self, edges, height, width):
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                       threshold=50, minLineLength=170, maxLineGap=30)
        if hough_lines is None:
            return []

        lines_data = []
        for line in [l[0] for l in hough_lines]:
            x1, y1, x2, y2 = line
            angle    = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            bottom_x = self._extrapolate_line_point(line, height - 1)
            top_x    = self._extrapolate_line_point(line, 0)

            if bottom_x is not None and 25 < angle < 85 and 0 <= top_x <= width:
                lines_data.append({'bottom_x': bottom_x, 'top_x': top_x, 'line': line})

        if not lines_data:
            return []

        X      = np.array([ld['bottom_x'] for ld in lines_data]).reshape(-1, 1)
        db     = DBSCAN(eps=60, min_samples=1).fit(X)
        labels = db.labels_

        grouped = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            grouped.setdefault(label, []).append(lines_data[i])

        final_lines = []
        for group in grouped.values():
            top_coords  = [ld['top_x'] for ld in group]
            median_top  = np.median(top_coords)
            valid_lines = [ld['line'] for ld in group if abs(ld['top_x'] - median_top) < 50]
            if len(valid_lines) >= 2:
                final_lines.extend(valid_lines)

        return final_lines

    # ════════════════════════════════════════════════════════════════════════
    #  LINEE ESTREME & VANISHING POINT
    # ════════════════════════════════════════════════════════════════════════

    def _get_extreme_lines(self, lines, height, width):
        if not lines:
            return None, None

        leftmost_line  = None
        rightmost_line = None
        min_x = float('inf')
        max_x = float('-inf')

        for line in lines:
            x_base = self._extrapolate_line_point(line, height)
            if x_base is None:
                continue
            if x_base < min_x:
                min_x = x_base
                leftmost_line = line
            if x_base > max_x:
                max_x = x_base
                rightmost_line = line

        return leftmost_line, rightmost_line

    def _compute_vanishing_point(self, lines):
        if len(lines) < 2:
            return None

        A, B = [], []
        for x1, y1, x2, y2 in lines:
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            norm = np.sqrt(a * a + b * b)
            if norm == 0:
                continue
            A.append([a / norm, b / norm])
            B.append([-c / norm])

        if len(A) < 2:
            return None

        vp, _, _, _ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
        return int(vp[0][0]), int(vp[1][0])

    def _adjust_extreme_grass_lines(self, line_left_curr, line_right_curr,
                                     height, camera_movement, vanishing_point):
        """
        Aggiusta le linee estreme tenendo conto del movimento camera e del
        vanishing point. Aggiorna lo stato interno della classe (corregge il
        bug degli int passati per valore).
        """
        xl1, yl1, xl2, yl2 = line_left_curr
        xr1, yr1, xr2, yr2 = line_right_curr

        angle_left   = np.abs(np.arctan2(yl2 - yl1, xl2 - xl1) * 180.0 / np.pi)
        angle_right  = np.abs(np.arctan2(yr2 - yr1, xr2 - xr1) * 180.0 / np.pi)
        top_x_left   = self._extrapolate_line_point(line_left_curr,  0)
        bottom_x_left= self._extrapolate_line_point(line_left_curr,  height - 1)
        top_x_right  = self._extrapolate_line_point(line_right_curr, 0)
        bottom_x_right=self._extrapolate_line_point(line_right_curr, height - 1)

        extreme_lines = []

        # ── Primo frame: nessuna storia ──────────────────────────────────────
        if not self.line_left and not self.line_right:
            self.line_left.append((line_left_curr,  angle_left,  top_x_left,  bottom_x_left))
            self.line_right.append((line_right_curr, angle_right, top_x_right, bottom_x_right))
            extreme_lines = [line_left_curr, line_right_curr]
            self.line_prev       = line_left_curr
            self.angle_left_prev = angle_left
            self.top_x_prev      = top_x_left
            self.bottom_x_prev   = bottom_x_left
            return extreme_lines

        # ── Linea SINISTRA ───────────────────────────────────────────────────
        accept_left = (
            self.adjusted_left > ADJUSTED_THRESHOLD
            or (bottom_x_left < self.line_left[-1][3]
                and abs(bottom_x_left - self.line_left[-1][3]) > 200
                and abs(top_x_left   - self.line_left[-1][2]) < 200)
            or (abs(angle_left - self.line_left[-1][1]) < 10
                and abs(bottom_x_left - self.line_left[-1][3]) < 60
                and abs(top_x_left   - self.line_left[-1][2]) < 70)
        )

        if accept_left:
            self.line_left.append((line_left_curr, angle_left, top_x_left, bottom_x_left))
            extreme_lines.append(line_left_curr)
            self.adjusted_left   = 0
            self.line_prev       = line_left_curr
            self.angle_left_prev = angle_left
            self.top_x_prev      = top_x_left
            self.bottom_x_prev   = bottom_x_left
        else:
            self.adjusted_left += 1
            xl1, yl1, xl2, yl2 = self.line_left[-1][0]
            dx, dy = camera_movement[0], camera_movement[1]
            line_adjusted = (xl1 - dx, yl1 - dy, xl2 - dx, yl2 - dy)
            bottom_x_adj  = self._extrapolate_line_point(line_adjusted, height - 1)
            angle_adj     = np.abs(np.arctan2((yl2 - dy) - (yl1 - dy),
                                               (xl2 - dx) - (xl1 - dx)) * 180.0 / np.pi)

            if vanishing_point is not None:
                self.v_flag = 1
                line_adjusted, angle_adj = self._adjust_line_to_vanishing_point(
                    vanishing_point, bottom_x_adj, height)

            if (vanishing_point is not None
                    and (vanishing_point[0] - self.top_x_prev) > 120
                    and abs(angle_adj - self.angle_left_prev) < 10):
                self.line_left.append((self.line_prev, self.angle_left_prev,
                                       self.top_x_prev, self.bottom_x_prev))
                extreme_lines.append(self.line_prev)
            else:
                top_x_adj = vanishing_point[0] if self.v_flag else self._extrapolate_line_point(line_adjusted, 0)
                self.line_left.append((line_adjusted, angle_adj, top_x_adj, bottom_x_adj))
                extreme_lines.append(line_adjusted)
                self.line_prev       = line_adjusted
                self.angle_left_prev = angle_adj
                self.top_x_prev      = top_x_adj
                self.bottom_x_prev   = bottom_x_adj

        # ── Linea DESTRA ─────────────────────────────────────────────────────
        accept_right = (
            self.adjusted_right > ADJUSTED_THRESHOLD
            or (bottom_x_right > self.line_right[-1][3]
                and abs(bottom_x_right - self.line_right[-1][3]) > 250
                and abs(top_x_right   - self.line_right[-1][2]) < 200)
            or (abs(angle_right - self.line_right[-1][1]) < 10
                and abs(bottom_x_right - self.line_right[-1][3]) < 60
                and abs(top_x_right   - self.line_right[-1][2]) < 70)
        )

        if accept_right:
            self.line_right.append((line_right_curr, angle_right, top_x_right, bottom_x_right))
            extreme_lines.append(line_right_curr)
            self.adjusted_right = 0
        else:
            self.adjusted_right += 1
            xr1, yr1, xr2, yr2 = self.line_right[-1][0]
            dx, dy = camera_movement[0], camera_movement[1]
            line_adjusted = (xr1 - dx, yr1 - dy, xr2 - dx, yr2 - dy)
            bottom_x_adj  = self._extrapolate_line_point(line_adjusted, height - 1)
            angle_adj     = np.abs(np.arctan2((yr2 - dy) - (yr1 - dy),
                                               (xr2 - dx) - (xr1 - dx)) * 180.0 / np.pi)

            if vanishing_point is not None:
                line_adjusted, angle_adj = self._adjust_line_to_vanishing_point(
                    vanishing_point, bottom_x_adj, height)

            top_x_adj = self._extrapolate_line_point(line_adjusted, 0)
            self.line_right.append((line_adjusted, angle_adj, top_x_adj, bottom_x_adj))
            extreme_lines.append(line_adjusted)

        return extreme_lines if len(extreme_lines) == 2 else None

    # ════════════════════════════════════════════════════════════════════════
    #  UTILITY GEOMETRICA
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extrapolate_line_point(line, target_y):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            return x1
        if abs(y2 - y1) < 1:
            return None
        m = (y2 - y1) / (x2 - x1)
        return int((target_y - y1) / m + x1)

    @staticmethod
    def _extrapolate_horizontal_line(line, target_x):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            return y1
        m = (y2 - y1) / (x2 - x1)
        return int(m * (target_x - x1) + y1)

    @staticmethod
    def _adjust_line_to_vanishing_point(vanishing_point, bottom_x_adj, height):
        vx, vy = vanishing_point
        line_adjusted = (int(vx), int(vy), int(bottom_x_adj), height - 1)
        angle = np.abs(np.degrees(np.arctan2(height - 1 - vy, bottom_x_adj - vx)))
        return line_adjusted, angle

    # ════════════════════════════════════════════════════════════════════════
    #  DISEGNO
    # ════════════════════════════════════════════════════════════════════════

    def _draw_grass_lines(self, image, lines):
        if not lines:
            return image
        height, width, _ = image.shape
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 255), -1)
            cv2.circle(image, (int(x2), int(y2)), 5, (255, 0, 0), -1)
            top_x    = self._extrapolate_line_point(line, 0)
            bottom_x = self._extrapolate_line_point(line, height - 1)
            if top_x is not None and bottom_x is not None:
                cv2.line(image, (top_x, 0), (bottom_x, height - 1), (0, 255, 0), 2)
        return image

    def _draw_extreme_grass_lines(self, image, extreme_lines):
        if not extreme_lines:
            return image
        height, _, _ = image.shape
        for line in extreme_lines:
            top_x    = self._extrapolate_line_point(line, 0)
            bottom_x = self._extrapolate_line_point(line, height - 1)
            if top_x is not None and bottom_x is not None:
                cv2.line(image, (top_x, 0), (bottom_x, height - 1), (0, 0, 255), 4)
        return image

    def _draw_field_lines(self, image, horizontal_lines, roi_mask=None):
        if not horizontal_lines:
            return image
        img_copy = image.copy()
        height, width = img_copy.shape[:2]

        if roi_mask is not None:
            overlay = img_copy.copy()
            overlay[roi_mask == 255] = (0, 200, 255)
            cv2.addWeighted(overlay, 0.15, img_copy, 0.85, 0, img_copy)
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_copy, contours, -1, (0, 255, 255), 2)

        for line in horizontal_lines:
            if line is None:
                continue
            y_left  = self._extrapolate_horizontal_line(line, 0)
            y_right = self._extrapolate_horizontal_line(line, width - 1)
            cv2.line(img_copy, (0, y_left), (width - 1, y_right), (255, 255, 255), 3)

        return img_copy


# ════════════════════════════════════════════════════════════════════════════
#  HELPER PRIVATO — Stabilizzatore EMA per le linee bianche
# ════════════════════════════════════════════════════════════════════════════

class _LineStabilizer:
    """
    Applica EMA (Exponential Moving Average) alla coordinata Y
    delle due linee orizzontali tra frame consecutivi.
    Rigetta aggiornamenti con salti troppo grandi (outlier).
    """

    def __init__(self):
        self.far_y   = None
        self.near_y  = None
        self._last_far  = None
        self._last_near = None

    def update(self, far_line, near_line):
        new_far_y  = self._mean_y(far_line)
        new_near_y = self._mean_y(near_line)

        self.far_y  = self._ema(self.far_y,  new_far_y)
        self.near_y = self._ema(self.near_y, new_near_y)

        stable_far  = self._shift_to_y(far_line  if far_line  is not None else self._last_far,  self.far_y)
        stable_near = self._shift_to_y(near_line if near_line is not None else self._last_near, self.near_y)

        if far_line  is not None: self._last_far  = far_line
        if near_line is not None: self._last_near = near_line

        return stable_far, stable_near

    @staticmethod
    def _mean_y(line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        return (y1 + y2) / 2.0

    @staticmethod
    def _ema(current, new_val):
        if new_val is None:
            return current
        if current is None:
            return new_val
        if abs(new_val - current) >= MAX_JUMP_PX:
            return current          # outlier: teniamo il valore precedente
        return EMA_ALPHA * new_val + (1 - EMA_ALPHA) * current

    @staticmethod
    def _shift_to_y(line, target_y):
        if line is None or target_y is None:
            return None
        x1, y1, x2, y2 = line
        dy = target_y - (y1 + y2) / 2.0
        return (x1, int(y1 + dy), x2, int(y2 + dy))