# Bilanciamento Luci
# Maschera Colore 
# Rilevamento Bordi 
# Regressione Matematica.
import numpy as np 
import cv2
from sklearn.cluster import DBSCAN

img_name = './input_videos/image.png'

# ── COSTANTI DI TUNING ──────────────────────────────────────────────────────
WHITE_V_MIN       = 160   # Valore minimo HSV per "bianco" (abbassato da 180)
WHITE_S_MAX       = 60    # Saturazione massima HSV per "bianco"
ROI_TOP_FRAC      = 0.22  # La ROI parte dal 18% dall'alto (era 22%) → include la linea lontana > da 20 a 28
ROI_BOTTOM_FRAC   = 0.89  # La ROI finisce al 87% → esclude la pubblicità a bordocampo > da 87 a 82
HOUGH_THRESHOLD   = 60
HOUGH_MIN_LEN     = 200
HOUGH_MAX_GAP     = 25
ANGLE_TOLERANCE   = 20    # gradi: linea "orizzontale" se angolo < 20°
FAR_LINE_MAX_Y    = 0.42  # la linea lontana deve stare sopra il 42% del frame
NEAR_LINE_MIN_Y   = 0.58  # la linea vicina deve stare sotto il 58% del frame
EMA_ALPHA         = 0.35  # smoothing temporale (0=congelato, 1=nessuno smoothing)
MAX_JUMP_PX       = 80    # salto massimo ammesso tra frame consecutivi
# ────────────────────────────────────────────────────────────────────────────

roi_top_curr = ROI_TOP_FRAC
roi_bottom_curr = ROI_BOTTOM_FRAC

# ── STABILIZZATORE TEMPORALE ────────────────────────────────────────────────

class LineStabilizer:
    """
    Applica EMA (Exponential Moving Average) ai parametri y
    delle due linee orizzontali tra frame consecutivi.
    Rigetta aggiornamenti con salti troppo grandi (outlier).
    """
    def __init__(self):
        self.far_y  = None   # y media della linea lontana
        self.near_y = None   # y media della linea vicina

    def _line_mean_y(self, line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        return (y1 + y2) / 2.0

    def _shift_line_to_y(self, line, target_y):
        """Trasla verticalmente la linea al target_y mantenendo la pendenza."""
        if line is None:
            return None
        x1, y1, x2, y2 = line
        dy = target_y - (y1 + y2) / 2.0
        return (x1, int(y1 + dy), x2, int(y2 + dy))

    def update(self, far_line, near_line):
        """
        Aggiorna lo stato con le nuove linee rilevate.
        Ritorna (far_line_stabile, near_line_stabile).
        """
        new_far_y  = self._line_mean_y(far_line)
        new_near_y = self._line_mean_y(near_line)

        # ── Linea lontana ──
        if new_far_y is not None:
            if self.far_y is None:
                self.far_y = new_far_y          # primo frame: accettiamo sempre
            else:
                jump = abs(new_far_y - self.far_y)
                if jump < MAX_JUMP_PX:           # salto accettabile → EMA
                    self.far_y = EMA_ALPHA * new_far_y + (1 - EMA_ALPHA) * self.far_y
                # altrimenti: ignoriamo l'outlier, teniamo il valore precedente

        # ── Linea vicina ──
        if new_near_y is not None:
            if self.near_y is None:
                self.near_y = new_near_y
            else:
                jump = abs(new_near_y - self.near_y)
                if jump < MAX_JUMP_PX:
                    self.near_y = EMA_ALPHA * new_near_y + (1 - EMA_ALPHA) * self.near_y

        # Ricostruiamo le linee traslate alla y smoothed
        stable_far  = self._shift_line_to_y(far_line  if far_line  is not None else self._last_far,  int(self.far_y))  if self.far_y  is not None else None
        stable_near = self._shift_line_to_y(near_line if near_line is not None else self._last_near, int(self.near_y)) if self.near_y is not None else None

        # Salviamo l'ultima linea valida come fallback
        if far_line  is not None: self._last_far  = far_line
        if near_line is not None: self._last_near = near_line

        return stable_far, stable_near

    # Inizializziamo i fallback come None
    _last_far  = None
    _last_near = None
# -------------------------------------------------------------------------------------------------------------------------------
 
            ####################################
            ##       BILANCIAMENTO LUCI       ##
            ####################################

# Per "pulire" l'immagine e rendere le linee d'erba visibili ovunque, 
# la tecnica migliore è l'uso del CLAHE (Contrast Limited Adaptive Histogram Equalization).

# Come funziona il Bilanciamento Adattivo (CLAHE)
# A differenza di un bilanciamento normale (che schiarisce tutto il frame allo stesso modo), 
# il CLAHE divide l'immagine in una griglia (chiamata tileGridSize) e ottimizza il contrasto cella per cella.
# 
# Distribuzione locale: 
# Se una cella è molto scura (ombra a destra), il CLAHE ne espande i valori per renderla visibile. 
# Se è molto chiara (sole a sinistra), cerca di recuperare i dettagli.
# 
# Limite del rumore: 
# Il "Contrast Limited" impedisce all'algoritmo di esagerare, 
# evitando che il rumore digitale diventi troppo evidente nelle zone scure.

def preprocess_image(image):
    # 1. Convertiamo da BGR (colori standard) a LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Dividiamo i canali: L (Luce), A (Verde-Rosso), B (Blu-Giallo)
    l, a, b = cv2.split(lab)
    
    # 3. Creiamo l'oggetto CLAHE
    # clipLimit: quanto deve essere "aggressivo" il contrasto (2.0 - 5.0 è l'ideale)
    # tileGridSize: la dimensione della griglia di analisi
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    # 4. Applichiamo solo al canale della luce (L)
    l_balanced = clahe.apply(l)
    
    # 5. Riuniamo i canali e torniamo in BGR per le fasi successive
    lab_final = cv2.merge((l_balanced, a, b))
    final_image = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    return final_image

# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##        MASCHERA COLORE        ##
            ###################################

def remove_noise_by_area(mask, min_area=1500):
    """
    Trova tutti i contorni nella maschera e rimuove quelli troppo piccoli.
    """
    # Trova i contorni (RETR_EXTERNAL prende solo i contorni esterni, non i buchi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Calcola l'area del singolo contorno
        area = cv2.contourArea(cnt)
        
        # Se l'area è più piccola del nostro limite, colora il contorno di nero
        if area < min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
            
    return mask

# Ora che abbiamo un'immagine con una luminosità bilanciata grazie al CLAHE, possiamo passare alla segmentazione del colore.
# 
# L'obiettivo è isolare i due toni di verde (erba chiara e erba scura) 
# per individuare il punto esatto in cui si toccano: quel "confine" è la nostra linea del campo.
# 
# 1. Perché usare lo spazio colore HSV?
# Invece del classico RGB (Rosso, Verde, Blu), per la maschera colore usiamo HSV (Hue, Saturation, Value). 
# È molto più efficace perché separa l'informazione del colore da quella della luce:
# - Hue (Tonalità): Il tipo di colore (es. "Verde"). Resta simile sia al sole che all'ombra.
# - Saturation (Saturazione): Quanto è intenso il colore.
# - Value (Valore): La luminosità.
# 
# In RGB, se un'ombra cade sul prato, cambiano tutti e tre i valori (R, G, B). 
# In HSV, cambierà principalmente il Value, rendendo molto più semplice dire al computer: 
# "Prendi tutto ciò che è verde, a prescindere da quanto sia illuminato".

def get_grass_masks(balanced_image):
    hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
    
    # --- MASCHERA CHIARA (Light) ---
    # Alziamo la saturazione minima (da 40 a 60) per ignorare il verde sbiadito
    # Alziamo il valore minimo (da 100 a 150) per prendere solo il "brillante"
    lower_light = np.array([35, 90, 145]) 
    upper_light = np.array([55, 255, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # --- MASCHERA SCURA (Dark) ---
    # Restringiamo il valore massimo (da 100 a 140) per non sovrapporci troppo al chiaro
    lower_dark = np.array([35, 50, 20])
    upper_dark = np.array([55, 255, 100])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
  # --- PULIZIA MORFOLOGICA POTENZIATA ---

    # 1. Kernel verticale molto più alto per "cucire" i buchi distanti
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 45))

    # 2. Applica una DILATAZIONE prima del closing (opzionale ma efficace)
    # Questo espande il bianco per "mangiare" le macchie nere interne
    #mask_dark = cv2.dilate(mask_dark, np.ones((3,3), np.uint8), iterations=1)
    
    # Opening: elimina piccoli puntini isolati (rumore negli spalti)
    kernel_open = np.ones((7,7), np.uint8)
    # Applichiamo Closing (unisce) e Opening (pulisce) a entrambe
    for m in [mask_light, mask_dark]:
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_v, dst=m)
        cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open, dst=m)

    # --- 3. FILTRO AREA (Il colpo finale) ---
    # Questa funzione colorerà di NERO i puntini bianchi che non sono strisce
    mask_light_final = remove_noise_by_area(mask_light, min_area=2000)
    mask_dark_final = remove_noise_by_area(mask_dark, min_area=2000)

    return mask_light_final, mask_dark_final

def get_white_lines_mask(balanced_image):
    hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)

    # Range principale: bianco puro
    mask1 = cv2.inRange(hsv,
                        np.array([0,   0,   WHITE_V_MIN]),
                        np.array([180, WHITE_S_MAX, 255]))

    # Range secondario: bianco "sporco" / giallastro sotto il sole
    mask2 = cv2.inRange(hsv,
                        np.array([15,  0,   200]),
                        np.array([40,  40,  255]))

    mask_white = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    return mask_white, mask_white

SENSITIVITY_TOP    = 8.5   # quanto si sposta roi_top per unità di delta
SENSITIVITY_BOTTOM = 8.0   # quanto si sposta roi_bottom per unità di delta

def get_field_roi_mask_strict(height, width, zoom_factor):
    """
    ROI più stretta: esclude le tribune in alto
    e la pubblicità a bordocampo in basso.
    """
    global roi_top_curr, roi_bottom_curr
    if(zoom_factor != 1):
        
        delta = 1.0 - zoom_factor  # positivo = zoom out, negativo = zoom in

        roi_top_curr    = ROI_TOP_FRAC    + delta * SENSITIVITY_TOP
        roi_bottom_curr = ROI_BOTTOM_FRAC - delta * SENSITIVITY_BOTTOM

        # Clamp per sicurezza
        roi_top_curr    = max(0.0, min(roi_top_curr,    1.0))
        roi_bottom_curr = max(0.0, min(roi_bottom_curr, 1.0))
        
    roi = np.zeros((height, width), dtype=np.uint8)
    points = np.array([
        [int(width * 0.01), int(height * roi_top_curr)],
        [int(width * 0.99), int(height * roi_top_curr)],
        [int(width * 0.99), int(height * roi_bottom_curr)],
        [int(width * 0.01), int(height * roi_bottom_curr)],
    ], np.int32)
    cv2.fillPoly(roi, [points], 255)
    return roi


def get_boundary_lines_separated(edges, height, width):
    """
    Rileva linee orizzontali e le separa subito in
    'lontana' (alta nel frame) e 'vicina' (bassa nel frame).
    Ritorna (linea_lontana, linea_vicina) come singole linee
    (la più lunga per categoria), oppure None se non trovata.
    """
    hough = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )

    if hough is None:
        return None, None

    far_candidates  = []  # linee nella metà superiore
    near_candidates = []  # linee nella metà inferiore

    for line in [l[0] for l in hough]:
        x1, y1, x2, y2 = line
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        # Teniamo solo linee quasi-orizzontali
        if not (angle < ANGLE_TOLERANCE or angle > 180 - ANGLE_TOLERANCE):
            continue

        # Lunghezza del segmento (più è lungo, più è affidabile)
        length = np.hypot(x2 - x1, y2 - y1)

        y_center = (y1 + y2) / 2.0

        if y_center < height * FAR_LINE_MAX_Y:
            far_candidates.append((length, line))
        elif y_center > height * NEAR_LINE_MIN_Y:
            near_candidates.append((length, line))
        # Le linee nella zona centrale vengono ignorate

    # Prendiamo la linea più lunga per ogni categoria
    best_far  = max(far_candidates,  key=lambda x: x[0])[1] if far_candidates  else None
    best_near = max(near_candidates, key=lambda x: x[0])[1] if near_candidates else None

    return best_far, best_near
# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##       RILEVAMENTO BORDI       ##
            ###################################

def get_clean_edges(mask_light):
    # 1. Smoothing: fondamentale per eliminare i bordi "seghettati"
    # Un Gaussian Blur leggero ammorbidisce i pixel prima di Canny
    blurred = cv2.GaussianBlur(mask_light, (5, 5), 0)
    
    # 2. Canny Edge Detection
    # Usiamo soglie distanti (es. 50 e 150) per ignorare il rumore residuo
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##       LINEE STABILI          ##
            ###################################

def get_stable_lines(edges, height, width):
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=170, maxLineGap=30)
    
    if hough_lines is None:
        return []

    lines_data = [] # Memorizziamo [bottom_x, top_x, original_line]
    
    for line in [l[0] for l in hough_lines]:
        x1, y1, x2, y2 = line
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        
        bottom_x = extrapolate_line_point(line, height - 1)
        top_x = extrapolate_line_point(line, 0)

        # Filtro angolare e di bordo
        if bottom_x is not None and 25 < angle < 85:
            if 0 <= top_x <= width:
                lines_data.append({
                    'bottom_x': bottom_x,
                    'top_x': top_x,
                    'line': line
                })

    if not lines_data:
        return []

    # 1. DBSCAN basato sulla coordinata bottom_x
    X = np.array([ld['bottom_x'] for ld in lines_data]).reshape(-1, 1)
    
    # Aumentiamo min_samples per eliminare i "single point" isolati (rumore)
    db = DBSCAN(eps=60, min_samples=1).fit(X)
    labels = db.labels_

    grouped_results = {}
    for i, label in enumerate(labels):
        if label == -1: continue # Salta il rumore isolato trovato da DBSCAN
        
        if label not in grouped_results:
            grouped_results[label] = []
        grouped_results[label].append(lines_data[i])

    final_lines = []

    # 2. Pulizia interna ai gruppi (Coerenza Top X)
    for label, group in grouped_results.items():
        # Calcoliamo la mediana della top_x per questo gruppo
        top_coords = [ld['top_x'] for ld in group]
        median_top = np.median(top_coords)
        
        # Teniamo solo le linee la cui top_x non dista più di 50px dalla mediana del gruppo
        valid_group_lines = [
            ld['line'] for ld in group 
            if abs(ld['top_x'] - median_top) < 50
        ]
            
        # Se dopo il filtro il gruppo è ancora solido, mediamo o prendiamo la più lunga
        if len(valid_group_lines) >= 2:
            # Opzione: aggiungi la linea media del gruppo o tutte le linee pulite
            final_lines.extend(valid_group_lines)
        
    return final_lines

def get_boundary_lines_simple(edges, height):
    """
    Rileva le linee bianche orizzontali senza DBSCAN.
    Ritorna (linea_lontana, linea_vicina).
    """
    # 1. Rilevamento linee con Hough (Parametri ottimizzati per linee lunghe)
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=170, maxLineGap=30)

    
    if hough_lines is None:
        return None, None

    candidate_lines = []
    
    for line in [l[0] for l in hough_lines]:
        x1, y1, x2, y2 = line
        
        # 2. Calcolo angolo (vicino a 0 gradi = orizzontale)
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        
        # Consideriamo solo linee molto piatte (tolleranza 15 gradi)
        if (angle < 25 or angle > 155) and ((y1 < height * 0.35 and y2 < height * 0.35) or (y1 > height * 0.80 and y2 > height * 0.80)):            
            candidate_lines.append(line)
    
    return candidate_lines

def adjust_line_to_vanishing_point(vanishing_point, bottom_x_adj, height):
    vx, vy = vanishing_point

    new_x1 = vx
    new_y1 = vy
    new_x2 = bottom_x_adj
    new_y2 = height - 1

    line_adjusted = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))

    angle_left_adjusted = np.abs(
        np.degrees(np.arctan2(new_y2 - new_y1,
                              new_x2 - new_x1))
    )
    return line_adjusted, angle_left_adjusted

def adjust_extreme_grass_lines(line_left_curr, line_right_curr, height, camera_movement, line_left, line_right, adjusted_left, adjusted_right, vanishing_point, v_flag, line_prev, angle_left_prev, top_x_prev, bottom_x_prev):
    # Questa funzione prende le linee correnti e le confronta con quelle precedenti, 
    # tenendo conto del movimento della camera e del vanishing point per decidere se accettare la nuova linea o mantenere quella vecchia.
    # Restituisce la lista delle linee estreme da disegnare (può essere la nuova linea, quella vecchia o una versione "aggiustata" della vecchia).
    adjusted_threshold = 35
    xl1, yl1, xl2, yl2 = line_left_curr
    xr1, yr1, xr2, yr2 = line_right_curr

    angle_left = np.abs(np.arctan2(yl2 - yl1, xl2 - xl1) * 180.0 / np.pi)
    angle_right = np.abs(np.arctan2(yr2 - yr1, xr2 - xr1) * 180.0 / np.pi)
    top_x_left = extrapolate_line_point(line_left_curr, 0)
    bottom_x_left = extrapolate_line_point(line_left_curr, height - 1)
    top_x_right = extrapolate_line_point(line_right_curr, 0)
    bottom_x_right = extrapolate_line_point(line_right_curr, height - 1)

    extreme_lines = []

    if len(line_left) == 0 and len(line_right) == 0:
        line_left.append((line_left_curr, angle_left, top_x_left, bottom_x_left))
        line_right.append((line_right_curr, angle_right, top_x_right, bottom_x_right))
        extreme_lines.append(line_left_curr)
        extreme_lines.append(line_right_curr)
        line_prev = line_left_curr
        angle_left_prev = angle_left
        top_x_prev = top_x_left
        bottom_x_prev = bottom_x_left
    else:
        if adjusted_left > adjusted_threshold or (bottom_x_left < line_left[-1][3] and np.abs(bottom_x_left -line_left[-1][3]) > 200 and np.abs(top_x_left -line_left[-1][2]) < 200) or (np.abs(angle_left - line_left[-1][1])<10 and (np.abs(bottom_x_left -line_left[-1][3])<60 and np.abs(top_x_left -line_left[-1][2])<70)):
            line_left.append((line_left_curr, angle_left, top_x_left, bottom_x_left))
            extreme_lines.append(line_left_curr)
            adjusted_left = 0
            line_prev = line_left_curr
            angle_left_prev = angle_left
            top_x_prev = top_x_left
            bottom_x_prev = bottom_x_left
        else:
            adjusted_left += 1
            xl1, yl1, xl2, yl2 = line_left[-1][0]
            line_adjusted = (xl1-camera_movement[0], yl1-camera_movement[1], xl2-camera_movement[0], yl2-camera_movement[1])
            angle_left_adjusted = np.abs(np.arctan2(yl2-camera_movement[1] - yl1-camera_movement[1], xl2 -camera_movement[0] - xl1 -camera_movement[0]) * 180.0 / np.pi)
            top_x_adj = extrapolate_line_point(line_adjusted, 0)
            bottom_x_adj = extrapolate_line_point(line_adjusted, height - 1)

            
            if vanishing_point is not None:
                v_flag=1      
                line_adjusted, angle_left_adjusted = adjust_line_to_vanishing_point(vanishing_point, bottom_x_adj, height)

            if ((vanishing_point[0] - top_x_prev) > 120 and vanishing_point is not None and (np.abs(angle_left_adjusted - angle_left_prev)<10 )):
                line_left.append((line_prev,angle_left_prev, top_x_prev, bottom_x_prev))
                extreme_lines.append(line_prev)
            else:
                line_left.append((line_adjusted,angle_left_adjusted, vanishing_point[0] if v_flag else top_x_adj, bottom_x_adj))
                extreme_lines.append(line_adjusted)
                line_prev = line_adjusted
                angle_left_prev = angle_left_adjusted
                top_x_prev = vanishing_point[0] if v_flag else top_x_adj
                bottom_x_prev = bottom_x_adj

        if adjusted_right > adjusted_threshold or (bottom_x_right > line_right[-1][3] and np.abs(bottom_x_right -line_right[-1][3]) > 250 and np.abs(top_x_right -line_right[-1][2]) < 200) or (np.abs(angle_right - line_right[-1][1])<10 and (np.abs(bottom_x_right -line_right[-1][3])<60 and np.abs(top_x_right -line_right[-1][2])<70)):
            line_right.append((line_right_curr, angle_right, top_x_right, bottom_x_right))
            extreme_lines.append(line_right_curr)
            adjusted_right = 0
        else:
            adjusted_right += 1
            xr1, yr1, xr2, yr2 = line_right[-1][0]
            line_adjusted = (xr1-camera_movement[0], yr1-camera_movement[1], xr2-camera_movement[0], yr2-camera_movement[1])
            angle_right_adjusted = np.abs(np.arctan2(yr2-camera_movement[1] - yr1-camera_movement[1], xr2 -camera_movement[0] - xr1 -camera_movement[0]) * 180.0 / np.pi)
            top_x_adj = extrapolate_line_point(line_adjusted, 0)
            bottom_x_adj = extrapolate_line_point(line_adjusted, height - 1)

            if vanishing_point is not None:
                line_adjusted, angle_right_adjusted = adjust_line_to_vanishing_point(vanishing_point, bottom_x_adj, height)

            line_right.append((line_adjusted,angle_right_adjusted, top_x_adj, bottom_x_adj))
            extreme_lines.append(line_adjusted)
            
    return extreme_lines if len(extreme_lines) == 2 else None


# -------------------------------------------------------------------------------------------------------------------------------

            ###################################
            ##     LINEE ESTREME DEL PRATO   ##
            ###################################

def get_extreme_lines(lines, height, width):
    if not lines: return None, None

    leftmost_line = None
    rightmost_line = None
    min_x = float('inf')
    max_x = float('-inf')

    for line in lines:
        # 1. Ottieni la X proiettata sul fondo del video
        x_base = extrapolate_line_point(line, height)
        
        if x_base is None: continue

        # 2. Aggiorna la linea più a sinistra
        if x_base < min_x:
            min_x = x_base
            leftmost_line = line

        # 3. Aggiorna la linea più a destra
        if x_base > max_x:
            max_x = x_base
            rightmost_line = line

    return leftmost_line, rightmost_line

# -------------------------------------------------------------------------------------------------------------------------------

        ##################################
        ##       VANISHING POINT        ##
        ##################################

def compute_vanishing_point(lines):
    """
    lines: lista di linee [(x1,y1,x2,y2), ...]
    return: (x, y) vanishing point oppure None
    """

    if len(lines) < 2:
        return None

    A = []
    B = []

    for x1, y1, x2, y2 in lines:

        # Forma ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2

        # Normalizzazione per stabilità numerica
        norm = np.sqrt(a*a + b*b)
        if norm == 0:
            continue

        a /= norm
        b /= norm
        c /= norm

        A.append([a, b])
        B.append([-c])

    A = np.array(A)
    B = np.array(B)

    if len(A) < 2:
        return None

    # Risoluzione least squares
    vp, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return int(vp[0][0]), int(vp[1][0])

# -------------------------------------------------------------------------------------------------------------------------------
            ###################################
            ##    DISEGNO LINEE SUL PRATO    ##
            ###################################

def extrapolate_line_point(line, target_y):
    """Calcola la X di una linea data una certa Y (anche fuori frame)"""

    x1, y1, x2, y2 = line

    if x2 - x1 == 0: return x1 # Linea verticale
     # Linea orizzontale (o quasi)
    if abs(y2 - y1) < 1:
        return None
    
    m = (y2 - y1) / (x2 - x1) # Pendenza
    # Formula: y - y1 = m(x - x1)  => x = (y - y1)/m + x1
    target_x = (target_y - y1) / m + x1
    return int(target_x)

def extrapolate_horizontal_line(line, target_x):
    """Calcola la Y di una linea data una certa X (estensione orizzontale)"""
    x1, y1, x2, y2 = line
    
    # Se la linea è perfettamente verticale (non dovrebbe succedere qui)
    if x2 - x1 == 0:
        return y1
        
    # Calcolo pendenza m e intercetta q: y = mx + q
    m = (y2 - y1) / (x2 - x1)
    # y - y1 = m(x - x1)  => y = m(target_x - x1) + y1
    target_y = m * (target_x - x1) + y1
    
    return int(target_y)

def draw_grass_lines(image, lines):
    if lines is None:
        return image

    height, width, _ = image.shape
    
    for line in lines:
        # 1. Recuperiamo le coordinate del segmento rilevato
        x1, y1, x2, y2 = line
        
        # 2. DISEGNO DEI PUNTI ORIGINALI (Punti di inizio e fine rilevati)
        # Disegniamo dei cerchietti rossi per vederli bene
        # cv2.circle(immagine, centro, raggio, colore, spessore)
        cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 255), -1) # Punto 1 (Rosso)
        cv2.circle(image, (int(x2), int(y2)), 5, (255, 0, 0), -1) # Punto 2 (Blu)

        # 3. ESTENSIONE DELLA LINEA
        top_x = extrapolate_line_point(line, 0)
        bottom_x = extrapolate_line_point(line, height - 1)
        
        if top_x is None or bottom_x is None:
            continue
            
        # 4. DISEGNO DELLA LINEA ESTRAPOLATA (Verde)
        cv2.line(image, (top_x, 0), (bottom_x, height - 1), (0, 255, 0), 2)
        
    return image

def draw_field_lines(image, horizontal_lines, roi_mask=None):
    """
    Disegna le linee orizzontali (bordo campo) estendendole per tutta la larghezza.
    horizontal_lines: lista di tuple o array [(x1,y1,x2,y2), ...]
    """
    if not horizontal_lines:
        return image

    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    
    # ── DEBUG: disegno ROI come trapezio ────────────────────────────────────
    if roi_mask is not None:
        # Overlay semitrasparente giallo all'interno della ROI
        overlay = img_copy.copy()
        overlay[roi_mask == 255] = (0, 200, 255)  # Arancione-giallo
        cv2.addWeighted(overlay, 0.15, img_copy, 0.85, 0, img_copy)

        # Contorno del trapezio in giallo pieno
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_copy, contours, -1, (0, 255, 255), 2)
    
    for line in horizontal_lines:
        if line is None:
            continue
            
        # 1. Calcoliamo i punti della linea alle estremità sinistra (x=0) e destra (x=width)
        y_left = extrapolate_horizontal_line(line, 0)
        y_right = extrapolate_horizontal_line(line, width - 1)

        # 2. Disegno della linea principale (Bianca, spessa)
        # Usiamo il colore bianco (255, 255, 255) per le linee di delimitazione
        cv2.line(img_copy, (0, y_left), (width - 1, y_right), (255, 255, 255), 3)

    return img_copy

def draw_extreme_grass_lines(image, extreme_lines):
    if not extreme_lines:
        return image

    height, width, _ = image.shape
    
    for line in extreme_lines:
        x1, y1, x2, y2 = line
        
        top_x = extrapolate_line_point(line, 0)
        bottom_x = extrapolate_line_point(line, height - 1)
        
        if top_x is None or bottom_x is None:
            continue
            
        # Linea estreme in rosso (spessore maggiore per evidenziare)
        cv2.line(image, (top_x, 0), (bottom_x, height - 1), (0, 0, 255), 4)
        
    return image

# -------------------------------------------------------------------------------------------------------------------------------
            
            ###################################
            ##             MAIN              ##
            ###################################
def draw_detected_grass_lines_on_video(video_frames, camera_movement_per_frame, type):
    output_frames = []
    
    # Per type=0 (erba) — stato esistente invariato
    line_left, line_right = [], []
    adjusted_left, adjusted_right = 0, 0
    line_prev, angle_left_prev, top_x_prev, bottom_x_prev = None, None, None, None
    v_flag = 0

    # Per type=1 (linee bianche) — nuovo stabilizzatore
    stabilizer = LineStabilizer()

    for frame_idx, frame in enumerate(video_frames):
        height, width = frame.shape[:2]
        output_frame = frame.copy()

        preprocess_imaged = preprocess_image(frame)

        if type:
            # ── PIPELINE LINEE BIANCHE (RISCRITTA) ──────────────────────────
            
            # 1. Maschera bianco migliorata
            mask_white, _ = get_white_lines_mask(preprocess_imaged)

            # 2. ROI più stretta (esclude tribune e pubblicità)
            roi = get_field_roi_mask_strict(height, width,camera_movement_per_frame[frame_idx][2])
            mask_roi = cv2.bitwise_and(mask_white, roi)

            # 3. Bordi
            edges = get_clean_edges(mask_roi)

            # 4. Rilevamento linee già separate (lontana / vicina)
            far_raw, near_raw = get_boundary_lines_separated(edges, height, width)

            # 5. Stabilizzazione temporale
            far_line, near_line = stabilizer.update(far_raw, near_raw)

            # 6. Disegno
            output_frame = draw_field_lines(output_frame, 
                                             [l for l in [far_line, near_line] if l is not None], roi_mask=roi)

        else:
            # ── PIPELINE ERBA (invariata) ────────────────────────────────────
            grass_l, grass_d = get_grass_masks(preprocess_imaged)
            field_roi = get_field_roi_mask(grass_l, grass_d)
            mask_light, mask_dark = grass_l, grass_d

            edges_light   = get_clean_edges(mask_light)
            edges_dark    = get_clean_edges(mask_dark)
            combined      = cv2.bitwise_or(edges_light, edges_dark)
            edges_combined = cv2.bitwise_and(combined, field_roi)

            grass_lines     = get_stable_lines(edges_combined, height, width)
            vanishing_point = compute_vanishing_point(grass_lines)
            camera_movement = camera_movement_per_frame[frame_idx]

            line_left_curr, line_right_curr = get_extreme_lines(grass_lines, height, width)
            if line_left_curr is None or line_right_curr is None:
                continue

            extreme_grass_lines = adjust_extreme_grass_lines(
                line_left_curr, line_right_curr, height, camera_movement,
                line_left, line_right, adjusted_left, adjusted_right,
                vanishing_point, v_flag, line_prev, angle_left_prev,
                top_x_prev, bottom_x_prev
            )
            v_flag = 0

            output_frame = draw_grass_lines(output_frame, grass_lines)
            output_frame = draw_extreme_grass_lines(output_frame, extreme_grass_lines)

        output_frames.append(output_frame)

    return output_frames

