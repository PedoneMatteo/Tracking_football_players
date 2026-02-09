# Bilanciamento Luci
# Maschera Colore 
# Rilevamento Bordi 
# Regressione Matematica.
import numpy as np 
import cv2

img_name = './input_videos/image.png'
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

def get_stable_lines(edges):
    # threshold=50: numero minimo di intersezioni per definire una linea
    # minLineLength=100: scarta tutti i segmentini piccoli (giocatori, rumore)
    # maxLineGap=50: unisce segmenti che hanno un "buco" in mezzo
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=170, maxLineGap=30)
    
    stable_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # CALCOLO PENDENZA: Fondamentale per eliminare il caos
            # Le strisce del campo sono quasi verticali (prospettiva a parte)
            # Calcoliamo l'angolo: 90 gradi è verticale pura
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            # FILTRO SEVERO: Teniamo solo linee tra 70 e 110 gradi
            # Questo eliminerà tutte quelle linee orizzontali o diagonali pazze
            if 25 < angle < 85:
                stable_lines.append((x1, y1, x2, y2))
                
    return stable_lines

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

def draw_detected_grass_lines_on_video(video_frames):
    output_frames = []
    for frame_idx, frame in enumerate(video_frames):
        # 1) Bilanciamento luci
        preprocess_imaged = preprocess_image(frame)

        # 2) Maschera colore
        mask_light, mask_dark = get_grass_masks(preprocess_imaged)
        
        # Uniamo le maschere per vedere la copertura totale
        combined_grass = cv2.bitwise_or(mask_light, mask_dark)

        # 3) Rilevamento bordi sulle aree combinate
        edges_light = get_clean_edges(mask_light)
        edges_dark = get_clean_edges(mask_dark)
        edges_combined = cv2.bitwise_or(edges_light, edges_dark)

        # 4) Rilevamento linee stabili
        grass_lines = get_stable_lines(edges_combined)

        # ... dopo aver ottenuto 'grass_lines' dalla funzione get_stable_lines ...

        height, width = frame.shape[:2]
        line_left, line_right = get_extreme_lines(grass_lines, height, width)

        # Creiamo una lista con solo le due linee trovate per poter usare la tua funzione draw
        extreme_lines = []
        if line_left is not None: extreme_lines.append(line_left)
        if line_right is not None: extreme_lines.append(line_right)

        # Disegna il risultato
        output_frame = draw_grass_lines(frame.copy(), grass_lines)
        output_frame = draw_extreme_grass_lines(output_frame, extreme_lines)
        output_frames.append(output_frame)
    return output_frames
