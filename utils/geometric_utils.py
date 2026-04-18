# ════════════════════════════════════════════════════════════════════════
#  UTILITY GEOMETRICA
# ════════════════════════════════════════════════════════════════════════

import numpy as np

def extrapolate_line_point(line, target_y):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return x1
    if abs(y2 - y1) < 1:
        return None
    m = (y2 - y1) / (x2 - x1)
    return int((target_y - y1) / m + x1)

def extrapolate_horizontal_line(line, target_x):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return y1
    m = (y2 - y1) / (x2 - x1)
    return int(m * (target_x - x1) + y1)

def adjust_line_to_vanishing_point(vanishing_point, bottom_x_adj, height):
    vx, vy = vanishing_point
    line_adjusted = (int(vx), int(vy), int(bottom_x_adj), height - 1)
    angle = np.abs(np.degrees(np.arctan2(height - 1 - vy, bottom_x_adj - vx)))
    return line_adjusted, angle


def line_intersection(line1, line2):
    """Intersezione tra due segmenti/rette definiti come (x1,y1,x2,y2)."""
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (x, y)

# ════════════════════════════════════════════════════════════════════════
#  CALCOLO TRAPEZI
# ════════════════════════════════════════════════════════════════════════

def compute_trapezoid(far_line, near_line, line_left, line_right, height):
    """
    Calcola i 4 vertici del trapezio dall'intersezione delle linee.
    Ritorna np.ndarray shape (4,2) float32 oppure None.
    """
    if line_left is None or line_right is None:
        return None

    def intersect_with_y(line, y):
        """Restituisce x dove la line interseca la retta orizzontale y=y."""
        x = extrapolate_line_point(line, y)
        return x

    # Bordo inferiore: usa near_line se disponibile, altrimenti y=0
    if near_line is not None:
        # Intersezione linea_sinistra con near_line
        bl = line_intersection(line_left,  near_line)
        br = line_intersection(line_right, near_line)
        if bl is None or br is None:
            return None
        bl_x, bl_y = bl
        br_x, br_y = br
    else:
        bl_x = intersect_with_y(line_left,  height - 1)
        br_x = intersect_with_y(line_right, height - 1)
        if bl_x is None or br_x is None:
            return None
        bl_y = br_y = 0
          
    # Bordo superiore: usa far_line se disponibile, altrimenti y=0
    if far_line is not None:
        # Intersezione linea_sinistra con far_line
        tl = line_intersection(line_left,  far_line)
        tr = line_intersection(line_right, far_line)
        if tl is None or tr is None:
            return None
        tl_x, tl_y = tl
        tr_x, tr_y = tr
    else:
        tl_x = intersect_with_y(line_left,  0)
        tr_x = intersect_with_y(line_right, 0)
        if tl_x is None or tr_x is None:
            return None
        tl_y = tr_y = 0
        
    vertices = np.array([
        [tl_x,  tl_y],   # top-left
        [tr_x,  tr_y],   # top-right
        [br_x,  br_y],   # bottom-right
        [bl_x,  bl_y],   # bottom-left
    ], dtype=np.float32)
    
    return vertices
    #return extend_trapezoid(vertices, offset=30.0)
    
def extend_trapezoid(vertices, offset):
    # vertices è np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])
    # Indici: 0=TL, 1=TR, 2=BR, 3=BL
    
    new_vertices = np.copy(vertices)

    def extend_pair(p_top, p_bottom):
        # Vettore direzione da Top a Bottom
        v = p_bottom - p_top
        length = np.linalg.norm(v)
        if length == 0:
            return p_top, p_bottom
        
        # Vettore unitario
        u = v / length
        
        # Estendi i punti
        new_top = p_top - (u * offset)
        new_bottom = p_bottom + (u * offset)
        return new_top, new_bottom

    # Lato Sinistro (tra TL [0] e BL [3])
    new_vertices[0], new_vertices[3] = extend_pair(vertices[0], vertices[3])
    
    # Lato Destro (tra TR [1] e BR [2])
    new_vertices[1], new_vertices[2] = extend_pair(vertices[1], vertices[2])
    
    return new_vertices