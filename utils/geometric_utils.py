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