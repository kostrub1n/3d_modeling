from PIL import Image
import os, math

W, H = 220, 200
img = Image.new("RGB", (W, H), "white")
px = img.load()

def set_px(x, y, color=(0, 0, 0)):
    if 0 <= x < W and 0 <= y < H:
        px[x, y] = color

def point_in_triangle(p, a, b, c):
    (x, y) = p
    (x1, y1) = a
    (x2, y2) = b
    (x3, y3) = c
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if denom == 0: return False
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    l3 = 1 - l1 - l2
    return (0 <= l1 <= 1) and (0 <= l2 <= 1) and (0 <= l3 <= 1)

def dist2(p, q): return (p[0]-q[0])**2 + (p[1]-q[1])**2

# Брезенхэм: отрезок
def bresenham_line(x0, y0, x1, y1, color=(0,0,0), dash=None, clip_outside_tri=None):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    dash_on = True
    if dash is not None:
        dash_len, gap_len = dash
        counter = dash_len
    else:
        dash_len = gap_len = counter = 0

    tri = clip_outside_tri

    while True:
        ok = True
        if tri is not None:
            ok = not point_in_triangle((x0, y0), *tri)
        if dash is None:
            if ok: set_px(x0, y0, color)
        else:
            if dash_on and ok: set_px(x0, y0, color)
            counter -= 1
            if counter == 0:
                dash_on = not dash_on
                counter = dash_len if dash_on else gap_len

        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy

# Брезенхэм: окружность (со штрихами)
def bresenham_circle(cx, cy, r, color=(0,0,0), arc=None, dash=None):
    x = r; y = 0; d = 1 - r
    dash_on = True
    if dash is not None:
        dash_len, gap_len = dash
        counter = dash_len
    else:
        dash_len = gap_len = counter = 0

    def maybe_plot(X, Y):
        nonlocal dash_on, counter
        if dash is None:
            set_px(int(X), int(Y), color)
        else:
            if dash_on: set_px(int(X), int(Y), color)
            counter -= 1
            if counter == 0:
                dash_on = not dash_on
                counter = dash_len if dash_on else gap_len

    def plot8(cx, cy, x, y):
        pts = [
            (cx + x, cy + y), (cx + y, cy + x),
            (cx - y, cy + x), (cx - x, cy + y),
            (cx - x, cy - y), (cx - y, cy - x),
            (cx + y, cy - x), (cx + x, cy - y),
        ]
        if arc is None:
            for X, Y in pts: maybe_plot(X, Y)
        else:
            a0, a1 = arc
            if a1 < a0: a1 += 2*math.pi
            for X, Y in pts:
                ang = math.atan2(Y - cy, X - cx)
                if ang < a0: ang += 2*math.pi
                if a0 <= ang <= a1: maybe_plot(X, Y)

    while x >= y:
        plot8(cx, cy, x, y)
        y += 1
        if d <= 0: d += 2*y + 1
        else: x -= 1; d += 2*y - 2*x + 1

# Кривая Безье + дуга окружности через Безье
def draw_cubic_bezier(p0, p1, p2, p3, steps=500, color=(0,0,0), clip_outside_tri=None):
    def B(t):
        x = ((1-t)**3)*p0[0] + 3*((1-t)**2)*t*p1[0] + 3*(1-t)*(t**2)*p2[0] + (t**3)*p3[0]
        y = ((1-t)**3)*p0[1] + 3*((1-t)**2)*t*p1[1] + 3*(1-t)*(t**2)*p2[1] + (t**3)*p3[1]
        return int(round(x)), int(round(y))
    x_prev, y_prev = B(0.0)
    for i in range(1, steps + 1):
        t = i / steps
        x, y = B(t)
        bresenham_line(x_prev, y_prev, x, y, color, clip_outside_tri=clip_outside_tri)
        x_prev, y_prev = x, y

def draw_circular_arc_bezier(center, R, t0, t1, segments=None, color=(0,0,0), clip_outside_tri=None):
    while t1 < t0: t1 += 2*math.pi
    total = t1 - t0
    if segments is None:
        segments = max(1, int((total / (math.pi/2)) + 0.999))
    dth = total / segments
    cx, cy = center
    th = t0
    for _ in range(segments):
        th2 = th + dth
        p0 = (cx + R*math.cos(th),  cy + R*math.sin(th))
        p3 = (cx + R*math.cos(th2), cy + R*math.sin(th2))
        alpha = 4/3 * math.tan((th2 - th)/4) * R
        p1 = (p0[0] - alpha*math.sin(th),  p0[1] + alpha*math.cos(th))
        p2 = (p3[0] + alpha*math.sin(th2), p3[1] - alpha*math.cos(th2))
        draw_cubic_bezier(p0, p1, p2, p3, steps=300, color=color, clip_outside_tri=clip_outside_tri)
        th = th2

# Координаты
C = (100, 100); r_small = 20; r_dashed = 17; R_big = 90
A  = (60, 130); Bv = (100, 50); Cv = (140, 130)
outer_tri = (A, Bv, Cv)

Ap = (56, 133); Bp = (100, 45); Cp = (144, 133)
X = (0, 65); Y = (200, 110)

bresenham_line(*A, *Bv); bresenham_line(*Bv, *Cv); bresenham_line(*Cv, *A)

dash = (6, 4)
bresenham_line(*Ap, *Bp, dash=dash); bresenham_line(*Bp, *Cp, dash=dash); bresenham_line(*Cp, *Ap, dash=dash)

bresenham_circle(C[0], C[1], r_small)
bresenham_circle(C[0], C[1], r_dashed, dash=(5, 4))

t0 = math.atan2(X[1] - C[1], X[0] - C[0])
t1 = math.atan2(Y[1] - C[1], Y[0] - C[0])

def pick_lower_arc(center, R, a, b):
    cx, cy = center
    a0, a1 = a, b
    while a1 < a0: a1 += 2 * math.pi
    mid1 = a0 + (a1 - a0) / 2
    y_mid1 = cy + R * math.sin(mid1)

    b0, b1 = b, a
    while b1 < b0: b1 += 2 * math.pi
    mid2 = b0 + (b1 - b0) / 2
    y_mid2 = cy + R * math.sin(mid2)

    return (a0, a1) if y_mid1 >= y_mid2 else (b0, b1)

arc_start, arc_end = pick_lower_arc(C, R_big, t0, t1)
draw_circular_arc_bezier(C, R_big, arc_start, arc_end)

bresenham_line(*X, *Y, clip_outside_tri=outer_tri)

fill_image_path = "maxresdefault.jpg"
if not os.path.exists(fill_image_path):
    tex = Image.new("RGB", (120, 120))
    for i in range(tex.size[0]):
        for j in range(tex.size[1]):
            tex.putpixel((i, j), (180 + (i*2) % 75, 120 + (j*3) % 60, 160 + ((i+j) % 70)))
    tex.save(fill_image_path)

texture = Image.open(fill_image_path).convert("RGB")
mask = Image.new("L", (W, H), 0)
mpx = mask.load()
for y in range(H):
    for x in range(W):
        if point_in_triangle((x, y), *outer_tri) and dist2((x, y), C) > r_small*r_small:
            mpx[x, y] = 255

tex_scaled = texture.resize((W, H))
img.paste(tex_scaled, (0, 0), mask)
def mark(p, color=(0, 0, 0)):
    x, y = p
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            set_px(x+dx, y+dy, color)
for p in (A, Bv, Cv, Ap, Bp, Cp, X, Y, C):
    mark(p)

img.save("TYT.png")
