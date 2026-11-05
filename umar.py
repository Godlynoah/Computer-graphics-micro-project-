"""
hospital3d.py  -  Aster Medical: Two-Floor Hospital
===================================================================
Techniques : PERSPECTIVE PROJECTION + Z-BUFFER (hidden-surface removal)
Objective  : a 3D hospital environment with realistic navigation and depth rendering.
Outcome    : an interactive hospital model with correct perspective and hidden-surface rendering.

The 3D engine is written from scratch with NumPy (no OpenGL / game engine).

INSTALL    pip install numpy pygame pillow      (pygame = window only, pillow = screenshots)

RUN        python hospital3d.py                          interactive walk-through
           python hospital3d.py --shot out.png --pos 15.5 8.3 --yaw 180 --mode depth
           python hospital3d.py --tour tour.gif          animated guided tour
           python hospital3d.py --export my_folder       3D model files: OBJ + MTL + GLB

CONTROLS   W/A/S/D walk   Arrows or hold RIGHT mouse = look   Shift run
           1..9,0 teleport to a room        Z  Z-buffer ON/OFF (see the errors without it)
           V depth-buffer view   F hidden-line wireframe   C ceiling   M mini-map
           E lift   Walk stairs in west lobby / PageUp/PageDown floor shortcuts   TAB department directory
           T two-floor tour   O overview   F12 screenshot   F3 render statistics
           [ ] resolution   - = field of view   H help   ESC quit

FILE LAYOUT (sections below)
  1. Engine    : Scene, Camera, collision, render() = projection + Z-buffer, mini-map
  2. Hospital  : floor plan, rooms, furniture, waypoints, tour path
  3. Exporter  : OBJ/MTL/GLB writer
  4. Viewer    : pygame window, screenshots, tour GIF, command line
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np


# ============================================================================
# 1. ENGINE - scene, camera, perspective projection + Z-buffer
# ============================================================================

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
WALL_H = 3.2          # ceiling height (m)
FLOOR_HEIGHT = 3.8    # two physically stacked storeys, including structural slab
FLOOR_NAMES = ("Ground / Clinical care", "Level 1 / Recovery & rehabilitation")
DOOR_H = 2.2          # door height (m)
WALL_T = 0.20         # wall thickness (m)
EYE_H = 1.65          # camera (eye) height
PLAYER_R = 0.28       # collision radius
NEAR = 0.08           # near clipping plane
FAR = 45.0            # used for fog / depth visualisation
SKY = np.array([0.60, 0.77, 0.94])
FOG = np.array([0.82, 0.88, 0.94])

WALL = (0.94, 0.95, 0.94)
STEEL = (0.72, 0.75, 0.78)
WHITE = (0.96, 0.97, 0.98)
WOOD = (0.62, 0.44, 0.28)
DARK = (0.16, 0.18, 0.22)
RED = (0.82, 0.12, 0.14)
GREEN = (0.20, 0.62, 0.38)
BLUE = (0.25, 0.50, 0.82)
TEAL = (0.12, 0.58, 0.62)
LEAF = (0.18, 0.55, 0.25)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


L1 = _unit((0.55, 0.80, 0.30))
L2 = _unit((-0.45, 0.45, -0.75))


# ----------------------------------------------------------------------------
# Scene description
# ----------------------------------------------------------------------------
class Scene:
    """Collects triangles (the 3D model) and 2-D colliders (for navigation)."""

    def __init__(self):
        self.tris = []        # (v0, v1, v2, normal, colour, colour2, tile, emit, group)
        self.colliders = []   # (x0, z0, x1, z1)
        self.rooms = []       # (name, x0, z0, x1, z1)
        self.group = 0        # 0 = normal, 1 = ceiling (can be toggled off)
        self.floor_quads = [] # (x0,z0,x1,z1,col,col2,tile,y) for model export
        self._arrays = None

    # -- primitives ---------------------------------------------------------
    def quad(self, p0, p1, p2, p3, n, col, col2=None, tile=0.0, emit=0.0):
        col2 = col if col2 is None else col2
        n = _unit(n)
        for tri in ((p0, p1, p2), (p0, p2, p3)):
            self.tris.append((np.array(tri[0], float), np.array(tri[1], float),
                              np.array(tri[2], float), n, col, col2, tile, emit,
                              self.group))

    def box(self, x0, y0, z0, x1, y1, z1, col, solid=True, emit=0.0):
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        if z1 < z0: z0, z1 = z1, z0
        q = self.quad
        q((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0), col, emit=emit)
        q((x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (-1, 0, 0), col, emit=emit)
        q((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (0, 1, 0), col, emit=emit)
        q((x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1), (0, 0, 1), col, emit=emit)
        q((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (0, 0, -1), col, emit=emit)
        if y0 > 0.02:
            q((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0), col, emit=emit)
        if solid and y0 < 1.2:
            self.colliders.append((x0, z0, x1, z1))

    def floor(self, x0, z0, x1, z1, col, col2, tile, y=0.0):
        self.floor_quads.append((x0, z0, x1, z1, col, col2, tile, y))
        self.quad((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0),
                  (0, 1, 0), col, col2, tile)

    def ceiling(self, x0, z0, x1, z1, col=(0.97, 0.97, 0.98), lights=True):
        old, self.group = self.group, 1
        self.quad((x0, WALL_H, z0), (x1, WALL_H, z0), (x1, WALL_H, z1), (x0, WALL_H, z1),
                  (0, -1, 0), col)
        if lights:   # emissive light panels
            nx = max(1, int((x1 - x0) // 3.4))
            nz = max(1, int((z1 - z0) // 3.4))
            for i in range(nx):
                for j in range(nz):
                    cx = x0 + (i + 0.5) * (x1 - x0) / nx
                    cz = z0 + (j + 0.5) * (z1 - z0) / nz
                    self.quad((cx - 0.6, WALL_H - 0.01, cz - 0.3), (cx + 0.6, WALL_H - 0.01, cz - 0.3),
                              (cx + 0.6, WALL_H - 0.01, cz + 0.3), (cx - 0.6, WALL_H - 0.01, cz + 0.3),
                              (0, -1, 0), (1, 1, 0.95), emit=1.0)
        self.group = old

    # -- walls --------------------------------------------------------------
    def wall_x(self, z, xa, xb, doors=(), t=WALL_T):
        """Wall running along X at depth z, with door gaps [(x_centre, width)]."""
        cur = xa
        for xc, w in sorted(doors):
            d0, d1 = xc - w / 2, xc + w / 2
            self.box(cur, 0, z - t / 2, d0, WALL_H, z + t / 2, WALL)
            self.box(d0, DOOR_H, z - t / 2, d1, WALL_H, z + t / 2, WALL, solid=False)
            cur = d1
        self.box(cur, 0, z - t / 2, xb, WALL_H, z + t / 2, WALL)

    def wall_z(self, x, za, zb, doors=(), t=WALL_T):
        cur = za
        for zc, w in sorted(doors):
            d0, d1 = zc - w / 2, zc + w / 2
            self.box(x - t / 2, 0, cur, x + t / 2, WALL_H, d0, WALL)
            self.box(x - t / 2, DOOR_H, d0, x + t / 2, WALL_H, d1, WALL, solid=False)
            cur = d1
        self.box(x - t / 2, 0, cur, x + t / 2, WALL_H, zb, WALL)

    # -- finalise -------------------------------------------------------------
    def build(self):
        """Convert the Python triangle list into NumPy arrays (done once)."""
        n = len(self.tris)
        V = np.zeros((n, 3, 3)); N = np.zeros((n, 3)); C1 = np.zeros((n, 3)); C2 = np.zeros((n, 3))
        T = np.zeros(n); E = np.zeros(n); G = np.zeros(n, int)
        for i, (a, b, c, nn, c1, c2, tile, emit, grp) in enumerate(self.tris):
            V[i, 0], V[i, 1], V[i, 2] = a, b, c
            N[i] = nn; C1[i] = c1; C2[i] = c2; T[i] = tile; E[i] = emit; G[i] = grp
        # static lighting (two directional lights + ambient); emissive = unlit
        shade = 0.56 + 0.34 * np.clip(N @ L1, 0, 1) + 0.18 * np.clip(N @ L2, 0, 1) + 0.05 * N[:, 1]
        shade = np.where(G == 1, 0.92, shade)
        shade = np.where(E > 0, 1.12, shade)
        self._arrays = dict(V=V, N=N, C1=np.clip(C1 * shade[:, None], 0, 1),
                            C2=np.clip(C2 * shade[:, None], 0, 1), T=T, G=G)
        return self._arrays

    @property
    def arrays(self):
        return self._arrays if self._arrays is not None else self.build()

    def room_at(self, x, z, level=0):
        best = "Outside"
        rooms = self.levels[level].rooms if hasattr(self, "levels") else self.rooms
        for name, x0, z0, x1, z1 in rooms:
            if x0 <= x <= x1 and z0 <= z <= z1:
                best = name
        return best


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self, x=2.0, z=12.0, yaw=math.pi / 2, pitch=0.0, fov=72.0):
        self.x, self.z = x, z
        self.y = EYE_H
        self.level = 0
        self.overview = False
        self.yaw, self.pitch, self.fov = yaw, pitch, fov

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    def basis(self):
        sy, cy, sp, cp = math.sin(self.yaw), math.cos(self.yaw), math.sin(self.pitch), math.cos(self.pitch)
        f = np.array([sy * cp, sp, cy * cp])
        r = np.array([cy, 0.0, -sy])
        u = np.array([-sp * sy, cp, -sp * cy])
        return r, u, f

    def move(self, scene, forward, strafe, dt_dist=1.0):
        """Walk with collision detection (axis-separated sliding)."""
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        dx = (sy * forward + cy * strafe) * dt_dist
        dz = (cy * forward - sy * strafe) * dt_dist
        # Substeps prevent tunnelling through thin walls on slower frames.
        steps = max(1, math.ceil(math.hypot(dx, dz) / (PLAYER_R * 0.45)))
        for _ in range(steps):
            nx = self.x + dx / steps
            if not collides(scene, nx, self.z, level=self.level):
                self.x = nx
            nz = self.z + dz / steps
            if not collides(scene, self.x, nz, level=self.level):
                self.z = nz


def collides(scene, x, z, r=PLAYER_R, level=0):
    if x < 0.45 or x > 39.55 or z < 0.45 or z > 23.55:
        return True
    colliders = scene.levels[level].colliders if hasattr(scene, "levels") else scene.colliders
    for x0, z0, x1, z1 in colliders:
        cx = min(max(x, x0), x1)
        cz = min(max(z, z0), z1)
        if (x - cx) ** 2 + (z - cz) ** 2 < r * r:
            return True
    return False


# ----------------------------------------------------------------------------
# Renderer: perspective projection + Z-buffer
# ----------------------------------------------------------------------------
def render(scene, cam, W=480, H=300, use_zbuffer=True, show_ceiling=True,
           mode="normal", fog_strength=0.28):
    """Render one frame.

    mode: 'normal' | 'depth' (visualise the Z-buffer) | 'wire' (hidden-line wireframe)
    use_zbuffer=False draws triangles in model order (painter without sorting)
    to demonstrate the hidden-surface errors that the Z-buffer removes.
    Returns (frame uint8 HxWx3, invz float32 HxW).
    """
    A = scene.arrays
    V, Nn, C1, C2, T, G = A["V"], A["N"], A["C1"], A["C2"], A["T"], A["G"]
    if "level" in A and not cam.overview:
        active = A["level"] == cam.level
        V, Nn, C1, C2, T, G = (a[active] for a in (V, Nn, C1, C2, T, G))
    r, u, f = cam.basis()
    eye = cam.pos
    fpx = (W / 2.0) / math.tan(math.radians(cam.fov) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    # --- 1. view transform (all triangles at once) -------------------------
    d = V - eye                                   # (N,3,3)
    xc, yc, zc = d @ r, d @ u, d @ f              # camera-space coordinates
    # --- 2. culling ----------------------------------------------------------
    facing = np.einsum("ij,ij->i", Nn, eye - V[:, 0]) > 0       # back-face culling
    keep = facing & (zc.max(axis=1) > NEAR)
    # Reject triangles wholly outside a side of the camera frustum.
    hx, hy = W / (2 * fpx), H / (2 * fpx)
    keep &= ~((xc < -zc * hx).all(1) | (xc > zc * hx).all(1) |
              (yc < -zc * hy).all(1) | (yc > zc * hy).all(1))
    if not show_ceiling:
        keep &= G == 0
    idx = np.nonzero(keep)[0]

    zbuf = np.zeros((H, W), np.float32)           # stores 1/z  (0 = infinitely far)
    img = np.empty((H, W, 3), np.float32)
    horizon = np.linspace(0.78, 1.0, H, dtype=np.float32)[:, None, None]
    img[:] = SKY[None, None, :] * horizon if mode != "wire" else 0.0
    xs_grid = np.arange(W, dtype=np.float32) + 0.5
    ys_grid = np.arange(H, dtype=np.float32) + 0.5
    drawn = []                                    # projected triangles (for wireframe)

    for i in idx:
        poly = [(xc[i, k], yc[i, k], zc[i, k], V[i, k]) for k in range(3)]
        # --- 3. near-plane clipping (Sutherland-Hodgman, one plane) ------------
        if min(p[2] for p in poly) < NEAR:
            out = []
            for a in range(3):
                p, q = poly[a], poly[(a + 1) % 3]
                pin, qin = p[2] >= NEAR, q[2] >= NEAR
                if pin:
                    out.append(p)
                if pin != qin:
                    t = (NEAR - p[2]) / (q[2] - p[2])
                    out.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(3)) +
                               (p[3] + t * (q[3] - p[3]),))
            poly = out
            if len(poly) < 3:
                continue
        for k in range(1, len(poly) - 1):                 # triangulate the polygon
            tri = (poly[0], poly[k], poly[k + 1])
            # --- 4. perspective projection --------------------------------------
            sx = [cx + fpx * p[0] / p[2] for p in tri]
            sy = [cy - fpx * p[1] / p[2] for p in tri]
            iz = [1.0 / p[2] for p in tri]
            x0 = max(int(math.floor(min(sx))), 0); x1 = min(int(math.ceil(max(sx))), W)
            y0 = max(int(math.floor(min(sy))), 0); y1 = min(int(math.ceil(max(sy))), H)
            if x0 >= x1 or y0 >= y1:
                continue
            den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
            if abs(den) < 1e-9:
                continue
            # --- 5. rasterise: barycentric coordinates for every pixel in bbox --
            X = xs_grid[x0:x1][None, :]
            Y = ys_grid[y0:y1][:, None]
            l0 = ((sy[1] - sy[2]) * (X - sx[2]) + (sx[2] - sx[1]) * (Y - sy[2])) / den
            l1 = ((sy[2] - sy[0]) * (X - sx[2]) + (sx[0] - sx[2]) * (Y - sy[2])) / den
            l2 = 1.0 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            invz = l0 * iz[0] + l1 * iz[1] + l2 * iz[2]   # 1/z is linear in screen space
            # --- 6. Z-buffer test --------------------------------------------------
            sub = zbuf[y0:y1, x0:x1]
            vis = inside & (invz > sub) if use_zbuffer else inside
            if not vis.any():
                continue
            sub[vis] = invz[vis]
            if mode == "wire":
                drawn.append((sx, sy, iz, C1[i]))
                continue
            zv = 1.0 / invz[vis]
            if T[i] > 0:                                   # checker-tiled floor
                w0, w1, w2 = l0[vis] * iz[0], l1[vis] * iz[1], l2[vis] * iz[2]
                ws = w0 + w1 + w2
                wx = (w0 * tri[0][3][0] + w1 * tri[1][3][0] + w2 * tri[2][3][0]) / ws
                wz = (w0 * tri[0][3][2] + w1 * tri[1][3][2] + w2 * tri[2][3][2]) / ws
                par = (np.floor(wx / T[i]) + np.floor(wz / T[i])) % 2 == 0
                col = np.where(par[:, None], C1[i], C2[i])
                # Fine grout lines are perspective-correct, just like the tile pattern.
                fx, fz = np.mod(wx, T[i]), np.mod(wz, T[i])
                grout = (fx < 0.012) | (fz < 0.012)
                col = col * np.where(grout, 0.84, 1.0)[:, None]
            else:
                col = np.broadcast_to(C1[i], (zv.size, 3))
            fog = np.clip(zv / FAR, 0, 1)[:, None] * fog_strength
            col = col * (1 - fog) + FOG * fog
            img[y0:y1, x0:x1][vis] = col

    if mode == "wire":                                     # hidden-line wireframe
        for sx, sy, iz, colr in drawn:
            for a in range(3):
                b = (a + 1) % 3
                n = int(max(abs(sx[b] - sx[a]), abs(sy[b] - sy[a]))) + 1
                t = np.linspace(0, 1, n)
                px = (sx[a] + t * (sx[b] - sx[a])).astype(int)
                py = (sy[a] + t * (sy[b] - sy[a])).astype(int)
                pz = iz[a] + t * (iz[b] - iz[a])
                ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                px, py, pz = px[ok], py[ok], pz[ok]
                ok = pz >= zbuf[py, px] * 0.985
                img[py[ok], px[ok]] = np.clip(colr * 1.4 + 0.15, 0, 1)
    if mode == "depth":
        return depth_image(zbuf), zbuf
    if mode == "normal":
        # Gentle vignette; diagnostic depth/wire views retain unmodified values.
        yy, xx = np.ogrid[-1:1:complex(H), -1:1:complex(W)]
        img *= (1.0 - 0.055 * np.clip(xx * xx + yy * yy, 0, 1.6))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), zbuf


def depth_image(zbuf, far=28.0):
    """Colour-map the Z-buffer (near = bright yellow, far = deep blue)."""
    z = np.where(zbuf > 0, 1.0 / np.maximum(zbuf, 1e-6), far)
    t = np.clip(z / far, 0, 1) ** 0.6
    stops = np.array([[1.0, 0.93, 0.35], [0.95, 0.45, 0.25], [0.45, 0.20, 0.55], [0.04, 0.07, 0.25]])
    pos = np.array([0.0, 0.25, 0.6, 1.0])
    out = np.stack([np.interp(t, pos, stops[:, c]) for c in range(3)], axis=-1)
    return (out * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Mini-map (pure NumPy overlay)
# ----------------------------------------------------------------------------
def minimap(scene, cam, scale=5):
    """Top-down floor plan (40 x 24 m) with the player and view direction."""
    w, h = 40 * scale, 24 * scale
    m = np.zeros((h, w, 3), np.uint8)
    m[:] = (222, 237, 236)
    local = scene.levels[cam.level] if hasattr(scene, "levels") else scene
    for x0, z0, x1, z1 in local.colliders:
        a, b = int(x0 * scale), int(z0 * scale)
        c, d = max(int(x1 * scale), a + 1), max(int(z1 * scale), b + 1)
        m[b:d, a:c] = (50, 62, 84) if (x1 - x0 < 0.3 or z1 - z0 < 0.3) else (150, 165, 190)
    px, pz = int(cam.x * scale), int(cam.z * scale)
    for k in range(1, 7 * scale // 2):                       # view direction
        qx, qz = int(px + math.sin(cam.yaw) * k), int(pz + math.cos(cam.yaw) * k)
        if 0 <= qx < w and 0 <= qz < h:
            m[qz, qx] = (230, 60, 60)
    m[max(pz - 2, 0):pz + 3, max(px - 2, 0):px + 3] = (230, 40, 40)
    out = np.pad(m, ((2, 2), (2, 2), (0, 0)), constant_values=30)
    return out


def overlay(frame, patch, corner="br", margin=6):
    H, W = frame.shape[:2]
    h, w = patch.shape[:2]
    if corner == "br":
        y, x = H - h - margin, W - w - margin
    else:
        y, x = margin, margin
    if y < 0 or x < 0:
        return frame
    frame = frame.copy()
    frame[y:y + h, x:x + w] = patch
    return frame

# ============================================================================
# 2. HOSPITAL - floor plan, rooms and furniture
# ============================================================================

T = WALL_T / 2
DOOR_W = 1.4
ROOM_X = [12, 19, 26, 33, 40]
DOOR_XC = [15.5, 22.5, 29.5, 36.5]

SOUTH = [("Ward A", BLUE), ("Ward B", BLUE), ("Intensive Care (ICU)", RED), ("Operating Theatre", GREEN)]
NORTH = [("Pharmacy", GREEN), ("Consultation", (0.95, 0.60, 0.15)),
         ("Radiology", (0.55, 0.35, 0.75)), ("Laboratory", (0.90, 0.78, 0.15))]


# ---------------------------------------------------------------- furniture
def bed(s, cx, z0, head_south=True, blanket=BLUE):
    """Hospital bed, 1.0 x 2.0 m, head at z0 (against the wall)."""
    x0, x1 = cx - 0.5, cx + 0.5
    s.box(x0, 0.25, z0, x1, 0.45, z0 + 2.0, STEEL)                  # frame
    s.box(x0, 0.45, z0 + 0.05, x1, 0.62, z0 + 1.95, WHITE, solid=False)   # mattress
    s.box(x0 + 0.04, 0.62, z0 + 0.8, x1 - 0.04, 0.68, z0 + 1.9, blanket, solid=False)  # blanket
    s.box(x0 + 0.12, 0.62, z0 + 0.1, x1 - 0.12, 0.72, z0 + 0.5, (0.98, 0.98, 1.0), solid=False)  # pillow
    s.box(x0, 0.25, z0, x1, 1.05, z0 + 0.06, STEEL)                 # headboard
    s.box(x0, 0.25, z0 + 1.94, x1, 0.8, z0 + 2.0, STEEL)            # footboard


def cabinet(s, x0, z0, x1, z1, h=0.75, col=WHITE):
    s.box(x0, 0, z0, x1, h, z1, col)


def chair(s, x, z, back="s", col=(0.25, 0.45, 0.75)):
    s.box(x - 0.22, 0.40, z - 0.22, x + 0.22, 0.46, z + 0.22, col)
    s.box(x - 0.04, 0.0, z - 0.04, x + 0.04, 0.40, z + 0.04, DARK, solid=False)
    bx = {"s": (x - 0.22, 0.46, z - 0.24, x + 0.22, 0.90, z - 0.20),
          "n": (x - 0.22, 0.46, z + 0.20, x + 0.22, 0.90, z + 0.24),
          "w": (x - 0.24, 0.46, z - 0.22, x - 0.20, 0.90, z + 0.22),
          "e": (x + 0.20, 0.46, z - 0.22, x + 0.24, 0.90, z + 0.22)}[back]
    s.box(*bx, col, solid=False)
    s.colliders.append((x - 0.25, z - 0.25, x + 0.25, z + 0.25))


def plant(s, x, z):
    s.box(x - 0.2, 0, z - 0.2, x + 0.2, 0.4, z + 0.2, (0.70, 0.40, 0.28))
    s.box(x - 0.3, 0.4, z - 0.3, x + 0.3, 0.95, z + 0.3, LEAF, solid=False)
    s.box(x - 0.2, 0.95, z - 0.2, x + 0.2, 1.35, z + 0.2, (0.22, 0.62, 0.30), solid=False)


def monitor_stand(s, x, z):
    s.box(x - 0.2, 0, z - 0.2, x + 0.2, 0.1, z + 0.2, DARK, solid=False)
    s.box(x - 0.03, 0.1, z - 0.03, x + 0.03, 1.3, z + 0.03, STEEL, solid=False)
    s.box(x - 0.3, 1.3, z - 0.05, x + 0.3, 1.75, z + 0.05, DARK, solid=False)
    s.box(x - 0.26, 1.34, z - 0.07, x + 0.26, 1.71, z - 0.05, (0.2, 0.85, 0.4), solid=False, emit=0.6)
    s.colliders.append((x - 0.25, z - 0.25, x + 0.25, z + 0.25))


def desk(s, x0, z0, x1, z1, col=WOOD):
    s.box(x0, 0.70, z0, x1, 0.76, z1, col)
    s.box(x0 + 0.05, 0, z0 + 0.05, x0 + 0.12, 0.70, z1 - 0.05, DARK, solid=False)
    s.box(x1 - 0.12, 0, z0 + 0.05, x1 - 0.05, 0.70, z1 - 0.05, DARK, solid=False)


def window(s, x, z, axis, width=2.0, inward=1):
    """Sky-blue window panel on an exterior wall. axis='z': panel lies on a z-plane."""
    emit = 0.85
    if axis == "z":
        z0 = z if inward > 0 else z - 0.03
        s.box(x - width / 2, 1.0, z0, x + width / 2, 2.3, z0 + 0.03, (0.62, 0.84, 0.98), solid=False, emit=emit)
    else:
        x0 = x if inward > 0 else x - 0.03
        s.box(x0, 1.0, z - width / 2, x0 + 0.03, 2.3, z + width / 2, (0.62, 0.84, 0.98), solid=False, emit=emit)


def sign(s, x, z, axis, col, w=1.5):
    """Department sign above a door."""
    if axis == "z":
        s.box(x - w / 2, 2.4, z, x + w / 2, 2.8, z + 0.05 if z > 0 else z - 0.05, col, solid=False, emit=0.5)
    else:
        s.box(x, 2.4, z - w / 2, x + 0.05, 2.8, z + w / 2, col, solid=False, emit=0.5)


# ------------------------------------------------------------------- build
def build_ground():
    s = Scene()
    W, D = 40.0, 24.0

    # ---- floors (checker tiles) -------------------------------------------
    s.floor(0, 0, 12, D, (0.88, 0.85, 0.80), (0.80, 0.76, 0.70), 1.0)          # lobby
    s.floor(12, 10, W, 14, (0.82, 0.88, 0.92), (0.74, 0.82, 0.88), 0.8)       # corridor
    room_floor = [(0.80, 0.90, 0.96), (0.80, 0.90, 0.96), (0.96, 0.86, 0.86), (0.76, 0.90, 0.82)]
    for i in range(4):
        s.floor(ROOM_X[i], 0, ROOM_X[i + 1], 10, room_floor[i], tuple(c * 0.91 for c in room_floor[i]), 0.7)
    north_floor = [(0.82, 0.93, 0.84), (0.98, 0.90, 0.78), (0.88, 0.84, 0.95), (0.97, 0.95, 0.76)]
    for i in range(4):
        s.floor(ROOM_X[i], 14, ROOM_X[i + 1], D, north_floor[i], tuple(c * 0.91 for c in north_floor[i]), 0.7)
    # wayfinding stripes along corridor floor
    s.quad((12, 0.012, 11.0), (12, 0.012, 11.15), (W, 0.012, 11.15), (W, 0.012, 11.0), (0, 1, 0), GREEN)
    s.quad((12, 0.012, 12.85), (12, 0.012, 13.0), (W, 0.012, 13.0), (W, 0.012, 12.85), (0, 1, 0), BLUE)

    # ---- outer walls ---------------------------------------------------------
    s.wall_x(0, -T, W + T)                                  # south
    s.wall_x(D, -T, W + T)                                  # north
    s.wall_z(W, -T, D + T)                                  # east
    s.wall_z(0, -T, D + T, doors=[(12.0, 3.2)])             # west with main entrance

    # ---- lobby / corridor boundary (x = 12) -------------------------------------
    s.wall_z(12, 0, 10)
    s.wall_z(12, 14, D)
    # ---- corridor walls with doors ----------------------------------------------------
    s.wall_x(10, 12, W, doors=[(x, DOOR_W) for x in DOOR_XC])
    s.wall_x(14, 12, W, doors=[(x, DOOR_W) for x in DOOR_XC])
    # ---- room partitions -------------------------------------------------------------------
    for x in ROOM_X[1:4]:
        s.wall_z(x, 0, 10)
        s.wall_z(x, 14, D)

    # ---- ceilings --------------------------------------------------------------------------
    s.ceiling(0, 0, 12, D)
    s.ceiling(12, 10, W, 14)
    for i in range(4):
        s.ceiling(ROOM_X[i], 0, ROOM_X[i + 1], 10)
        s.ceiling(ROOM_X[i], 14, ROOM_X[i + 1], D)

    # ---- department signs (corridor side) & windows ---------------------------------------------
    for i, xc in enumerate(DOOR_XC):
        sign(s, xc, 10 + T, "z", SOUTH[i][1])
        sign(s, xc, 14 - T, "z", NORTH[i][1])
        window(s, xc, T, "z")                                   # south exterior wall
        window(s, xc, D - T - 0.03, "z")                        # north exterior wall
    window(s, W - T - 0.03, 5.0, "x", 3.0); window(s, W - T - 0.03, 19.0, "x", 3.0)
    for xw in (3.5, 8.5):
        window(s, xw, T, "z", 3.0); window(s, xw, D - T - 0.03, "z", 3.0)
    sign(s, T, 12.0, "x", GREEN, 3.0)                               # ENTRANCE sign (inside)
    # big red cross on lobby wall
    s.box(11.80, 1.55, 18.7, 11.9, 3.05, 20.3, WHITE, solid=False)
    s.box(11.76, 1.95, 19.3, 11.80, 2.65, 19.7, RED, solid=False, emit=0.3)
    s.box(11.76, 2.15, 18.95, 11.80, 2.45, 20.05, RED, solid=False, emit=0.3)

    # ====================== LOBBY: reception + waiting area ============================
    desk(s, 5.2, 16.5, 6.4, 21.5, (0.85, 0.87, 0.90))                # reception counter
    s.box(5.2, 0.76, 16.5, 5.6, 1.15, 21.5, TEAL)                      # counter front panel
    s.box(5.2, 1.15, 16.5, 6.4, 1.2, 21.5, (0.20, 0.22, 0.26), solid=False)
    for zz in (17.5, 20.0):                                             # PCs behind counter
        s.box(6.6, 0.76, zz - 0.3, 6.65, 1.2, zz + 0.3, DARK, solid=False)
        s.box(6.62, 0.78, zz - 0.28, 6.64, 1.18, zz + 0.28, (0.5, 0.8, 1.0), solid=False, emit=0.5)
    for row, xr in enumerate((2.5, 4.3, 6.1)):                          # waiting chairs
        for k in range(5):
            chair(s, xr, 1.8 + k * 0.8, "w", (0.25, 0.45, 0.75) if row % 2 == 0 else (0.30, 0.65, 0.55))
    s.box(8.2, 0.0, 3.0, 9.2, 0.42, 5.0, WOOD)                           # coffee table
    for x, z in ((0.9, 0.9), (0.9, 23.1), (11.0, 0.9), (11.0, 23.1), (0.9, 8.5), (0.9, 15.5)):
        plant(s, x, z)
    s.colliders += [(0.7, 0.7, 1.1, 1.1)]

    # ====================== corridor furniture ==============================================
    for xb in (17.5, 24.5, 31.5):
        for zb, side in ((10.2, 1), (13.4, -1)):
            s.box(xb, 0.0, zb, xb + 2.2, 0.42, zb + 0.4, (0.30, 0.55, 0.80))
            s.box(xb, 0.42, zb + (0.34 if side == 1 else 0.0), xb + 2.2, 0.85,
                  zb + (0.4 if side == 1 else 0.06), (0.30, 0.55, 0.80), solid=False)
    s.box(38.6, 0, 10.15, 38.95, 0.8, 10.45, RED)                       # fire extinguisher cabinet
    plant(s, 39.2, 13.4)

    # ====================== SOUTH ROOMS ===================================================
    # Ward A  (x 12..19)
    for bx in (13.4, 15.5, 17.6):
        bed(s, bx, 0.1, blanket=BLUE)
    for bx in (14.45, 16.55):
        cabinet(s, bx - 0.2, 0.1, bx + 0.2, 0.5, 0.65)
    desk(s, 17.2, 5.5, 18.8, 6.4); chair(s, 18.0, 6.9, "n")
    # Ward B  (x 19..26)
    for bx in (20.4, 22.5, 24.6):
        bed(s, bx, 0.1, blanket=(0.35, 0.72, 0.62))
    for bx in (21.45, 23.55):
        cabinet(s, bx - 0.2, 0.1, bx + 0.2, 0.5, 0.65)
    s.box(19.1, 0, 6.0, 19.7, 0.9, 8.4, (0.85, 0.87, 0.90))              # supply trolley / wardrobe
    # ICU (x 26..33)
    for bx in (28.0, 31.0):
        bed(s, bx, 0.1, blanket=(0.85, 0.50, 0.50))
        monitor_stand(s, bx + 0.9, 0.6)
    s.box(27.6, 0, 4.4, 31.4, 0.95, 5.2, (0.85, 0.87, 0.90))            # nurses' station
    s.box(27.6, 0.95, 4.4, 31.4, 1.0, 5.2, TEAL, solid=False)
    # Operating theatre (x 33..40)
    s.box(35.9, 0.0, 3.3, 37.1, 0.12, 5.9, DARK)                          # table base
    s.box(36.0, 0.12, 3.5, 37.0, 0.85, 5.7, STEEL)                         # pedestal block
    s.box(35.8, 0.85, 3.3, 37.2, 0.95, 6.0, (0.60, 0.85, 0.80))           # table top
    s.box(36.44, 2.55, 4.55, 36.56, WALL_H, 4.65, STEEL, solid=False)     # lamp mast
    s.box(35.7, 2.35, 4.0, 37.3, 2.55, 5.2, STEEL, solid=False)           # lamp disc
    s.box(35.8, 2.31, 4.1, 37.2, 2.35, 5.1, (1.0, 1.0, 0.85), solid=False, emit=1.0)
    desk(s, 38.6, 3.0, 39.8, 5.2, STEEL)                                   # instrument table
    monitor_stand(s, 34.3, 3.4); monitor_stand(s, 34.3, 6.0)
    cabinet(s, 33.15, 8.8, 35.5, 9.8, 1.8, (0.90, 0.92, 0.95))

    # ====================== NORTH ROOMS ===================================================
    # Pharmacy (x 12..19)
    s.box(12.2, 0, 23.3, 18.8, 2.4, 23.8, (0.88, 0.90, 0.94))             # shelf unit
    colours = [RED, BLUE, GREEN, (0.95, 0.75, 0.2), (0.7, 0.4, 0.8)]
    for row in range(4):
        for k in range(14):
            s.box(12.3 + k * 0.46, 0.3 + row * 0.55, 23.0, 12.6 + k * 0.46, 0.6 + row * 0.55, 23.3,
                  colours[(row + k) % 5], solid=False)
    s.box(13.0, 0, 19.6, 18.0, 1.05, 20.4, (0.86, 0.90, 0.88))             # service counter
    s.box(13.0, 1.05, 19.6, 18.0, 1.1, 20.4, GREEN, solid=False)
    # Consultation (x 19..26)
    desk(s, 21.0, 19.0, 23.6, 20.2)
    chair(s, 22.3, 20.8, "n", (0.4, 0.4, 0.45)); chair(s, 21.7, 18.2, "s"); chair(s, 22.9, 18.2, "s")
    s.box(19.2, 0.0, 15.2, 20.0, 0.55, 17.6, (0.75, 0.88, 0.95))          # examination couch
    cabinet(s, 24.8, 22.8, 25.9, 23.8, 1.9, (0.88, 0.9, 0.94))
    # Radiology (x 26..33)  - CT/X-ray gantry with bed and console
    gx = 29.5
    s.box(gx - 1.0, 0.0, 19.3, gx - 0.6, 2.2, 20.5, WHITE)                 # CT ring: left
    s.box(gx + 0.6, 0.0, 19.3, gx + 1.0, 2.2, 20.5, WHITE)                  # right
    s.box(gx - 1.0, 1.9, 19.3, gx + 1.0, 2.2, 20.5, WHITE, solid=False)    # top
    s.box(gx - 0.6, 0.0, 19.3, gx + 0.6, 0.6, 20.5, (0.80, 0.82, 0.88), solid=False)  # bottom
    s.box(gx - 0.3, 0.7, 16.2, gx + 0.3, 0.85, 22.3, STEEL, solid=False)  # patient bed through gantry
    s.box(gx - 0.12, 0.0, 16.4, gx + 0.12, 0.7, 17.2, DARK, solid=False)
    desk(s, 26.4, 15.3, 28.4, 16.1, (0.85, 0.87, 0.9))
    monitor_stand(s, 27.4, 16.9)
    # Laboratory (x 33..40)
    for zb in (18.0, 20.6):
        s.box(34.5, 0, zb, 38.6, 0.9, zb + 0.9, (0.84, 0.88, 0.90))        # lab bench
        s.box(34.5, 0.9, zb, 38.6, 0.94, zb + 0.9, (0.25, 0.28, 0.34), solid=False)
        for k in range(3):
            xm = 35.2 + k * 1.2
            s.box(xm, 0.94, zb + 0.3, xm + 0.25, 1.25, zb + 0.6, DARK, solid=False)   # microscopes
    s.box(33.3, 0, 22.2, 34.5, 2.0, 23.8, (0.88, 0.92, 0.95))              # refrigerator
    s.box(36.0, 0, 22.8, 39.7, 2.3, 23.8, (0.88, 0.90, 0.94))              # storage shelves

    # ---- room registry (for HUD / navigation) ---------------------------------------------------
    s.rooms = [("Main Lobby & Reception", 0, 0, 12, D), ("Corridor", 12, 10, W, 14)]
    for i in range(4):
        s.rooms.append((SOUTH[i][0], ROOM_X[i], 0, ROOM_X[i + 1], 10))
        s.rooms.append((NORTH[i][0], ROOM_X[i], 14, ROOM_X[i + 1], D))
    s.build()
    return s


# Teleport targets (key 1..9, 0): name, x, z, yaw(deg)
WAYPOINTS = [
    ("Entrance", 1.6, 12.0, 90),
    ("Waiting Area", 9.5, 7.5, 235),
    ("Ward A", 15.5, 8.3, 180),
    ("Ward B", 22.5, 8.3, 180),
    ("ICU", 29.5, 8.5, 180),
    ("Operating Theatre", 36.5, 8.5, 180),
    ("Pharmacy", 15.5, 15.4, 0),
    ("Consultation", 22.5, 15.4, 0),
    ("Radiology", 29.5, 15.0, 0),
    ("Laboratory", 36.5, 15.4, 0),
]

# Guided tour (x, z) polyline
TOUR = [(1.6, 12.0), (14.0, 12.0), (22.5, 12.0), (22.5, 7.5), (22.5, 12.0), (29.5, 12.0), (29.5, 6.5),
        (29.5, 12.0), (36.5, 12.0), (36.5, 6.0), (36.5, 12.0), (36.5, 19.0), (36.5, 12.0),
        (29.5, 12.0), (29.5, 19.0), (29.5, 12.0), (15.5, 12.0), (15.5, 19.5)]

# ============================================================================
# TWO-FLOOR EXPANSION / architectural finishes
# ============================================================================
NAVY = (0.10, 0.20, 0.26)
SAGE = (0.31, 0.56, 0.51)
SAND = (0.83, 0.75, 0.62)
ACCENTS = (TEAL, SAGE)
UPPER_SOUTH = [("Recovery A", TEAL), ("Recovery B", TEAL),
               ("Physiotherapy", SAGE), ("Rehabilitation", SAGE)]
UPPER_NORTH = [("Family Lounge", SAND), ("Counselling", TEAL),
               ("Staff Lounge", SAGE), ("Training Room", BLUE)]
LIFT_X, LIFT_Z = 9.5, 17.1

# Small geometry font: signs remain visible in OBJ/GLB exports as well as the viewer.
GLYPHS = {
 'A': '01110/10001/10001/11111/10001/10001/10001',
 'B': '11110/10001/10001/11110/10001/10001/11110',
 'C': '01111/10000/10000/10000/10000/10000/01111',
 'D': '11110/10001/10001/10001/10001/10001/11110',
 'E': '11111/10000/10000/11110/10000/10000/11111',
 'F': '11111/10000/10000/11110/10000/10000/10000',
 'G': '01111/10000/10000/10111/10001/10001/01111',
 'H': '10001/10001/10001/11111/10001/10001/10001',
 'I': '111/010/010/010/010/010/111',
 'J': '00111/00010/00010/00010/10010/10010/01100',
 'K': '10001/10010/10100/11000/10100/10010/10001',
 'L': '10000/10000/10000/10000/10000/10000/11111',
 'M': '10001/11011/10101/10101/10001/10001/10001',
 'N': '10001/11001/10101/10011/10001/10001/10001',
 'O': '01110/10001/10001/10001/10001/10001/01110',
 'P': '11110/10001/10001/11110/10000/10000/10000',
 'Q': '01110/10001/10001/10001/10101/10010/01101',
 'R': '11110/10001/10001/11110/10100/10010/10001',
 'S': '01111/10000/10000/01110/00001/00001/11110',
 'T': '11111/00100/00100/00100/00100/00100/00100',
 'U': '10001/10001/10001/10001/10001/10001/01110',
 'V': '10001/10001/10001/10001/10001/01010/00100',
 'W': '10001/10001/10001/10101/10101/10101/01010',
 'X': '10001/10001/01010/00100/01010/10001/10001',
 'Y': '10001/10001/01010/00100/00100/00100/00100',
 'Z': '11111/00001/00010/00100/01000/10000/11111',
 '0': '01110/10001/10011/10101/11001/10001/01110',
 '1': '010/110/010/010/010/010/111',
 '2': '01110/10001/00001/00010/00100/01000/11111',
 '3': '11110/00001/00001/01110/00001/00001/11110',
 '4': '10010/10010/10010/11111/00010/00010/00010',
 '5': '11111/10000/10000/11110/00001/00001/11110',
 '6': '01110/10000/10000/11110/10001/10001/01110',
 '7': '11111/00001/00010/00100/01000/01000/01000',
 '8': '01110/10001/10001/01110/10001/10001/01110',
 '9': '01110/10001/10001/01111/00001/00001/01110',
 ' ': '000/000/000/000/000/000/000',
 '-': '000/000/000/111/000/000/000',
}


def lettering(s, label, x, y, z, width=2.4, height=0.22, facing=1, col=WHITE):
    """Centered, legible mesh lettering on an X/Y plane, viewed from +/-Z."""
    glyphs = [GLYPHS.get(c, GLYPHS[' ']).split('/') for c in label.upper()]
    columns = sum(len(g[0]) + 1 for g in glyphs)
    px = min(width / max(columns, 1), height / 7)
    cursor = -columns * px / 2
    for glyph in glyphs:
        for row, bits in enumerate(glyph):
            start = None
            for k, bit in enumerate(bits + '0'):
                if bit == '1' and start is None:
                    start = k
                elif bit == '0' and start is not None:
                    a, b = x - facing * (cursor + start * px), x - facing * (cursor + k * px)
                    lo, hi = y + (6 - row) * px, y + (7 - row) * px
                    s.quad((a, lo, z), (b, lo, z), (b, hi, z), (a, hi, z),
                           (0, 0, facing), col, emit=0.3)
                    start = None
        cursor += (len(glyph[0]) + 1) * px


def cylinder(s, x, z, radius, y0, y1, col, segments=10):
    for k in range(segments):
        a, b = 2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments
        p, q = (x + radius * math.cos(a), z + radius * math.sin(a)), (x + radius * math.cos(b), z + radius * math.sin(b))
        mid = (a + b) / 2
        s.quad((p[0], y0, p[1]), (q[0], y0, q[1]), (q[0], y1, q[1]), (p[0], y1, p[1]),
               (math.cos(mid), 0, math.sin(mid)), col)
        s.quad((x, y1, z), (p[0], y1, p[1]), (q[0], y1, q[1]), (x, y1, z), (0, 1, 0), col)


def plant(s, x, z):
    """Faceted foliage with a ceramic pot and a soft, concentric contact shadow."""
    for radius, colour in ((0.43, (0.76, 0.79, 0.76)), (0.32, (0.63, 0.68, 0.64))):
        cylinder(s, x, z, radius, 0.003, 0.005, colour, 12)
    cylinder(s, x, z, 0.23, 0.01, 0.43, (0.79, 0.78, 0.70))
    cylinder(s, x, z, 0.04, 0.43, 1.05, WOOD, 6)
    for dx, dz, cy, radius in ((0, 0, 1.13, 0.35), (-0.19, 0.08, 0.83, 0.24), (0.18, -0.11, 0.92, 0.25)):
        center = np.array([x + dx, cy, z + dz])
        top, bot = center + (0, radius, 0), center - (0, radius * 0.8, 0)
        for k in range(6):
            a, b = k * math.pi / 3, (k + 1) * math.pi / 3
            p = center + (radius * math.cos(a), 0, radius * math.sin(a))
            q = center + (radius * math.cos(b), 0, radius * math.sin(b))
            for peak in (top, bot):
                n = np.cross(q - p, peak - p)
                if np.dot(n, (p + q + peak) / 3 - center) < 0:
                    n = -n
                s.quad(p, q, peak, peak, n, (0.22, 0.45 + 0.025 * k, 0.33))
    s.colliders.append((x - 0.24, z - 0.24, x + 0.24, z + 0.24))


def window(s, x, z, axis, width=2.0, inward=1):
    """Inset daylight panels with white mullions and sill (opaque stylized glazing)."""
    if axis == 'z':
        s.box(x - width / 2 - 0.06, 0.94, z - 0.01, x + width / 2 + 0.06, 2.44, z + 0.055, WHITE, False)
        s.box(x - width / 2, 1.02, z - 0.015, x + width / 2, 2.36, z + 0.06, (0.65, 0.82, 0.86), False, 0.35)
        for xx in (x - 0.025,):
            s.box(xx, 1.02, z - 0.02, xx + 0.05, 2.36, z + 0.065, WHITE, False)
        s.box(x - width / 2 - 0.1, 0.91, z - 0.12, x + width / 2 + 0.1, 0.98, z + 0.15, WHITE, False)
    else:
        s.box(x - 0.01, 0.94, z - width / 2 - 0.06, x + 0.055, 2.44, z + width / 2 + 0.06, WHITE, False)
        s.box(x - 0.015, 1.02, z - width / 2, x + 0.06, 2.36, z + width / 2, (0.65, 0.82, 0.86), False, 0.35)
        s.box(x - 0.02, 1.02, z - 0.025, x + 0.065, 2.36, z + 0.025, WHITE, False)


def finish_floor(s, level):
    accent = ACCENTS[level]
    south = SOUTH if level == 0 else UPPER_SOUTH
    north = NORTH if level == 0 else UPPER_NORTH
    # Muted stone and porcelain replace high-contrast checkerboards.
    replacement = []
    for tri in s.tris:
        a, b, c, n, col, col2, tile, emit, grp = tri
        if tile:
            col = (0.84, 0.85, 0.81) if a[0] < 12 else (0.82, 0.87, 0.86)
            col2 = tuple(v * 0.97 for v in col)
        replacement.append((a, b, c, n, col, col2, tile, emit, grp))
    s.tris = replacement
    s.floor_quads = [(a, b, c, d, (0.84, 0.85, 0.81) if a < 12 else (0.82, 0.87, 0.86),
                      tuple(v * 0.97 for v in ((0.84, 0.85, 0.81) if a < 12 else (0.82, 0.87, 0.86))), t, y)
                     for a, b, c, d, _, _, t, y in s.floor_quads]
    for z, facing, departments in ((10.13, 1, south), (13.87, -1, north)):
        for i, xc in enumerate(DOOR_XC):
            left, right = ROOM_X[i] + 0.12, ROOM_X[i + 1] - 0.12
            # Wall protection panels, rail and skirting; leave the door opening clear.
            for xa, xb in ((left, xc - 0.77), (xc + 0.77, right)):
                s.box(xa, 0.08, z - 0.025, xb, 0.88, z + 0.025, (0.76, 0.83, 0.81), False)
                s.box(xa, 0.91, z - 0.045, xb, 0.98, z + 0.045, accent, False)
                s.box(xa, 0.02, z - 0.045, xb, 0.12, z + 0.045, NAVY, False)
            for xx in (xc - 0.77, xc + 0.70):
                s.box(xx, 0.02, z - 0.12, xx + 0.07, DOOR_H, z + 0.12, (0.65, 0.72, 0.71), False)
            s.box(xc - 0.77, DOOR_H, z - 0.12, xc + 0.77, DOOR_H + 0.07, z + 0.12, accent, False)
            s.box(xc - 1.43, 2.40, z - 0.045, xc + 1.43, 2.87, z + 0.045, NAVY, False)
            label = departments[i][0].replace('Intensive Care (ICU)', 'ICU').replace('Operating Theatre', 'Surgery')
            lettering(s, label, xc, 2.52, z + facing * 0.052, 2.62, 0.22, facing)
            # Small raised threshold with department colour.
            s.box(xc - 0.68, 0.004, z - 0.18, xc + 0.68, 0.009, z + 0.18, accent, False)
    # Long, recessed light strips in the circulation spine.
    for z in (10.55, 13.45):
        s.box(12.2, 3.08, z, 39.8, 3.13, z + 0.045, (0.93, 0.97, 0.89), False, 1)
    # Wall art and timber feature slats in the lobby.
    s.box(2.0, 1.0, 0.12, 8.6, 2.65, 0.16, NAVY, False)
    lettering(s, 'Aster Medical', 5.3, 2.03, 0.17, 5.8, 0.38)
    lettering(s, 'Care above all', 5.3, 1.48, 0.17, 4.5, 0.20, col=(0.67, 0.86, 0.80))
    for x in np.arange(8.8, 11.5, 0.22):
        s.box(x, 0.12, 0.12, x + 0.075, 2.95, 0.22, SAND, False)
    # Lift cab, open entrance, back-wall label and call panel.
    s.box(8.1, 0, 16, 8.25, WALL_H, 18.9, NAVY)
    s.box(10.75, 0, 16, 10.9, WALL_H, 18.9, NAVY)
    s.box(8.1, 0, 18.75, 10.9, WALL_H, 18.9, NAVY)
    s.box(8.1, 2.28, 15.95, 10.9, WALL_H, 16.1, NAVY, False)
    s.floor(8.25, 16, 10.75, 18.75, (0.53, 0.62, 0.64), (0.56, 0.65, 0.67), 0.5, 0.008)
    lettering(s, 'Lift  ' + ('G' if level == 0 else '1'), 9.5, 2.57, 15.94, 2.3, 0.22, -1)
    lettering(s, 'Press E', 9.5, 1.55, 18.73, 1.9, 0.18, -1)
    s.box(10.92, 1.03, 16.02, 11.12, 1.55, 16.13, STEEL, False)
    s.box(10.96, 1.20, 16.0, 11.08, 1.36, 16.025, accent, False, 1)
    s.rooms.append(('Lift', 8.1, 16, 10.9, 18.9))
    # Slabs make the second storey unambiguous in the exterior view and exports.
    s.box(-0.2, -0.19, -0.2, 40.2, -0.025, 24.2, (0.61, 0.68, 0.69), False)
    for z in (-0.13, 24.13):
        s.box(-0.1, 2.92, z - 0.06, 40.1, 3.18, z + 0.06, accent, False)
    s._arrays = None


def build_upper():
    s = Scene()
    s.floor(0, 0, 12, 24, WHITE, WHITE, 1)
    s.floor(12, 10, 40, 14, WHITE, WHITE, 0.8)
    s.wall_x(0, -T, 40 + T)
    s.wall_x(24, -T, 40 + T)
    s.wall_z(0, -T, 24 + T)
    s.wall_z(40, -T, 24 + T)
    s.wall_z(12, 0, 10)
    s.wall_z(12, 14, 24)
    for z in (10, 14):
        s.wall_x(z, 12, 40, [(x, DOOR_W) for x in DOOR_XC])
    for x in ROOM_X[1:4]:
        s.wall_z(x, 0, 10)
        s.wall_z(x, 14, 24)
    s.ceiling(0, 0, 12, 24)
    s.ceiling(12, 10, 40, 14)
    s.rooms = [('Recovery Lobby', 0, 0, 12, 24), ('Recovery Corridor', 12, 10, 40, 14)]
    for i in range(4):
        for za, zb, names in ((0, 10, UPPER_SOUTH), (14, 24, UPPER_NORTH)):
            s.floor(ROOM_X[i], za, ROOM_X[i + 1], zb, WHITE, WHITE, 0.8)
            s.ceiling(ROOM_X[i], za, ROOM_X[i + 1], zb)
            s.rooms.append((names[i][0], ROOM_X[i], za, ROOM_X[i + 1], zb))
        window(s, DOOR_XC[i], 0.13, 'z', 3.2)
        window(s, DOOR_XC[i], 23.82, 'z', 3.2)
    for z in (4, 8, 20):
        window(s, 0.13, z, 'x', 2.4)
    # Recovery rooms: fewer beds, bedside lighting and private seating.
    for origin in (12, 19):
        for bx in (origin + 1.5, origin + 5.0):
            bed(s, bx, 1.0, blanket=SAGE)
            cabinet(s, bx + 0.62, 1.0, bx + 1.15, 1.65, 0.65, SAND)
            cylinder(s, bx + 0.86, 1.3, 0.11, 0.65, 0.96, WHITE, 8)
            chair(s, bx, 5.0, 's', SAGE)
        plant(s, origin + 0.65, 8.9)
    # Physiotherapy: parallel bars, treatment couches and mats.
    for x in (27.1, 28.5):
        for z in (2.4, 6.5):
            s.box(x, 0, z, x + 0.07, 1.05, z + 0.07, STEEL)
        s.box(x, 1.02, 2.4, x + 0.08, 1.10, 6.57, STEEL, False)
    s.box(30, 0, 2, 31.8, 0.58, 4.5, (0.58, 0.76, 0.71))
    for x in (30.0, 31.2):
        s.box(x, 0.006, 5.3, x + 0.9, 0.025, 7.7, (0.41, 0.59, 0.67), False)
    # Rehabilitation: two exercise machines, balance blocks, storage.
    for x in (34.4, 37.8):
        s.box(x, 0, 2, x + 1.1, 0.18, 4.3, NAVY)
        s.box(x + 0.10, 0.18, 2.2, x + 1.0, 0.22, 4.1, DARK, False)
        for xx in (x, x + 1.03):
            s.box(xx, 0.2, 2, xx + 0.07, 1.3, 2.07, STEEL, False)
        s.box(x, 1.26, 2, x + 1.1, 1.44, 2.18, NAVY, False)
    cabinet(s, 38.9, 7.0, 39.8, 9.6, 1.6, SAND)
    # Family lounge and staff lounge: chairs facing coffee tables.
    for cx in (15.5, 29.5):
        desk(s, cx - 0.8, 19.4, cx + 0.8, 20.3, SAND)
        for x in (cx - 1.5, cx + 1.5):
            for z in (18.7, 21.0):
                chair(s, x, z, 'n' if z > 20 else 's', SAGE)
        plant(s, cx + 2.3, 22.8)
    desk(s, 21, 19.4, 23.6, 20.5, SAND)
    for x in (21.7, 23):
        chair(s, x, 18.5, 's', SAGE)
    chair(s, 22.3, 21.3, 'n', NAVY)
    # Training room with an unobstructed central aisle.
    for x in (34.4, 37.6):
        for z in (18, 20.5):
            desk(s, x, z, x + 1.3, z + 0.65, SAND)
            chair(s, x + 0.65, z + 1.2, 'n', BLUE)
    s.box(34.2, 1.25, 23.78, 38.8, 2.55, 23.87, NAVY, False)
    lettering(s, 'Learning Centre', 36.5, 1.85, 23.76, 3.8, 0.22, -1)
    # Upper reception, visitor lounge and greenery.
    desk(s, 4.0, 18.5, 6.5, 19.7, SAND)
    for x in (2.5, 5.1, 7.7):
        for z in (4, 6):
            chair(s, x, z, 's', SAGE)
    for x, z in ((1, 1), (11, 1), (1, 23), (11, 23), (3, 15)):
        plant(s, x, z)
    return s


def build_hospital():
    """Build both storeys in world space; keep navigation data local to each floor."""
    levels = [build_ground(), build_upper()]
    combined = Scene()
    combined.levels = levels
    triangle_levels = []
    for level, local in enumerate(levels):
        finish_floor(local, level)
        offset = np.array([0, level * FLOOR_HEIGHT, 0])
        for a, b, c, n, c1, c2, tile, emit, group in local.tris:
            combined.tris.append((a + offset, b + offset, c + offset, n, c1, c2, tile, emit, group))
            triangle_levels.append(level)
        combined.floor_quads.extend((*q[:7], q[7] + level * FLOOR_HEIGHT) for q in local.floor_quads)
    combined.colliders = levels[0].colliders
    combined.rooms = levels[0].rooms
    combined.build()['level'] = np.array(triangle_levels, dtype=np.int8)
    return combined


def set_floor(cam, level, at_lift=False):
    cam.level = int(level)
    cam.y = EYE_H + FLOOR_HEIGHT * cam.level
    cam.overview = False
    if at_lift:
        cam.x, cam.z, cam.yaw, cam.pitch = LIFT_X, LIFT_Z, math.pi, 0


def near_lift(cam):
    return 8.35 <= cam.x <= 10.65 and 14.4 <= cam.z <= 18.4


def floor_waypoints(level):
    if level == 0:
        return WAYPOINTS
    result = [('Recovery Lobby', 1.6, 12, 90), ('Visitor Lounge', 9.5, 7.5, 235)]
    result += [(name, DOOR_XC[i], 8.3, 180) for i, (name, _) in enumerate(UPPER_SOUTH)]
    result += [(name, DOOR_XC[i], 15.4, 0) for i, (name, _) in enumerate(UPPER_NORTH)]
    return result


def overview_camera():
    cam = Camera(-24, -30, math.atan2(44, 42), -0.32, 62)
    cam.y, cam.overview = 24.0, True
    return cam


# Camera tours use free aisles; the upper level is visited after the lift transfer.
TOUR = [(1.6, 12), (9.5, 12), (15.5, 12), (15.5, 7.5), (15.5, 12),
        (22.5, 12), (22.5, 7.5), (22.5, 12), (29.5, 12), (29.5, 6.5),
        (29.5, 12), (36.5, 12), (36.5, 7), (36.5, 12), (36.5, 16),
        (36.5, 12), (29.5, 12), (29.5, 17), (29.5, 12), (22.5, 12),
        (22.5, 17), (22.5, 12), (15.5, 12), (15.5, 17), (15.5, 12),
        (9.5, 12), (9.5, 17.1)]
UPPER_TOUR = [(9.5, 17.1), (9.5, 12), (15.5, 12), (15.5, 7), (15.5, 12),
              (22.5, 12), (22.5, 7), (22.5, 12), (29.5, 12), (29.5, 7),
              (29.5, 12), (36.5, 12), (36.5, 7), (36.5, 12), (36.5, 16),
              (36.5, 12), (29.5, 12), (29.5, 17), (29.5, 12), (22.5, 12),
              (22.5, 17), (22.5, 12), (15.5, 12), (15.5, 17), (15.5, 12), (9.5, 12)]


def tour_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def building_tour(cam, distance):
    ground_length = tour_length(TOUR)
    level = 0 if distance < ground_length else 1
    local_distance = distance if level == 0 else distance - ground_length
    x, z, yaw, _ = tour_state(local_distance, TOUR if level == 0 else UPPER_TOUR)
    set_floor(cam, level)
    cam.x, cam.z = x, z
    return yaw, ground_length + tour_length(UPPER_TOUR)


# ============================================================================
# 3. EXPORTER - OBJ / MTL / GLB
# ============================================================================

def collect(scene):
    """Return list of (v0,v1,v2, normal, rgb, emit, group) in export space (right-handed: z -> -z)."""
    out = []
    def add(a, b, c, n, col, emit, grp):
        a, b, c = [np.array([p[0], p[1], -p[2]], float) for p in (a, b, c)]
        n = np.array([n[0], n[1], -n[2]], float)
        if np.dot(np.cross(b - a, c - a), n) < 0:      # make winding agree with the outward normal
            b, c = c, b
        out.append((a, b, c, n, tuple(col), emit, grp))
    for a, b, c, n, col, col2, tile, emit, grp in scene.tris:
        if tile == 0:
            add(a, b, c, n, col, emit, grp)
    for x0, z0, x1, z1, col, col2, tile, y in scene.floor_quads:      # checker tiles as real geometry
        nx, nz = max(1, round((x1 - x0) / tile)), max(1, round((z1 - z0) / tile))
        xs = np.linspace(x0, x1, nx + 1); zs = np.linspace(z0, z1, nz + 1)
        for i in range(nx):
            for j in range(nz):
                cc = col if (i + j) % 2 == 0 else col2
                p = [(xs[i], y, zs[j]), (xs[i], y, zs[j + 1]), (xs[i + 1], y, zs[j + 1]), (xs[i + 1], y, zs[j])]
                add(p[0], p[1], p[2], (0, 1, 0), cc, 0, 0); add(p[0], p[2], p[3], (0, 1, 0), cc, 0, 0)
    return out


def write_obj(tris, path):
    base = os.path.splitext(path)[0]
    mats, verts, norms, vi, ni = {}, [], [], {}, {}
    faces = {}                                    # (group, material) -> list of faces
    def idx(table, lst, v):
        k = tuple(np.round(v, 4))
        if k not in table:
            table[k] = len(lst) + 1; lst.append(k)
        return table[k]
    for a, b, c, n, col, emit, grp in tris:
        m = "mat_%02x%02x%02x%s" % (*[int(round(min(max(x, 0), 1) * 255)) for x in col], "_light" if emit else "")
        mats[m] = (col, emit)
        f = [(idx(vi, verts, p), idx(ni, norms, n)) for p in (a, b, c)]
        faces.setdefault(("ceiling" if grp else "building", m), []).append(f)
    with open(path, "w") as f:
        f.write("# 3D Hospital - exported from the Python engine (metres, Y-up)\nmtllib %s\n" % os.path.basename(base + ".mtl"))
        for v in verts: f.write("v %.4f %.4f %.4f\n" % v)
        for n in norms: f.write("vn %.3f %.3f %.3f\n" % n)
        for (grp, m), fl in sorted(faces.items()):
            f.write("o %s\nusemtl %s\n" % (grp, m))
            for t in fl: f.write("f " + " ".join("%d//%d" % p for p in t) + "\n")
    with open(base + ".mtl", "w") as f:
        for m, (col, emit) in mats.items():
            f.write("newmtl %s\nKd %.3f %.3f %.3f\nKa %.3f %.3f %.3f\nKs 0.05 0.05 0.05\nNs 10\n" % (m, *col, *[c * 0.5 for c in col]))
            if emit: f.write("Ke %.3f %.3f %.3f\n" % tuple(min(c, 1) * 0.9 for c in col))
            f.write("\n")
    return len(tris), len(verts)


def write_glb(tris, path):
    P = np.array([[t[0], t[1], t[2]] for t in tris], np.float32).reshape(-1, 3)
    N = np.repeat(np.array([t[3] for t in tris], np.float32), 3, axis=0)
    C = np.repeat(np.array([np.clip(np.array(t[4]) * (1.0 if not t[5] else 1.0), 0, 1) for t in tris], np.float32), 3, axis=0)
    I = np.arange(len(P), dtype=np.uint32)
    blobs = [P.tobytes(), N.tobytes(), C.tobytes(), I.tobytes()]
    offs, o = [], 0
    for b in blobs: offs.append(o); o += len(b)
    bin_data = b"".join(blobs); bin_data += b"\0" * (-len(bin_data) % 4)
    js = {"asset": {"version": "2.0", "generator": "Hospital3D python engine"},
          "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "Hospital"}],
          "materials": [{"name": "vertex_colour", "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.85}}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
          "buffers": [{"byteLength": len(bin_data)}],
          "bufferViews": [{"buffer": 0, "byteOffset": offs[i], "byteLength": len(blobs[i]), "target": 34962 if i < 3 else 34963} for i in range(4)],
          "accessors": [
              {"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3", "min": P.min(0).tolist(), "max": P.max(0).tolist()},
              {"bufferView": 1, "componentType": 5126, "count": len(P), "type": "VEC3"},
              {"bufferView": 2, "componentType": 5126, "count": len(P), "type": "VEC3"},
              {"bufferView": 3, "componentType": 5125, "count": len(I), "type": "SCALAR"}]}
    jb = json.dumps(js, separators=(",", ":")).encode(); jb += b" " * (-len(jb) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(bin_data)))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A)); f.write(jb)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942)); f.write(bin_data)


def export_all(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    tris = collect(build_hospital())
    n, v = write_obj(tris, os.path.join(outdir, "hospital.obj"))
    write_glb(tris, os.path.join(outdir, "hospital.glb"))
    print("exported %d triangles (%d unique vertices) -> %s/hospital.obj, .mtl, .glb" % (n, v, outdir))



# ============================================================================
# 4. VIEWER - window, screenshots, tour, command line
# ============================================================================

# ----------------------------------------------------------------------------
def tour_state(dist, points=None):
    """Position and heading along the TOUR polyline after walking `dist` metres."""
    pts = TOUR if points is None else points
    seg_len = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg_len)
    dist = min(dist, total - 1e-6)
    for i, L in enumerate(seg_len):
        if dist <= L:
            t = dist / L
            x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
            z = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
            yaw = math.atan2(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            return x, z, yaw, total
        dist -= L
    return pts[-1][0], pts[-1][1], 0.0, total


def smooth_angle(cur, target, k):
    d = (target - cur + math.pi) % (2 * math.pi) - math.pi
    return cur + d * k


# ----------------------------------------------------------------------------
def camera_from_args(args):
    if args.overview:
        return overview_camera()
    cam = Camera(args.pos[0], args.pos[1], math.radians(args.yaw), math.radians(args.pitch), args.fov)
    set_floor(cam, args.floor)
    return cam


def save_image(path, frame):
    from pathlib import Path
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def run_shot(args):
    scene = build_hospital()
    cam = camera_from_args(args)
    frame, _ = render(scene, cam, *args.size, use_zbuffer=not args.noz,
                      show_ceiling=not args.noceiling, mode=args.mode)
    if args.minimap and not cam.overview:
        frame = overlay(frame, minimap(scene, cam, 4))
    save_image(args.shot, frame)
    print('Saved', args.shot)


def run_tour_gif(args):
    from pathlib import Path
    from PIL import Image, ImageDraw
    scene, cam = build_hospital(), Camera()
    total = tour_length(TOUR) + tour_length(UPPER_TOUR)
    frames, yaw, last_floor = [], None, -1
    for k in range(args.frames):
        ty, _ = building_tour(cam, k / max(1, args.frames - 1) * total)
        yaw = ty if yaw is None or last_floor != cam.level else smooth_angle(yaw, ty, 0.35)
        last_floor, cam.yaw = cam.level, yaw
        frame, _ = render(scene, cam, *args.size, mode=args.mode, use_zbuffer=not args.noz,
                          show_ceiling=not args.noceiling)
        image = Image.fromarray(overlay(frame, minimap(scene, cam, 2)))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 29), fill=(19, 37, 46))
        draw.text((10, 8), 'ASTER MEDICAL  /  ' + FLOOR_NAMES[cam.level], fill=(225, 245, 238))
        frames.append(image)
        if k % 20 == 0:
            print(f'Tour frame {k + 1}/{args.frames}', flush=True)
    Path(args.tour).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.tour, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print('Saved', args.tour)


def run_viewer(args):
    try:
        import pygame
    except ImportError:
        sys.exit('Install the viewer dependencies: python -m pip install -r requirements.txt')
    from pathlib import Path
    scene, cam = build_hospital(), camera_from_args(args)
    pygame.init()
    screen = pygame.display.set_mode((1152, 720), pygame.RESIZABLE)
    pygame.display.set_caption('Aster Medical | Hospital 3D')
    fonts = [pygame.font.SysFont('Helvetica,Arial', size, bold=bold)
             for size, bold in ((14, False), (17, False), (26, True), (15, True), (36, True))]
    clock = pygame.time.Clock()
    res_levels, res_i = [0.30, 0.42, 0.55, 0.70, 0.85, 1.0], 2
    use_z, ceiling, show_map = not args.noz, not args.noceiling, True
    show_help, directory, diagnostics = False, False, False
    mode = args.mode
    touring, tour_dist, tour_yaw = False, 0.0, 0.0
    ride, saved_walk, buttons = None, None, []
    toast, toast_t = 'Welcome. TAB opens the directory. H shows all controls.', 6.0
    running, frame_count, screenshot_requested = True, 0, False
    palette = dict(bg=(19, 35, 43), muted=(160, 185, 189), light=(235, 244, 240), accent=(93, 211, 180))

    def label(message, x, y, font=1, color=None):
        screen.blit(fonts[font].render(str(message), True, color or palette['light']), (x, y))

    def panel(rect, alpha=232, color=None):
        surface = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(surface, (*(color or palette['bg']), alpha), surface.get_rect(), border_radius=14)
        screen.blit(surface, rect[:2])

    def button(rect, title, action, active=False):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(screen, (42, 104, 98) if active else ((48, 75, 83) if hover else (31, 51, 60)), rect, border_radius=8)
        label(title, rect.x + 12, rect.y + 9, 3)
        buttons.append((rect, action))

    def teleport(index):
        nonlocal touring, directory, toast, toast_t, saved_walk
        if cam.overview:
            cam.overview = False
            saved_walk = None
        name, x, z, yaw = floor_waypoints(cam.level)[index]
        set_floor(cam, cam.level)
        cam.x, cam.z, cam.yaw, cam.pitch = x, z, math.radians(yaw), 0
        touring, directory = False, False
        toast, toast_t = name + ' / ' + FLOOR_NAMES[cam.level], 3.0

    def start_ride(target):
        nonlocal ride, touring, directory, toast, toast_t, saved_walk
        if target == cam.level or ride:
            return
        touring, directory, saved_walk = False, False, None
        cam.overview = False
        cam.x, cam.z, cam.yaw, cam.pitch = LIFT_X, LIFT_Z, math.pi, 0
        cam.y = EYE_H + cam.level * FLOOR_HEIGHT
        ride = {'elapsed': 0.0, 'target': target, 'arrived': False}
        toast, toast_t = 'Lift travelling to ' + FLOOR_NAMES[target], 2.5

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.12)
        width, height = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(800, event.w), max(600, event.h)), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not ride:
                for rect, action in reversed(buttons):
                    if rect.collidepoint(event.pos):
                        kind, value = action
                        if kind == 'floor':
                            start_ride(value)
                        elif kind == 'room':
                            teleport(value)
                        elif kind == 'directory':
                            directory = not directory
                        break
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_ESCAPE:
                    if directory or show_help:
                        directory = show_help = False
                    else:
                        running = False
                elif key == pygame.K_h:
                    show_help, directory = not show_help, False
                elif key == pygame.K_TAB and not ride:
                    directory, show_help = not directory, False
                    touring = False
                elif key == pygame.K_F12:
                    screenshot_requested = True
                elif key == pygame.K_F3:
                    diagnostics = not diagnostics
                elif ride:
                    continue
                elif key == pygame.K_e and not cam.overview:
                    if near_lift(cam):
                        start_ride(1 - cam.level)
                    else:
                        toast, toast_t = 'The lift is in the lobby. TAB opens the directory.', 3.0
                elif key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    start_ride(1 if key == pygame.K_PAGEUP else 0)
                elif key == pygame.K_o:
                    touring = False
                    if cam.overview and saved_walk:
                        cam, saved_walk = saved_walk, None
                    elif cam.overview:
                        cam = Camera()
                    else:
                        saved_walk, cam = cam, overview_camera()
                elif key == pygame.K_z:
                    use_z = not use_z
                    toast, toast_t = ('Z-buffer enabled' if use_z else 'Z-buffer disabled: demonstration mode'), 3
                elif key == pygame.K_v:
                    mode = 'normal' if mode == 'depth' else 'depth'
                elif key == pygame.K_f:
                    mode = 'normal' if mode == 'wire' else 'wire'
                elif key == pygame.K_c:
                    ceiling = not ceiling
                elif key == pygame.K_m:
                    show_map = not show_map
                elif key == pygame.K_t:
                    touring = not touring
                    tour_dist = 0
                    directory = False
                    if touring:
                        cam = Camera()
                        saved_walk = None
                        toast, toast_t = 'Guided tour / clinical care, then recovery level', 4
                elif key == pygame.K_LEFTBRACKET:
                    res_i = max(0, res_i - 1)
                elif key == pygame.K_RIGHTBRACKET:
                    res_i = min(len(res_levels) - 1, res_i + 1)
                elif key == pygame.K_MINUS:
                    cam.fov = max(40, cam.fov - 4)
                elif key == pygame.K_EQUALS:
                    cam.fov = min(110, cam.fov + 4)
                elif pygame.K_0 <= key <= pygame.K_9:
                    teleport((key - pygame.K_1) % 10)

        keys = pygame.key.get_pressed()
        if ride:
            ride['elapsed'] += dt
            if ride['elapsed'] >= 0.65 and not ride['arrived']:
                set_floor(cam, ride['target'], at_lift=True)
                ride['arrived'] = True
            if ride['elapsed'] >= 1.3:
                ride = None
        elif touring:
            previous_level = cam.level
            tour_dist += 2.6 * dt
            ty, total = building_tour(cam, tour_dist)
            tour_yaw = ty if tour_dist < 0.4 or previous_level != cam.level else smooth_angle(tour_yaw, ty, min(1, 5 * dt))
            cam.yaw, cam.pitch = tour_yaw, 0
            if previous_level != cam.level:
                toast, toast_t = 'Lift arrival / Recovery & rehabilitation', 4
            if tour_dist >= total:
                touring = False
                toast, toast_t = 'Tour complete. Explore either floor with TAB.', 4
        elif not directory and not show_help and not cam.overview:
            forward, strafe = int(keys[pygame.K_w]) - int(keys[pygame.K_s]), int(keys[pygame.K_d]) - int(keys[pygame.K_a])
            if forward or strafe:
                n = math.hypot(forward, strafe)
                speed = 4.2 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 2.6
                cam.move(scene, forward / n, strafe / n, speed * dt)
                # Stair zone: seamless floor transition at either end of the flight.
                if cam.level == 0 and 3.0 <= cam.x <= 6.2 and 9.65 <= cam.z <= 11.25:
                    set_floor(cam, 1)
                    cam.z = 10.8
                    toast, toast_t = 'Reached Level 1 via staircase', 2.5
                elif cam.level == 1 and 3.0 <= cam.x <= 6.2 and 4.0 <= cam.z <= 5.15:
                    set_floor(cam, 0)
                    cam.z = 5.2
                    toast, toast_t = 'Reached Ground Floor via staircase', 2.5
            cam.yaw += (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])) * 1.7 * dt
            cam.pitch = np.clip(cam.pitch + (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * 1.5 * dt, -0.95, 0.95)
            if pygame.mouse.get_pressed()[2]:
                mx, my = pygame.mouse.get_rel()
                cam.yaw += mx * 0.0035
                cam.pitch = np.clip(cam.pitch - my * 0.0035, -0.95, 0.95)
            else:
                pygame.mouse.get_rel()
        else:
            pygame.mouse.get_rel()

        width, height = screen.get_size()
        render_w, render_h = int(width * res_levels[res_i]), int(height * res_levels[res_i])
        started = time.perf_counter()
        frame, _ = render(scene, cam, render_w, render_h, use_zbuffer=use_z, show_ceiling=ceiling, mode=mode)
        elapsed = time.perf_counter() - started
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2)))
        screen.blit(pygame.transform.smoothscale(surf, (width, height)), (0, 0))
        buttons = []

        # Crisp native-resolution interface over the software-rendered scene.
        panel((20, 20, 360, 98))
        label('ASTER  /  MEDICAL CENTRE', 38, 34, 3, palette['accent'])
        label('Building overview' if cam.overview else scene.room_at(cam.x, cam.z, cam.level), 38, 57, 2)
        label('Two floors  /  18 destinations' if cam.overview else FLOOR_NAMES[cam.level], 38, 91, 0, palette['muted'])
        panel((width - 254, 20, 234, 58))
        button((width - 244, 30, 101, 38), 'G  Clinical', ('floor', 0), cam.level == 0 and not cam.overview)
        button((width - 137, 30, 107, 38), '1  Recovery', ('floor', 1), cam.level == 1 and not cam.overview)

        if show_map and not cam.overview:
            panel((width - 254, 90, 234, 192))
            label('YOU ARE HERE', width - 238, 102, 0, palette['accent'])
            mp = minimap(scene, cam, 5)
            ms = pygame.surfarray.make_surface(np.ascontiguousarray(mp.transpose(1, 0, 2)))
            screen.blit(ms, (width - 239, 128))
            label('Lift: west lobby  /  E to travel', width - 238, 260, 0, palette['muted'])
        if not directory and not show_help:
            panel((20, height - 69, width - 40, 49))
            label('WASD  Move     RMB  Look     TAB  Directory     H  Help', 38, height - 53, 0)
            button((width - 180, height - 62, 148, 35), 'Find a department', ('directory', None))
        if near_lift(cam) and not directory and not cam.overview and not ride:
            panel((width // 2 - 165, height - 125, 330, 40))
            label('E  /  Lift to ' + ('Level 1' if cam.level == 0 else 'Ground floor'), width // 2 - 145, height - 114, 3, palette['accent'])
        if toast_t > 0:
            toast_t -= dt
            text_surface = fonts[0].render(toast, True, palette['light'])
            tw = min(width - 48, text_surface.get_width() + 32)
            panel(((width - tw) // 2, height - 178, tw, 39))
            screen.blit(text_surface, ((width - tw) // 2 + 16, height - 167))
        if diagnostics:
            panel((20, 130, 360, 61))
            label(f'{render_w} x {render_h}  /  {1 / max(elapsed, 0.001):.1f} render FPS', 36, 141, 0)
            label(f'{mode.upper()}  /  Z-buffer {"ON" if use_z else "OFF"}  /  FOV {cam.fov:.0f}', 36, 165, 0, palette['muted'])
        if directory or show_help:
            veil = pygame.Surface((width, height), pygame.SRCALPHA)
            veil.fill((7, 18, 24, 155))
            screen.blit(veil, (0, 0))
            buttons = []
            x, y = (width - 680) // 2, (height - 480) // 2
            panel((x, y, 680, 480), 252)
            label('Hospital directory' if directory else 'Explore at your pace', x + 28, y + 25, 4)
            label('TAB / ESC to close' if directory else 'H / ESC to close', x + 450, y + 37, 0, palette['muted'])
            if directory:
                button((x + 28, y + 85, 300, 40), 'G  /  Clinical care', ('floor', 0), cam.level == 0)
                button((x + 346, y + 85, 306, 40), '1  /  Recovery & rehabilitation', ('floor', 1), cam.level == 1)
                for i, wp in enumerate(floor_waypoints(cam.level)):
                    row, col = divmod(i, 2)
                    button((x + 28 + col * 318, y + 146 + row * 52, 306, 43), f'{(i + 1) % 10}   {wp[0]}', ('room', i))
                label('Choose a floor, then a destination. Or walk to the lobby lift.', x + 28, y + 431, 0, palette['muted'])
            else:
                controls = [('W / A / S / D', 'Walk; hold Shift to move faster'),
                            ('Arrows / right mouse', 'Look around'), ('E', 'Use the lift when nearby'),
                            ('Page Up / Page Down', 'Quick transfer to upper / ground floor'),
                            ('TAB / 1..9, 0', 'Department directory / room shortcuts'),
                            ('T / O', 'Tour both floors / exterior overview'),
                            ('Z / V / F / C', 'Z-buffer / depth / wireframe / ceiling'),
                            ('M / F3 / F12', 'Map / performance information / screenshot'),
                            ('[ / ]   and   - / =', 'Render resolution and field of view')]
                for i, (key, desc) in enumerate(controls):
                    label(key, x + 28, y + 103 + i * 36, 3, palette['accent'])
                    label(desc, x + 255, y + 103 + i * 36, 0)
        if ride:
            fade = pygame.Surface((width, height), pygame.SRCALPHA)
            fade.fill((12, 28, 35, int(245 * math.sin(math.pi * ride['elapsed'] / 1.3))))
            screen.blit(fade, (0, 0))
            label('LIFT  /  ' + FLOOR_NAMES[ride['target']], width // 2 - 220, height // 2, 2)
        if screenshot_requested:
            output = Path(__file__).parent / 'screenshots' / time.strftime('hospital-%Y%m%d-%H%M%S.png')
            output.parent.mkdir(exist_ok=True)
            pygame.image.save(screen, str(output))
            screenshot_requested = False
            toast, toast_t = 'Screenshot saved to screenshots/' + output.name, 4
        pygame.display.flip()
        frame_count += 1
        if args.smoke_frames and frame_count >= args.smoke_frames:
            if args.ui_shot:
                Path(args.ui_shot).parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, args.ui_shot)
            running = False
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description='Aster Medical: two-floor 3D hospital, NumPy perspective + Z-buffer renderer')
    output = ap.add_mutually_exclusive_group()
    output.add_argument('--shot', help='render a PNG and exit')
    output.add_argument('--export', help='export both floors as OBJ, MTL and GLB')
    output.add_argument('--tour', help='render an animated tour of BOTH floors')
    ap.add_argument('--frames', type=int, default=90)
    ap.add_argument('--floor', type=int, choices=[0, 1], default=0, help='starting floor: 0=ground, 1=recovery')
    ap.add_argument('--overview', action='store_true', help='show the complete two-storey building')
    ap.add_argument('--pos', type=float, nargs=2, default=[1.6, 12.0], metavar=('X', 'Z'))
    ap.add_argument('--yaw', type=float, default=90)
    ap.add_argument('--pitch', type=float, default=0)
    ap.add_argument('--fov', type=float, default=72)
    ap.add_argument('--size', type=int, nargs=2, default=[960, 600], metavar=('W', 'H'))
    ap.add_argument('--mode', choices=['normal', 'depth', 'wire'], default='normal')
    ap.add_argument('--noz', action='store_true')
    ap.add_argument('--noceiling', action='store_true')
    ap.add_argument('--minimap', action='store_true')
    ap.add_argument('--smoke-frames', type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument('--ui-shot', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not 30 <= args.fov <= 120:
        ap.error('--fov must be between 30 and 120 degrees')
    if not all(64 <= n <= 4096 for n in args.size):
        ap.error('--size dimensions must each be between 64 and 4096')
    if not 2 <= args.frames <= 1800:
        ap.error('--frames must be between 2 and 1800')
    if not all(math.isfinite(n) for n in (*args.pos, args.yaw, args.pitch, args.fov)):
        ap.error('camera values must be finite numbers')
    if not -80 <= args.pitch <= 80:
        ap.error('--pitch must be between -80 and 80 degrees')
    if args.export:
        export_all(args.export)
    elif args.shot:
        run_shot(args)
    elif args.tour:
        args.size = args.size if args.size != [960, 600] else [360, 225]
        run_tour_gif(args)
    else:
        run_viewer(args)


if __name__ == '__main__':
    main()
