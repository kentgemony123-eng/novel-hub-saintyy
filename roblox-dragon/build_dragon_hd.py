"""
Detailed dragon for Roblox (single MeshPart, baked textures).

Run inside Blender (Scripting tab -> Open -> Run Script), or headless:
    blender --background --python build_dragon_hd.py
    python build_dragon_hd.py           (with `pip install bpy`)

What it does:
  1. Builds the organic body (torso, legs, toes, neck, sculpted head with brow
     ridges, cheekbones, open jaw, eye sockets, nostrils) from overlapping
     shapes, fuses them with a voxel remesh, then decimates to a game budget.
  2. Adds hard-surface details: ridged horns, teeth, claws, dorsal blades,
     eyes with slit pupils, and bat-style wings with a billowing membrane.
  3. UV-unwraps everything into one layout and bakes a 1024px color map
     (scales, belly plates, ambient occlusion) and a 1024px normal map.

Outputs (next to this script):
    dragon_hd.blend, dragon_hd.fbx, dragon_hd.glb,
    dragon_hd_color.png, dragon_hd_normal.png

The dragon faces -Y in Blender with its origin between its feet.
"""

import math
import os

import bpy  # noqa: I001 - bpy must be imported before bmesh outside Blender
import bmesh
import numpy as np
from mathutils import Euler, Vector
from mathutils.bvhtree import BVHTree

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRI_BUDGET = 19000      # Roblox MeshPart limit is 20,000 triangles
TEX_SIZE = 1024         # Roblox downsizes anything larger to 1024
VOXEL = 0.03            # remesh resolution for the organic body

# Colors (linear RGB)
RED = Vector((0.30, 0.022, 0.012))
RED_DARK = Vector((0.07, 0.006, 0.005))
BELLY = Vector((0.72, 0.46, 0.18))
MOUTH = Vector((0.18, 0.015, 0.02))
HORN_BASE = Vector((0.10, 0.075, 0.05))
HORN_TIP = Vector((0.75, 0.68, 0.52))
IVORY = Vector((0.85, 0.80, 0.65))
CLAW_BASE = Vector((0.30, 0.27, 0.22))
CLAW_TIP = Vector((0.04, 0.035, 0.03))
MEMBRANE_EDGE = Vector((0.12, 0.012, 0.008))
MEMBRANE_MID = Vector((0.55, 0.13, 0.03))
IRIS_IN = Vector((1.00, 0.70, 0.05))
IRIS_OUT = Vector((0.60, 0.12, 0.01))
BLACK = Vector((0.01, 0.01, 0.01))


# ==========================================================================
# Generic helpers
# ==========================================================================
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def select_only(objs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def apply_modifiers(obj):
    select_only([obj])
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def join(objs, name):
    select_only(objs)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def mesh_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    obj = link(bpy.data.objects.new(name, me))
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return obj


def set_attrs(obj, colors, belly=0.0, scaly=0.0, membrane=0.0, plate=0.0):
    """Per-vertex shading inputs read by the bake material."""
    me = obj.data
    n = len(me.vertices)

    def arr(v):
        a = np.asarray(v, dtype=np.float32)
        return np.full(n, float(a), np.float32) if a.ndim == 0 else a

    cols = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if len(cols) == 1:
        cols = np.repeat(cols, n, axis=0)
    rgba = np.ones((n, 4), np.float32)
    rgba[:, :3] = cols
    ca = me.color_attributes.get("base") or me.color_attributes.new("base", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", rgba.ravel())
    for key, val in (("belly", belly), ("scaly", scaly), ("membrane", membrane), ("plate", plate)):
        at = me.attributes.get(key) or me.attributes.new(key, "FLOAT", "POINT")
        at.data.foreach_set("value", arr(val))


def catmull(ctrl, n):
    """Sample n points along a Catmull-Rom spline through ctrl points."""
    P = [Vector(p) for p in ctrl]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    segs = len(ctrl) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * segs
        k = min(int(t), segs - 1)
        u = t - k
        p0, p1, p2, p3 = P[k], P[k + 1], P[k + 2], P[k + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
    return out


def interp(values, n):
    xs = np.linspace(0, 1, len(values))
    return list(np.interp(np.linspace(0, 1, n), xs, values))


def tube(name, ctrl, ctrl_r, n=12, segs=10, squash=None, ridges=0.0):
    """Tapered tube along a smooth path. Radius 0 at the end gives a point.
    squash=(axis, factor) flattens the cross-section along an axis (blades).
    Returns (obj, t) where t is each vertex's 0..1 position along the tube."""
    pts = catmull(ctrl, n) if len(ctrl) > 2 else [Vector(ctrl[0]).lerp(Vector(ctrl[1]), i / (n - 1)) for i in range(n)]
    rad = interp(ctrl_r, n)
    if ridges:
        rad = [r * (1.0 - ridges * (i % 2)) for i, r in enumerate(rad)]
    verts, faces, tv = [], [], []
    tangent = (pts[1] - pts[0]).normalized()
    ref = Vector((0, 0, 1)) if abs(tangent.z) < 0.9 else Vector((1, 0, 0))
    normal = tangent.cross(ref).normalized()
    rings = []
    for i, p in enumerate(pts):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        tangent = (b - a).normalized()
        normal = (normal - tangent * normal.dot(tangent)).normalized()
        binormal = tangent.cross(normal)
        t = i / (n - 1)
        if rad[i] < 1e-4:
            rings.append([len(verts)])
            verts.append(p)
            tv.append(t)
            continue
        ring = []
        for s in range(segs):
            ang = 2 * math.pi * s / segs
            off = (normal * math.cos(ang) + binormal * math.sin(ang)) * rad[i]
            if squash:
                axis, f = Vector(squash[0]).normalized(), squash[1]
                off -= axis * off.dot(axis) * (1 - f)
            ring.append(len(verts))
            verts.append(p + off)
            tv.append(t)
        rings.append(ring)
    r0 = rings[0]
    if len(r0) > 1:  # start cap
        c = len(verts)
        verts.append(pts[0])
        tv.append(0.0)
        faces += [(c, r0[(s + 1) % segs], r0[s]) for s in range(segs)]
    for r1, r2 in zip(rings, rings[1:]):
        if len(r1) == 1:
            faces += [(r1[0], r2[(s + 1) % segs], r2[s]) for s in range(segs)]
        elif len(r2) == 1:
            faces += [(r1[s], r1[(s + 1) % segs], r2[0]) for s in range(segs)]
        else:
            faces += [(r1[s], r1[(s + 1) % segs], r2[(s + 1) % segs], r2[s]) for s in range(segs)]
    if len(rings[-1]) > 1:  # end cap
        c = len(verts)
        verts.append(pts[-1])
        tv.append(1.0)
        faces += [(rings[-1][s], rings[-1][(s + 1) % segs], c) for s in range(segs)]
    obj = mesh_object(name, verts, faces)
    # mesh_object may merge verts; recompute t from nearest sample
    t_arr = np.array([nearest_param(v.co, pts) for v in obj.data.vertices], np.float32)
    return obj, t_arr


def nearest_param(co, pts):
    best, bi = 1e9, 0
    for i, p in enumerate(pts):
        d = (co - p).length_squared
        if d < best:
            best, bi = d, i
    return bi / (len(pts) - 1)


def ellipsoid(name, center, radii, rot=(0, 0, 0), segs=24, rings=14):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(name, me))
    obj.location = center
    obj.scale = radii
    obj.rotation_euler = Euler([math.radians(a) for a in rot])
    select_only([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def mirrored(fn):
    """Call fn(side) for side in (+1, -1) and collect results."""
    out = []
    for s in (1, -1):
        r = fn(s)
        out += r if isinstance(r, list) else [r]
    return out


# ==========================================================================
# Organic body
# ==========================================================================
SPINE = [  # (point, radius) tail tip -> back of head
    ((0, 9.3, 1.05), 0.07), ((0, 8.0, 1.3), 0.22), ((0, 6.4, 1.8), 0.40),
    ((0, 4.8, 2.4), 0.65), ((0, 3.2, 2.9), 0.95), ((0, 1.8, 3.1), 1.18),
    ((0, 0.0, 3.25), 1.32), ((0, -1.6, 3.35), 1.22), ((0, -2.65, 3.85), 0.80),
    ((0, -3.15, 4.75), 0.56), ((0, -3.45, 5.6), 0.46), ((0, -3.75, 6.25), 0.44),
]
HIP, CHEST = 5, 7
BACK_LEG = [((1.0, 1.7, 2.55), 0.65), ((1.2, 1.0, 1.5), 0.42),
            ((1.2, 1.65, 0.6), 0.26), ((1.2, 1.25, 0.2), (0.30, 0.16))]
FRONT_LEG = [((0.9, -1.55, 2.75), 0.50), ((1.1, -1.3, 1.6), 0.34),
             ((1.1, -1.75, 0.5), 0.23), ((1.1, -2.0, 0.18), (0.26, 0.14))]
FEET = [(1.2, 1.25, 0.2, 1.0), (1.1, -2.0, 0.18, 0.85)]  # x, y, z, size
EYE = Vector((0.335, -4.62, 6.68))
EYE_R = 0.13


def build_skin_body():
    verts = [p for p, _ in SPINE]
    radii = [r for _, r in SPINE]
    edges = [(i, i + 1) for i in range(len(SPINE) - 1)]
    for side in (1, -1):
        for root, leg in ((HIP, BACK_LEG), (CHEST, FRONT_LEG)):
            prev = root
            for (x, y, z), r in leg:
                verts.append((x * side, y, z))
                radii.append(r)
                edges.append((prev, len(verts) - 1))
                prev = len(verts) - 1
    me = bpy.data.meshes.new("Body")
    me.from_pydata(verts, edges, [])
    obj = link(bpy.data.objects.new("Body", me))
    obj.modifiers.new("skin", "SKIN")
    sk = me.skin_vertices[0].data
    for i, r in enumerate(radii):
        sk[i].radius = r if isinstance(r, tuple) else (r, r)
    sk[0].use_root = True
    obj.modifiers.new("sub", "SUBSURF").levels = 2
    apply_modifiers(obj)
    return obj


def build_organic_parts():
    parts = [build_skin_body()]
    E = ellipsoid
    # Muscles
    parts += mirrored(lambda s: [
        E("Thigh", (0.95 * s, 1.55, 2.35), (0.5, 0.85, 0.8), (15, 0, 0)),
        E("Shoulder", (0.85 * s, -1.6, 2.75), (0.42, 0.6, 0.65), (-10, 0, 0)),
        E("Calf", (1.2 * s, 1.35, 1.0), (0.28, 0.4, 0.45), (-30, 0, 0)),
    ])
    # Head
    parts += [
        E("Cranium", (0, -4.15, 6.55), (0.52, 0.62, 0.48)),
        E("Occiput", (0, -3.75, 6.6), (0.40, 0.40, 0.42)),
        E("Snout1", (0, -4.75, 6.45), (0.44, 0.60, 0.34)),
        E("Snout2", (0, -5.35, 6.37), (0.34, 0.50, 0.25)),
        E("Nose", (0, -5.85, 6.33), (0.26, 0.30, 0.19)),
        E("Bridge", (0, -5.0, 6.65), (0.13, 0.75, 0.10), (8, 0, 0)),
        E("Throat", (0, -3.8, 5.95), (0.35, 0.45, 0.32)),
        # Lower jaw, hinged open a little
        E("JawBack", (0, -4.25, 6.05), (0.42, 0.50, 0.22)),
        E("JawMid", (0, -4.9, 5.88), (0.31, 0.50, 0.14), (6, 0, 0)),
        E("JawFront", (0, -5.5, 5.78), (0.22, 0.38, 0.10), (6, 0, 0)),
        E("Chin", (0, -5.85, 5.75), (0.15, 0.15, 0.09)),
    ]
    parts += mirrored(lambda s: [
        E("Brow", (0.32 * s, -4.55, 6.90), (0.17, 0.40, 0.11), (10, 0, -12 * s)),
        E("Cheek", (0.42 * s, -4.35, 6.42), (0.17, 0.50, 0.17)),
        E("JawMuscle", (0.38 * s, -3.95, 6.25), (0.22, 0.35, 0.30)),
        E("NostrilBump", (0.15 * s, -5.95, 6.45), (0.10, 0.15, 0.08)),
    ])
    # Tail spade
    parts += mirrored(lambda s: E("Spade", (0.22 * s, 9.75, 1.02), (0.30, 0.55, 0.07), (0, 0, 35 * s)))
    # Toes
    for fx, fy, fz, k in FEET:
        for s in (1, -1):
            for dx in (-0.2, 0.0, 0.2):
                x0 = (fx + dx * 0.4) * s
                x1 = (fx + dx) * s
                o, _ = tube("Toe", [(x0, fy, fz + 0.05), (x1, fy - 0.45 * k, 0.1),
                                    (x1 + 0.02 * s * np.sign(dx), fy - 0.62 * k, 0.08)],
                            [0.13 * k, 0.09 * k, 0.07 * k], n=6, segs=8)
                parts.append(o)
    return parts


def build_organic():
    org = join(build_organic_parts(), "Organic")
    r = org.modifiers.new("remesh", "REMESH")
    r.mode = "VOXEL"
    r.voxel_size = VOXEL
    apply_modifiers(org)

    # Carve eye sockets and nostrils
    cutters = mirrored(lambda s: [
        ellipsoid("EyeSocket", (EYE.x * s + 0.03 * s, EYE.y, EYE.z), (0.17, 0.17, 0.15)),
        ellipsoid("Nostril", (0.15 * s, -6.08, 6.47), (0.05, 0.08, 0.04), (0, 0, 0), 12, 8),
    ])
    cutter = join(cutters, "Cutter")
    b = org.modifiers.new("carve", "BOOLEAN")
    b.operation = "DIFFERENCE"
    b.object = cutter
    try:
        b.solver = "MANIFOLD"  # fast and exact on the closed remeshed surface (Blender 4.5+)
    except TypeError:
        pass
    apply_modifiers(org)
    bpy.data.objects.remove(cutter)
    return org


def decimate_to(obj, tris):
    cur = tri_count(obj)
    if cur > tris:
        d = obj.modifiers.new("dec", "DECIMATE")
        d.ratio = tris / cur
        d.use_collapse_triangulate = True
        apply_modifiers(obj)


def color_organic(org):
    me = org.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    nm = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_get("normal", nm)
    co, nm = co.reshape(-1, 3), nm.reshape(-1, 3)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    ax = np.abs(x)

    top = smoothstep(0.3, 0.95, nm[:, 2])
    col = np.array(RED)[None, :] + (np.array(RED_DARK) - np.array(RED))[None, :] * (top * 0.85)[:, None]

    leg = smoothstep(0.55, 0.85, ax) * smoothstep(2.6, 2.0, z) * (np.abs(y + 0.4) < 2.6)
    belly = smoothstep(-0.2, -0.6, nm[:, 2]) * (1 - leg)
    neck = (y < -2.3) & (z > 3.4) & (z < 6.3) & (ax < 0.5)
    belly = np.maximum(belly, smoothstep(-0.35, -0.8, nm[:, 1]) * neck * smoothstep(0.3, -0.2, nm[:, 2]))
    belly *= (z > 0.35)  # not the toes
    col = col + (np.array(BELLY)[None, :] - col) * belly[:, None]

    # Mouth interior: palate faces down, jaw top faces up, inside the lips
    gap_mouth = (y < -4.55) & (ax < 0.27)
    palate = gap_mouth & (nm[:, 2] < -0.35) & (z > 5.98) & (z < 6.22)
    tongue = gap_mouth & (nm[:, 2] > 0.35) & (z > 5.75) & (z < 6.12)
    mouth = (palate | tongue).astype(np.float32)
    col = col + (np.array(MOUTH)[None, :] - col) * mouth[:, None]
    belly *= 1 - mouth

    # Darker skin around the eyes and nostrils, tail spade edge
    for s in (1, -1):
        d = np.linalg.norm(co - np.array((EYE.x * s, EYE.y, EYE.z)), axis=1)
        col *= (0.55 + 0.45 * smoothstep(0.17, 0.32, d))[:, None]
    col *= (0.7 + 0.3 * smoothstep(9.9, 9.4, y))[:, None]

    # Belly plate coordinate: arc length along the underside line
    line = np.array([(0, 9.3, 0.9), (0, 4.8, 1.8), (0, 1.8, 2.0), (0, -1.6, 2.2),
                     (0, -2.9, 3.4), (0, -3.6, 5.2), (0, -4.2, 5.8), (0, -5.9, 5.65)], np.float32)
    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    best_d = np.full(n, 1e9, np.float32)
    plate = np.zeros(n, np.float32)
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        ab = b - a
        t = np.clip(((co - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(co - (a + t[:, None] * ab), axis=1)
        m = d < best_d
        best_d[m] = d[m]
        plate[m] = cum[i] + t[m] * seg_len[i]
    scaly = 1.0 - mouth
    set_attrs(org, col, belly=belly, scaly=scaly, plate=plate)


# ==========================================================================
# Hard-surface details
# ==========================================================================
def gradient(t, a, b, power=1.0):
    t = np.asarray(t)[:, None] ** power
    return np.array(a)[None, :] * (1 - t) + np.array(b)[None, :] * t


def build_horns():
    parts = []

    def horn(name, ctrl, r0, n, segs, ridges=0.08):
        o, t = tube(name, ctrl, [r0, r0 * 0.75, r0 * 0.4, 0.0], n=n, segs=segs, ridges=ridges)
        set_attrs(o, gradient(t, HORN_BASE, HORN_TIP, 0.7))
        return o

    for s in (1, -1):
        parts.append(horn("Horn", [(0.28 * s, -3.95, 6.80), (0.40 * s, -3.45, 7.20),
                                   (0.50 * s, -2.85, 7.40), (0.56 * s, -2.30, 7.28)], 0.17, 18, 12))
        parts.append(horn("Horn2", [(0.48 * s, -3.95, 6.58), (0.75 * s, -3.6, 6.72),
                                    (0.95 * s, -3.2, 6.70), (1.08 * s, -2.9, 6.6)], 0.10, 12, 8))
        parts.append(horn("CheekSpike", [(0.48 * s, -3.95, 6.08), (0.68 * s, -3.7, 5.95),
                                         (0.80 * s, -3.45, 5.86)], 0.07, 6, 6, 0))
        parts.append(horn("CheekSpike2", [(0.42 * s, -4.25, 5.98), (0.58 * s, -4.05, 5.85),
                                          (0.66 * s, -3.85, 5.76)], 0.05, 6, 6, 0))
        parts.append(horn("BrowSpike", [(0.38 * s, -4.35, 6.98), (0.48 * s, -4.15, 7.1),
                                        (0.55 * s, -4.0, 7.12)], 0.05, 6, 6, 0))
    for y, ln in ((-5.45, 0.18), (-5.15, 0.14)):
        parts.append(horn("ChinSpike", [(0, y, 5.70), (0, y + 0.08, 5.70 - ln * 0.6),
                                        (0, y + 0.18, 5.70 - ln)], 0.045, 6, 6, 0))
    return parts


def build_teeth(bvh):
    parts = []

    def cast(origin, direction):
        # Only accept hits on the palate/jaw surface facing into the mouth
        loc, nrm, _i, _d = bvh.ray_cast(Vector(origin), Vector(direction), 0.35)
        return loc if loc is not None and nrm.dot(Vector(direction)) < -0.5 else None

    for s in (1, -1):
        for i, y in enumerate(np.linspace(-4.85, -5.9, 7)):
            w = float(np.interp(y, [-5.9, -4.85], [0.17, 0.27]))
            fang = i == 4
            # Upper tooth: hang down from the palate
            gap_mid = float(np.interp(y, [-5.9, -4.85], [6.02, 6.06]))
            p = cast((w * s, y, gap_mid), (0, 0, 1))
            if p is not None:
                ln = 0.22 if fang else 0.11
                o, t = tube("Tooth", [p + Vector((0, 0, 0.03)), p - Vector((0, 0.02, ln * 0.6)),
                                      p - Vector((0, 0.05, ln))], [0.045 if fang else 0.03, 0.022, 0.0], n=4, segs=5)
                set_attrs(o, gradient(t, IVORY * 0.85, IVORY))
                parts.append(o)
            # Lower tooth: stand up from the jaw, offset between the uppers
            p = cast((w * 0.6 * s, y - 0.08, gap_mid), (0, 0, -1))
            if p is not None and i < 6:
                ln = 0.15 if i == 5 else 0.08
                o, t = tube("Tooth", [p - Vector((0, 0, 0.03)), p + Vector((0, -0.02, ln * 0.6)),
                                      p + Vector((0, -0.04, ln))], [0.03, 0.02, 0.0], n=4, segs=5)
                set_attrs(o, gradient(t, IVORY * 0.85, IVORY))
                parts.append(o)
    # Tongue
    o, t = tube("Tongue", [(0, -4.6, 5.98), (0, -5.1, 5.95), (0, -5.45, 5.88)],
                [0.14, 0.12, 0.06], n=8, segs=10, squash=((0, 0, 1), 0.35))
    set_attrs(o, [(0.55, 0.08, 0.10)], scaly=0.0)
    parts.append(o)
    return parts


def build_eyes():
    parts = []
    for s in (1, -1):
        c = Vector((EYE.x * s, EYE.y, EYE.z))
        look = Vector((1.0 * s, -0.55, 0.12)).normalized()
        o = ellipsoid("Eye", c, (EYE_R, EYE_R, EYE_R), (0, 0, 0), 20, 12)
        co = np.array([v.co[:] for v in o.data.vertices], np.float32)
        d = co - np.array(c[:])
        d /= np.linalg.norm(d, axis=1)[:, None]
        facing = d @ np.array(look[:])
        iris = smoothstep(0.2, 0.95, facing)[:, None]
        col = np.array(IRIS_OUT)[None, :] * (1 - iris) + np.array(IRIS_IN)[None, :] * iris
        col *= smoothstep(-0.2, 0.3, facing)[:, None] * 0.9 + 0.1
        set_attrs(o, col)
        parts.append(o)
        # Vertical slit pupil sitting on the front of the eye
        side = look.cross(Vector((0, 0, 1))).normalized()
        pc = c + look * (EYE_R * 0.93)
        p, _ = tube("Pupil", [pc - Vector((0, 0, 0.095)), pc, pc + Vector((0, 0, 0.095))],
                    [0.0, 0.03, 0.0], n=7, segs=8, squash=(side, 0.3))
        set_attrs(p, [BLACK[:]])
        parts.append(p)
    return parts


def build_claws():
    parts = []
    for fx, fy, fz, k in FEET:
        for s in (1, -1):
            for dx in (-0.2, 0.0, 0.2):
                x = (fx + dx) * s + 0.02 * s * np.sign(dx)
                yt = fy - 0.62 * k
                o, t = tube("Claw", [(x, yt + 0.06, 0.1), (x, yt - 0.12, 0.09), (x, yt - 0.24, 0.0)],
                            [0.065 * k, 0.045 * k, 0.0], n=5, segs=6)
                set_attrs(o, gradient(t, CLAW_BASE, CLAW_TIP))
                parts.append(o)
    return parts


def build_spikes(bvh):
    parts = []
    for y in np.arange(-3.0, 9.0, 0.48):
        h = float(np.interp(y, [-3.0, -1.0, 1.0, 4.0, 9.0], [0.28, 0.52, 0.55, 0.38, 0.12]))
        loc, _n, _i, _d = bvh.ray_cast(Vector((0, y, 20)), Vector((0, 0, -1)))
        if not loc:
            continue
        base = loc - Vector((0, 0, 0.05))
        tip = base + Vector((0, h * 0.75, h))
        mid = base.lerp(tip, 0.5) + Vector((0, 0.05, -0.02))
        o, t = tube("Spike", [base, mid, tip], [h * 0.38, h * 0.22, 0.0], n=4, segs=5,
                    squash=((1, 0, 0), 0.3))
        set_attrs(o, gradient(t, RED_DARK, HORN_TIP, 1.8))
        parts.append(o)
    return parts


# --------------------------------------------------------------------------
# Wings: bones + Coons-patch membrane panels that billow between fingers
# --------------------------------------------------------------------------
def coons(A, B, C, D, ns, nu):
    """A(s): u=0 edge, B(s): u=1 edge, C(u): s=0 edge, D(u): s=1 edge."""
    P00, P10, P01, P11 = A(0), A(1), B(0), B(1)
    grid = []
    for i in range(ns + 1):
        s = i / ns
        row = []
        for j in range(nu + 1):
            u = j / nu
            p = ((1 - u) * A(s) + u * B(s) + (1 - s) * C(u) + s * D(u)
                 - ((1 - s) * (1 - u) * P00 + s * (1 - u) * P10 + (1 - s) * u * P01 + s * u * P11))
            row.append((p, s, u))
        grid.append(row)
    return grid


def curve_fn(ctrl):
    pts = catmull(ctrl, 33) if len(ctrl) > 2 else [Vector(ctrl[0]).lerp(Vector(ctrl[1]), i / 32) for i in range(33)]

    def f(t):
        x = t * 32
        i = min(int(x), 31)
        return pts[i].lerp(pts[i + 1], x - i)
    return f


def build_wing(s):
    S = Vector((0.7 * s, -1.3, 4.1))
    E = Vector((2.8 * s, -0.9, 5.6))
    W = Vector((4.9 * s, 0.0, 6.3))
    fingers = [  # (knuckle, tip)
        (Vector((6.6 * s, 0.4, 5.9)), Vector((7.9 * s, 1.1, 5.0))),
        (Vector((6.4 * s, 1.6, 5.2)), Vector((7.6 * s, 2.8, 3.9))),
        (Vector((5.5 * s, 2.2, 4.7)), Vector((6.0 * s, 3.9, 3.3))),
        (Vector((4.4 * s, 2.2, 4.8)), Vector((3.9 * s, 3.9, 3.5))),
    ]
    Bd = Vector((0.7 * s, 1.6, 3.8))
    parts = []

    # Bones
    o, t = tube("WingArm", [S, (S + E) / 2 + Vector((0, -0.1, 0.15)), E, W],
                [0.2, 0.17, 0.15, 0.13], n=14, segs=10)
    set_attrs(o, [RED[:]], scaly=1.0)
    parts.append(o)
    for k, tip in fingers:
        o, t = tube("Finger", [W, k, tip], [0.085, 0.06, 0.015], n=9, segs=6)
        set_attrs(o, [(RED_DARK * 1.4)[:]], scaly=1.0)
        parts.append(o)
    o, t = tube("ThumbClaw", [W, W + Vector((0.12 * s, -0.25, 0.3)), W + Vector((0.1 * s, -0.48, 0.38))],
                [0.07, 0.04, 0.0], n=6, segs=8)
    set_attrs(o, gradient(t, CLAW_BASE, CLAW_TIP))
    parts.append(o)

    # Membrane panels
    finger_fns = [curve_fn([W, k, tip]) for k, tip in fingers]
    arm_back = curve_fn([W, E, S])
    body_edge = curve_fn([S, Bd])

    def scallop(a, b, depth):
        def f(u):
            p = a.lerp(b, u)
            return p + (W - p).normalized() * depth * math.sin(math.pi * u) * (a - b).length
        return f

    verts, faces, su = [], [], []

    def add_panel(grid, billow):
        base = len(verts)
        nu = len(grid[0]) - 1
        for row in grid:
            for p, s_, u_ in row:
                bump = math.sin(math.pi * s_) * math.sin(math.pi * u_) * billow
                verts.append(p + Vector((0, 0.05, -1)) * bump)
                su.append(math.sin(math.pi * u_) * math.sin(math.pi * min(s_ * 1.3, 1)))
        for i in range(len(grid) - 1):
            for j in range(nu):
                a = base + i * (nu + 1) + j
                faces.append((a, a + 1, a + nu + 2, a + nu + 1))

    for a, b in zip(finger_fns, finger_fns[1:]):
        add_panel(coons(a, b, lambda u: W, scallop(a(1), b(1), 0.22), 10, 6), 0.22)
    add_panel(coons(finger_fns[-1], body_edge, arm_back, scallop(fingers[-1][1], Bd, 0.18), 10, 8), 0.35)

    mem = mesh_object("Membrane", verts, faces)
    sol = mem.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.035
    sol.offset = 0
    apply_modifiers(mem)
    co = np.array([v.co[:] for v in mem.data.vertices], np.float32)
    vco = np.array([v[:] for v in verts], np.float32)
    # transfer the edge->middle factor to solidified verts by nearest source vert
    idx = np.array([np.argmin(np.sum((vco - c) ** 2, axis=1)) for c in co])
    f = np.array(su, np.float32)[idx]
    set_attrs(mem, gradient(f, MEMBRANE_EDGE, MEMBRANE_MID, 0.6), membrane=1.0)
    parts.append(mem)
    return parts


# ==========================================================================
# Material, UVs and baking
# ==========================================================================
class Nodes:
    def __init__(self, mat):
        self.nt = mat.node_tree

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.nt.links.new(val, sock)
        else:
            sock.default_value = val

    def maprange(self, v, a, b, c, d):
        n = self.new("ShaderNodeMapRange")
        for i, val in enumerate((v, a, b, c, d)):
            self.put(n.inputs[i], val)
        return n.outputs[0]

    def mix(self, f, a, b):
        return self.maprange(f, 0.0, 1.0, a, b)

    def math(self, op, a, b=0.0):
        n = self.new("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        return n.outputs[0]

    def attr(self, name):
        return self.new("ShaderNodeAttribute", attribute_name=name)


def make_bake_material():
    mat = bpy.data.materials.new("DragonBake")
    N = Nodes(mat)
    nt = N.nt
    bsdf = nt.nodes["Principled BSDF"]

    coords = N.new("ShaderNodeTexCoord").outputs["Object"]
    vor_edge = N.new("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
    nt.links.new(coords, vor_edge.inputs["Vector"])
    vor_edge.inputs["Scale"].default_value = 9.0
    vor_cell = N.new("ShaderNodeTexVoronoi", feature="F1")
    nt.links.new(coords, vor_cell.inputs["Vector"])
    vor_cell.inputs["Scale"].default_value = 9.0
    cell_rand = N.new("ShaderNodeSeparateColor")
    nt.links.new(vor_cell.outputs["Color"], cell_rand.inputs[0])
    noise = N.new("ShaderNodeTexNoise")
    nt.links.new(coords, noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 4.0
    noise.inputs["Detail"].default_value = 6.0

    base = N.attr("base").outputs["Color"]
    belly = N.attr("belly").outputs["Fac"]
    scaly = N.attr("scaly").outputs["Fac"]
    membrane = N.attr("membrane").outputs["Fac"]
    plate = N.attr("plate").outputs["Fac"]

    scale_crease = N.maprange(vor_edge.outputs["Distance"], 0.0, 0.1, 0.35, 1.0)
    scale_shade = N.math("MULTIPLY", scale_crease, N.maprange(cell_rand.outputs[0], 0.0, 1.0, 0.85, 1.08))
    plate_f = N.math("FRACT", N.math("MULTIPLY", plate, 3.2))
    plate_shade = N.maprange(plate_f, 0.8, 1.0, 1.0, 0.45)
    skin = N.mix(belly, scale_shade, plate_shade)
    shade = N.mix(scaly, 1.0, skin)
    veins = N.maprange(noise.outputs["Fac"], 0.3, 0.7, 0.78, 1.15)
    shade = N.mix(membrane, shade, veins)

    ao = N.new("ShaderNodeAmbientOcclusion", samples=24)
    ao.inputs["Distance"].default_value = 0.6
    ao_shade = N.maprange(ao.outputs["AO"], 0.0, 1.0, 0.3, 1.0)
    total = N.math("MULTIPLY", shade, ao_shade)

    scale_col = N.new("ShaderNodeVectorMath", operation="SCALE")
    nt.links.new(base, scale_col.inputs[0])
    nt.links.new(total, scale_col.inputs["Scale"])
    nt.links.new(scale_col.outputs[0], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0

    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(shade, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def bake(obj):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.render.bake.margin = 6

    color_img = bpy.data.images.new("dragon_hd_color", TEX_SIZE, TEX_SIZE)
    normal_img = bpy.data.images.new("dragon_hd_normal", TEX_SIZE, TEX_SIZE)
    normal_img.colorspace_settings.name = "Non-Color"

    mat = make_bake_material()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    nt = mat.node_tree
    select_only([obj])
    for img, kind in ((color_img, "EMIT"), (normal_img, "NORMAL")):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        nt.nodes.active = node
        print(f"Baking {kind}...")
        bpy.ops.object.bake(type=kind, normal_space="TANGENT")
        img.filepath_raw = os.path.join(OUT_DIR, f"{img.name}.png")
        img.file_format = "PNG"
        img.save()

    # Final game material: baked color + normal map
    final = bpy.data.materials.new("DragonMaterial")
    fnt = final.node_tree
    bsdf = fnt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.55
    ct = fnt.nodes.new("ShaderNodeTexImage")
    ct.image = color_img
    fnt.links.new(ct.outputs["Color"], bsdf.inputs["Base Color"])
    nt_img = fnt.nodes.new("ShaderNodeTexImage")
    nt_img.image = normal_img
    nmap = fnt.nodes.new("ShaderNodeNormalMap")
    fnt.links.new(nt_img.outputs["Color"], nmap.inputs["Color"])
    fnt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    obj.data.materials.clear()
    obj.data.materials.append(final)
    bpy.data.materials.remove(mat)
    return color_img, normal_img


def unwrap(obj):
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004,
                             area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")


# ==========================================================================
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    org = build_organic()
    print(f"Organic after remesh: {tri_count(org)} tris")

    bvh = BVHTree.FromObject(org, bpy.context.evaluated_depsgraph_get())
    details = build_horns() + build_teeth(bvh) + build_eyes() + build_claws() + build_spikes(bvh)
    details += build_wing(1) + build_wing(-1)
    detail_tris = sum(tri_count(o) for o in details)
    decimate_to(org, TRI_BUDGET - detail_tris)
    print(f"Organic: {tri_count(org)} tris, details: {detail_tris} tris")
    color_organic(org)

    dragon = join([org] + details, "Dragon")
    bm = bmesh.new()
    bm.from_mesh(dragon.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(dragon.data)
    bm.free()
    for p in dragon.data.polygons:
        p.use_smooth = True

    unwrap(dragon)
    color_img, normal_img = bake(dragon)

    # Vertex data was only needed for baking; strip it so Roblox doesn't tint the mesh.
    for name in ("base", "belly", "scaly", "membrane", "plate"):
        a = dragon.data.attributes.get(name)
        if a:
            dragon.data.attributes.remove(a)

    bpy.context.scene.cursor.location = (0, 0, 0)
    select_only([dragon])
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    d = dragon.dimensions
    print(f"Dragon HD: {len(dragon.data.vertices)} verts, {tri_count(dragon)} triangles, "
          f"size {d.x:.1f} x {d.y:.1f} x {d.z:.1f}")

    color_img.pack()
    normal_img.pack()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "dragon_hd.blend"))
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT_DIR, "dragon_hd.fbx"),
                             apply_scale_options="FBX_SCALE_ALL", path_mode="COPY",
                             embed_textures=True, mesh_smooth_type="FACE")
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT_DIR, "dragon_hd.glb"), export_format="GLB")


if __name__ == "__main__":
    main()
