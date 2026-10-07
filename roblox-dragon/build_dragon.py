"""
Procedural low-poly dragon for Roblox.

Run inside Blender (Scripting tab -> Open -> Run Script), or headless:
    blender --background --python build_dragon.py
    python build_dragon.py            (with `pip install bpy`)

Outputs (next to this script):
    dragon.blend          - editable Blender file
    dragon.fbx            - import into Roblox Studio (3D Importer)
    dragon.glb            - alternative import format (texture embedded)
    dragon_palette.png    - color palette texture used by the mesh

The dragon is built as ONE mesh using ONE small palette texture, so it imports
into Roblox as a single MeshPart. It faces -Y in Blender (front view).
"""

import math
import os

import bpy  # noqa: I001 - bpy must be imported before bmesh outside Blender
import bmesh
from mathutils import Vector

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Palette: each color is a cell in a 4x4 texture. Faces get UVs at the
# center of their color's cell, so the whole model uses a single texture.
# --------------------------------------------------------------------------
PALETTE = {
    "body":     (0.62, 0.08, 0.06),
    "belly":    (0.95, 0.78, 0.45),
    "membrane": (0.85, 0.32, 0.10),
    "wingbone": (0.40, 0.05, 0.04),
    "horn":     (0.92, 0.88, 0.74),
    "eye":      (1.00, 0.82, 0.08),
    "pupil":    (0.03, 0.03, 0.03),
    "spike":    (0.30, 0.04, 0.03),
}
GRID = 4
CELL_PX = 16
TEX_SIZE = GRID * CELL_PX
PALETTE_INDEX = {name: i for i, name in enumerate(PALETTE)}


def cell_uv(name):
    i = PALETTE_INDEX[name]
    col, row = i % GRID, i // GRID
    return ((col + 0.5) / GRID, (row + 0.5) / GRID)


def make_palette_image():
    img = bpy.data.images.new("dragon_palette", TEX_SIZE, TEX_SIZE, alpha=False)
    px = [0.0] * (TEX_SIZE * TEX_SIZE * 4)
    for name, rgb in PALETTE.items():
        i = PALETTE_INDEX[name]
        cx, cy = (i % GRID) * CELL_PX, (i // GRID) * CELL_PX
        for y in range(cy, cy + CELL_PX):
            for x in range(cx, cx + CELL_PX):
                o = (y * TEX_SIZE + x) * 4
                px[o:o + 4] = [rgb[0], rgb[1], rgb[2], 1.0]
    img.pixels = px
    img.filepath_raw = os.path.join(OUT_DIR, "dragon_palette.png")
    img.file_format = "PNG"
    img.save()
    return img


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def apply_all_modifiers(obj):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def tag_color(obj, name):
    """Store the palette color per face in a custom int attribute."""
    attr = obj.data.attributes.get("pal") or obj.data.attributes.new("pal", "INT", "FACE")
    for i in range(len(obj.data.polygons)):
        attr.data[i].value = PALETTE_INDEX[name]


def skin_tube(name, verts, edges, radii, color, subsurf=1):
    """Build an organic tube from a vertex skeleton using the Skin modifier."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, [])
    obj = link(bpy.data.objects.new(name, me))
    obj.modifiers.new("skin", "SKIN")
    skin = me.skin_vertices[0].data
    for i, r in enumerate(radii):
        skin[i].radius = r if isinstance(r, tuple) else (r, r)
    skin[0].use_root = True
    if subsurf:
        s = obj.modifiers.new("sub", "SUBSURF")
        s.levels = subsurf
    apply_all_modifiers(obj)
    tag_color(obj, color)
    return obj


def cone(name, base, tip, radius, color, verts=6):
    base, tip = Vector(base), Vector(tip)
    direction = tip - base
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts,
                          radius1=radius, radius2=0.0, depth=direction.length)
    # create_cone is centred on origin along +Z; move base to origin
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, direction.length / 2))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(name, me))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.location = base
    tag_color(obj, color)
    return obj


def ellipsoid(name, center, scale, color, segs=10, rings=6):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(name, me))
    obj.location = center
    obj.scale = scale
    tag_color(obj, color)
    return obj


def surface_point(obj, origin, direction):
    """Where a ray from `origin` first hits `obj` (object has identity transform)."""
    hit, loc, _normal, _i = obj.ray_cast(Vector(origin), Vector(direction))
    return loc if hit else Vector(origin)


# --------------------------------------------------------------------------
# Dragon parts
# --------------------------------------------------------------------------
def build_body():
    # Spine from tail tip to nose, plus four legs branching off.
    spine = [
        ((0, 9.5, 1.0), 0.06),
        ((0, 8.0, 1.3), 0.22),
        ((0, 6.4, 1.8), 0.40),
        ((0, 4.8, 2.4), 0.65),
        ((0, 3.2, 2.9), 0.95),
        ((0, 1.8, 3.1), 1.20),  # 5 hips
        ((0, 0.0, 3.25), 1.35),
        ((0, -1.6, 3.35), 1.25),  # 7 chest
        ((0, -2.7, 3.9), 0.80),
        ((0, -3.2, 4.8), 0.58),
        ((0, -3.5, 5.7), 0.50),
        ((0, -3.9, 6.35), (0.55, 0.50)),  # 11 back of skull
        ((0, -4.4, 6.45), (0.60, 0.55)),
        ((0, -5.1, 6.30), (0.42, 0.34)),  # snout
        ((0, -5.7, 6.20), (0.30, 0.24)),  # nose
    ]
    verts = [p for p, _ in spine]
    radii = [r for _, r in spine]
    edges = [(i, i + 1) for i in range(len(spine) - 1)]

    hip, chest = 5, 7
    back_leg = [((1.0, 1.8, 2.5), 0.62), ((1.25, 1.2, 1.4), 0.45),
                ((1.25, 1.7, 0.45), 0.30), ((1.25, 1.1, 0.18), (0.38, 0.20))]
    front_leg = [((0.95, -1.6, 2.7), 0.50), ((1.15, -1.9, 1.5), 0.36),
                 ((1.15, -1.7, 0.42), 0.26), ((1.15, -2.2, 0.16), (0.34, 0.18))]
    for side in (1, -1):
        for root, leg in ((hip, back_leg), (chest, front_leg)):
            prev = root
            for (x, y, z), r in leg:
                verts.append((x * side, y, z))
                radii.append(r)
                edges.append((prev, len(verts) - 1))
                prev = len(verts) - 1

    body = skin_tube("Body", verts, edges, radii, "body", subsurf=1)

    # Downward-facing faces on the torso/neck/tail become belly-colored.
    attr = body.data.attributes["pal"]
    for poly in body.data.polygons:
        c = poly.center
        if poly.normal.z < -0.45 and c.z > 1.6 and abs(c.x) < 1.0:
            attr.data[poly.index].value = PALETTE_INDEX["belly"]
    return body


def build_head_details(body):
    parts = []
    # Lower jaw
    parts.append(ellipsoid("Jaw", (0, -5.0, 6.0), (0.36, 0.75, 0.16), "belly", 8, 5))
    for s in (1, -1):
        # Eyes + pupils
        parts.append(ellipsoid("Eye", (0.40 * s, -4.65, 6.70), (0.15, 0.15, 0.13), "eye", 8, 5))
        parts.append(ellipsoid("Pupil", (0.50 * s, -4.72, 6.71), (0.05, 0.06, 0.10), "pupil", 6, 4))
        # Horns sweeping back
        parts.append(cone("Horn", (0.32 * s, -4.1, 6.85), (0.55 * s, -2.9, 7.7), 0.17, "horn"))
        parts.append(cone("HornSmall", (0.50 * s, -4.3, 6.65), (0.95 * s, -3.6, 6.9), 0.09, "horn"))
        # Nostrils bumps
        n = surface_point(body, (0.10 * s, -9.0, 6.30), (0, 1, 0))
        parts.append(ellipsoid("Nostril", n + Vector((0, 0.02, 0)), (0.07, 0.07, 0.05), "spike", 6, 4))
    # Teeth
    for s in (1, -1):
        for y in (-5.25, -5.55):
            parts.append(cone("Tooth", (0.24 * s, y, 6.12), (0.24 * s, y - 0.03, 5.92), 0.05, "horn", 4))
    return parts


def build_spikes(body):
    parts = []
    # (y, height) along the spine from neck to tail; z is found on the back surface
    ridge = [(-3.0, 0.30), (-2.4, 0.40), (-1.6, 0.50), (-0.8, 0.58),
             (0.0, 0.62), (0.8, 0.60), (1.6, 0.55), (2.4, 0.50),
             (3.2, 0.45), (4.0, 0.40), (4.8, 0.36), (5.6, 0.31),
             (6.4, 0.27), (7.2, 0.22), (8.0, 0.18), (8.7, 0.14)]
    for y, h in ridge:
        top = surface_point(body, (0, y, 20), (0, 0, -1))
        base = top - Vector((0, 0, 0.06))
        parts.append(cone("Spike", base, base + Vector((0, h * 0.6, h)), h * 0.45, "spike", 4))
    # Tail fin (arrow head)
    parts.append(cone("TailFin", (0, 9.2, 1.1), (0, 10.4, 0.9), 0.55, "spike", 3))
    return parts


def build_claws():
    parts = []
    feet = [(1.25, 1.1, 0.18), (1.15, -2.2, 0.16)]
    for s in (1, -1):
        for fx, fy, fz in feet:
            for dx in (-0.16, 0.0, 0.16):
                base = (fx * s + dx, fy - 0.12, 0.12)
                tip = (fx * s + dx * 1.4, fy - 0.45, 0.0)
                parts.append(cone("Claw", base, tip, 0.07, "horn", 4))
    return parts


def build_wing(side):
    s = side
    shoulder = (0.75 * s, -1.2, 4.1)
    elbow = (3.0 * s, -0.6, 5.7)
    wrist = (5.2 * s, 0.3, 6.3)
    fingers = [(8.0 * s, 1.4, 4.9), (7.4 * s, 3.0, 3.8),
               (5.6 * s, 3.8, 3.4), (3.4 * s, 3.4, 3.6)]
    body_attach = (0.75 * s, 1.2, 3.9)

    parts = []
    # Wing arm + finger bones
    bverts = [shoulder, elbow, wrist] + fingers
    bedges = [(0, 1), (1, 2)] + [(2, 3 + i) for i in range(len(fingers))]
    bradii = [0.22, 0.17, 0.14, 0.05, 0.05, 0.05, 0.05]
    parts.append(skin_tube("WingBones", bverts, bedges, bradii, "wingbone", subsurf=0))
    parts.append(cone("WingClaw", wrist, (wrist[0] + 0.3 * s, wrist[1] - 0.4, wrist[2] + 0.45), 0.08, "horn", 4))

    # Membrane: scalloped edge between finger tips, filled as a fan from the wrist
    def scallop(a, b, depth=0.28):
        a, b, w = Vector(a), Vector(b), Vector(wrist)
        mid = (a + b) / 2
        return tuple(mid + (w - mid) * depth)

    outline = [elbow, fingers[0]]
    for i in range(1, len(fingers)):
        outline.append(scallop(fingers[i - 1], fingers[i]))
        outline.append(fingers[i])
    outline.append(scallop(fingers[-1], body_attach, 0.15))
    outline.append(body_attach)
    outline.append(shoulder)

    mverts = [wrist] + outline
    faces = [(0, i, i + 1) for i in range(1, len(mverts) - 1)]
    faces.append((0, len(mverts) - 1, 1))  # close shoulder -> elbow via wrist
    me = bpy.data.meshes.new("Membrane")
    me.from_pydata(mverts, [], faces)
    me.validate()
    mem = link(bpy.data.objects.new("Membrane", me))
    sol = mem.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.06
    sol.offset = 0
    apply_all_modifiers(mem)
    tag_color(mem, "membrane")
    parts.append(mem)
    return parts


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def assign_palette_uvs(obj):
    me = obj.data
    uv = me.uv_layers.new(name="UVMap")
    pal = me.attributes["pal"]
    names = list(PALETTE)
    for poly in me.polygons:
        u, v = cell_uv(names[pal.data[poly.index].value])
        for li in poly.loop_indices:
            uv.data[li].uv = (u, v)


def make_material(img):
    mat = bpy.data.materials.new("DragonMaterial")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.7
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def main():
    reset_scene()
    img = make_palette_image()

    body = build_body()
    parts = [body]
    parts += build_head_details(body)
    parts += build_spikes(body)
    parts += build_claws()
    parts += build_wing(1)
    parts += build_wing(-1)

    dragon = join(parts, "Dragon")

    # Clean up: merge duplicate verts, recalc normals, flat low-poly shading
    bm = bmesh.new()
    bm.from_mesh(dragon.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(dragon.data)
    bm.free()
    for p in dragon.data.polygons:
        p.use_smooth = False

    assign_palette_uvs(dragon)
    dragon.data.materials.append(make_material(img))

    # Put origin at the feet (ground level, centred) - handy in Roblox.
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    tris = sum(len(p.vertices) - 2 for p in dragon.data.polygons)
    dims = dragon.dimensions
    print(f"Dragon: {len(dragon.data.vertices)} verts, {tris} triangles, "
          f"size {dims.x:.1f} x {dims.y:.1f} x {dims.z:.1f}")

    img.pack()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "dragon.blend"))

    bpy.ops.export_scene.fbx(
        filepath=os.path.join(OUT_DIR, "dragon.fbx"),
        use_selection=False,
        apply_scale_options="FBX_SCALE_ALL",
        path_mode="COPY",
        embed_textures=True,
        mesh_smooth_type="FACE",
    )
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(OUT_DIR, "dragon.glb"),
        export_format="GLB",
    )


if __name__ == "__main__":
    main()
