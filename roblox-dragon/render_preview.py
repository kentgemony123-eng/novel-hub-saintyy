"""Render preview images of dragon_hd.blend (python render_preview.py)."""

import math
import os

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "dragon_hd.blend"))
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 48
sc.render.resolution_x, sc.render.resolution_y = 1000, 760

world = bpy.data.worlds.new("sky")
sc.world = world
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.68, 0.85, 1)
bg.inputs[1].default_value = 0.7
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sc.collection.objects.link(sun)
sun.data.energy = 4.0
sun.data.angle = math.radians(8)
sun.rotation_euler = (math.radians(45), math.radians(10), math.radians(-35))
bpy.ops.mesh.primitive_plane_add(size=80)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam

shots = {
    "front34": ((-14, -17, 10), (0, 0.5, 3.5), 50),
    "side": ((-24, 0.5, 5), (0, 0.5, 3.5), 50),
    "head": ((-2.6, -7.4, 7.2), (0, -4.9, 6.35), 50),
}
for name, (pos, target, lens) in shots.items():
    cam.location = pos
    cam.data.lens = lens
    cam.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(HERE, f"preview_hd_{name}.png")
    bpy.ops.render.render(write_still=True)
