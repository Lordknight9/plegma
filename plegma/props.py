# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

PART_MODES = [
    ('GLOBAL', "Global", "Follow the global choice (Quad or Triangle)"),
    ('QUAD', "Quad", "Remesh to quads"),
    ('TRI', "Triangle", "Remesh to even triangles"),
    ('KEEP', "Keep as is", "Keep the original geometry (e.g. eyelashes)"),
    ('DELETE', "Remove", "Leave out of the final object"),
]

# status of a part in the queue
STATUS = [
    ('PENDING', "Pending", "Waiting to be remeshed", 'TIME', 0),
    ('DONE', "Done", "Remeshed, waiting for your approval", 'CHECKMARK', 1),
    ('APPROVED', "Approved", "Locked: never recomputed unless you press Redo", 'LOCKED', 2),
    ('REDO', "Redo", "Marked to be computed again", 'FILE_REFRESH', 3),
]
STATUS_ICON = {s[0]: s[3] for s in STATUS}


class PlegmaPart(bpy.types.PropertyGroup):
    fs_id: IntProperty()
    name: StringProperty(name="Name", default="Part")
    color: FloatVectorProperty(name="Color", subtype='COLOR', size=3, min=0, max=1, default=(0.6, 0.6, 0.6))
    mode: EnumProperty(name="Type", items=PART_MODES, default='GLOBAL')
    density: FloatProperty(name="Density factor", min=0.05, soft_max=5.0, default=1.0,
                           description="Multiplier for the face count of this part (2 = twice as many)")
    mark: BoolProperty(name="Mark", description="Mark for merging", default=False)
    faces: IntProperty()
    area_pct: FloatProperty()
    status: EnumProperty(name="Status", items=[s[:4] + (s[4],) for s in STATUS], default='PENDING')
    geo_key: StringProperty()      # hash of the part geometry the result was made from
    cache_key: StringProperty()    # cache file of the result (geometry + settings)
    res_faces: IntProperty()


class PlegmaSettings(bpy.types.PropertyGroup):
    # split
    split_method: EnumProperty(name="Split by", items=[
        ('SMART', "Smart (shape)", "From the shape: parts are split at creases and grooves. Colour only helps if you want it to"),
        ('COLOR', "Colour (texture)", "Automatically from the texture or vertex colour - useful for AI models"),
        ('LOOSE', "Loose parts", "Every loose piece becomes a part"),
        ('MATERIAL', "Materials", "Every material becomes a part"),
        ('ANGLE', "Edge angle", "Regions separated by sharp edges"),
        ('FACESETS', "Face Sets", "Use the Face Sets you already painted"),
    ], default='SMART')
    detail: FloatProperty(name="Detail", min=0.0, max=1.0, default=0.5, subtype='FACTOR',
                          description="Higher = more, smaller parts (shallower creases count as borders)")
    color_weight: FloatProperty(name="Colour help", min=0.0, max=2.0, default=0.3,
                                description="0 = shape only. How much the texture colour helps with the borders")
    max_parts: IntProperty(name="Max parts", min=0, max=200, default=0, description="0 = automatic")
    color_count: IntProperty(name="Colour count", min=2, max=24, default=6,
                             description="Number of colour groups (hair, skin, eyes...)")
    smooth_iters: IntProperty(name="Border smoothing", min=0, max=30, default=6)
    min_region: FloatProperty(name="Minimum size %", min=0.0, max=10.0, default=0.15, precision=2,
                              description="Islands smaller than this % of the surface merge into a neighbour")
    also_loose: BoolProperty(name="+ loose pieces", default=True,
                             description="Also split pieces that are already separate")
    angle: FloatProperty(name="Angle", subtype='ANGLE', min=0.01, max=3.1, default=0.7)

    # segments (flow guides inside each part)
    seg_detail: FloatProperty(name="Segment detail", min=0.0, max=1.0, default=0.4, subtype='FACTOR',
                              description="Higher = more segments (gentler edges become borders)")
    seg_max: IntProperty(name="Max segments", min=0, max=500, default=0, description="0 = automatic")
    seg_min: FloatProperty(name="Minimum segment %", min=0.0, max=10.0, default=0.3, precision=2)

    # remesh
    mode: EnumProperty(name="Type", items=[
        ('QUAD', "Quad", "Quads (sculpting, animation, subdivision)"),
        ('TRI', "Triangle", "Even triangles (3D printing, games)"),
    ], default='QUAD')
    engine: EnumProperty(name="Quad engine", items=[
        ('AUTO', "Automatic", "The Plegma flow-field engine"),
        ('GUIDED', "Plegma (flow field)", "Quads in strips that follow curvature, sharp edges and segments"),
        ('QUADRIFLOW', "Quadriflow", "Built into Blender, stable"),
        ('QUADWILD', "QuadWild (experimental)", "Better edge flow, but still unstable on this PC"),
    ], default='AUTO')
    target_faces: IntProperty(name="Target faces", min=50, soft_max=200000, default=8000,
                              description="Approximate face count of the whole model (shared between parts by area)")
    sym_x: BoolProperty(name="X", default=False, description="Symmetry on the X axis")
    sym_y: BoolProperty(name="Y", default=False, description="Symmetry on the Y axis")
    sym_z: BoolProperty(name="Z", default=False, description="Symmetry on the Z axis")
    sym_space: EnumProperty(name="Axes", items=[
        ('LOCAL', "Local (object)", "Axes and origin of the object itself"),
        ('GLOBAL', "Global (world)", "World axes, centred on 0,0,0"),
        ('CURSOR', "3D Cursor", "Position and rotation of the 3D cursor"),
        ('OBJECT', "Other object", "Position and rotation of another object, e.g. an Empty"),
    ], default='LOCAL')
    sym_object: PointerProperty(name="Axis object", type=bpy.types.Object,
                                description="The object (e.g. an Empty) that defines the symmetry plane")
    sharp_angle: FloatProperty(name="Sharp edge angle", min=0, max=180, default=35)
    keep_sharp: BoolProperty(name="Keep sharp edges", default=True)
    tri_iterations: IntProperty(name="Triangle iterations", min=1, max=20, default=6)
    transfer: BoolProperty(name="Transfer UV & colours", default=True)
    vertex_groups: BoolProperty(name="Vertex group per part", default=True)
    hide_source: BoolProperty(name="Hide the source", default=True)
    mem_limit_mb: IntProperty(name="Memory limit (MB)", min=256, soft_max=16384, default=2048,
                              description="Memory the C++ core may use for one part. Bigger parts are simplified first")
    show_advanced: BoolProperty(name="More", default=False)
    queue_running: BoolProperty(default=False)


classes = (PlegmaPart, PlegmaSettings)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.plegma = PointerProperty(type=PlegmaSettings)
    bpy.types.Object.plegma_parts = CollectionProperty(type=PlegmaPart)
    bpy.types.Object.plegma_index = IntProperty(default=0)


def unregister():
    del bpy.types.Object.plegma_index
    del bpy.types.Object.plegma_parts
    del bpy.types.Scene.plegma
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
