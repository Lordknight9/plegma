# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

from .engine import quadwild
from . import jobs
from .props import STATUS_ICON


class PLEGMA_UL_parts(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        row.prop(item, "mark", text="")
        row.label(text="", icon=STATUS_ICON[item.status])
        sw = row.row(align=True)
        sw.scale_x = 0.35
        sw.prop(item, "color", text="")
        row.prop(item, "name", text="", emboss=False)
        sub = row.row(align=True)
        sub.scale_x = 0.9
        sub.prop(item, "mode", text="")
        row.label(text=f"{item.area_pct:.1f}%")
        op = row.operator("plegma.select_part", text="", icon='RESTRICT_SELECT_OFF', emboss=False)
        op.index = index


class PLEGMA_PT_main(bpy.types.Panel):
    bl_label = "Plegma"
    bl_idname = "PLEGMA_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Plegma"

    def draw(self, ctx):
        layout = self.layout
        ob = ctx.active_object
        if ob is None or ob.type != 'MESH':
            layout.label(text="Select a mesh object", icon='INFO')
            return
        layout.label(text=f"{ob.name}: {len(ob.data.polygons):,} faces", icon='MESH_DATA')


class _Sub:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Plegma"
    bl_parent_id = "PLEGMA_PT_main"

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH'


class PLEGMA_PT_split(_Sub, bpy.types.Panel):
    bl_label = "1. Split into parts"

    def draw(self, ctx):
        layout = self.layout
        s = ctx.scene.plegma
        ob = jobs.source_of(ctx.active_object)
        col = layout.column(align=True)
        col.prop(s, "split_method", text="")
        if s.split_method == 'AUTO':
            col.prop(s, "detail", slider=True)
            col.prop(s, "min_region")
            col.prop(s, "also_loose")
        elif s.split_method == 'SMART':
            col.prop(s, "detail", slider=True)
            col.prop(s, "color_weight")
            col.prop(s, "max_parts")
            col.prop(s, "min_region")
            col.prop(s, "also_loose")
        elif s.split_method == 'COLOR':
            col.prop(s, "color_count")
            col.prop(s, "smooth_iters")
            col.prop(s, "min_region")
            col.prop(s, "also_loose")
        elif s.split_method == 'ANGLE':
            col.prop(s, "angle")
            col.prop(s, "min_region")
        layout.operator("plegma.split", icon='MOD_EXPLODE')

        if len(ob.plegma_parts):
            layout.template_list("PLEGMA_UL_parts", "", ob, "plegma_parts", ob, "plegma_index", rows=6)
            if 0 <= ob.plegma_index < len(ob.plegma_parts):
                p = ob.plegma_parts[ob.plegma_index]
                layout.box().prop(p, "density")
            row = layout.row(align=True)
            row.operator("plegma.merge", icon='AUTOMERGE_ON')
            row.operator("plegma.refresh", icon='FILE_REFRESH')
            row = layout.row(align=True)
            row.operator("plegma.preview", text="Colours", icon='COLOR').on = True
            row.operator("plegma.preview", text="Normal", icon='SHADING_SOLID').on = False
        row = layout.row(align=True)
        row.operator("plegma.sculpt", icon='BRUSH_DATA')
        row.operator("plegma.assign", icon='ADD')

        box = layout.box()
        box.label(text="Segments (edge-flow guides)", icon='MOD_EDGESPLIT')
        col = box.column(align=True)
        col.prop(s, "seg_detail", slider=True)
        col.prop(s, "seg_max")
        col.prop(s, "seg_min")
        box.operator("plegma.segments", icon='MOD_TRIANGULATE')
        if ob.data.attributes.get("plegma_segment") is not None:
            row = box.row(align=True)
            row.operator("plegma.preview_segments", text="Show segments", icon='COLOR')
            row.operator("plegma.clear_segments", text="Clear", icon='X')


class PLEGMA_PT_remesh(_Sub, bpy.types.Panel):
    bl_label = "2. Remesh"

    def draw(self, ctx):
        layout = self.layout
        s = ctx.scene.plegma
        layout.row().prop(s, "mode", expand=True)
        layout.row().prop(s, "density_mode", expand=True)
        col = layout.column(align=True)
        if s.density_mode == 'AUTO':
            col.row().prop(s, "detail_level", expand=True)
            col.prop(s, "max_faces")
        else:
            col.prop(s, "target_faces")
        if s.mode == 'QUAD':
            col.prop(s, "engine", text="Engine")
            if s.engine == 'QUADWILD' and not quadwild.is_installed():
                box = layout.box()
                box.label(text="QuadWild is not downloaded", icon='ERROR')
                box.operator("plegma.install_engine", icon='IMPORT')
        else:
            col.prop(s, "tri_iterations")
        box = layout.box()
        row = box.row(align=True)
        row.label(text="Symmetry")
        row.prop(s, "sym_x", toggle=True)
        row.prop(s, "sym_y", toggle=True)
        row.prop(s, "sym_z", toggle=True)
        if s.sym_x or s.sym_y or s.sym_z:
            box.prop(s, "sym_space", text="")
            if s.sym_space == 'OBJECT':
                box.prop(s, "sym_object", text="")
        col = layout.column(align=True)
        col.prop(s, "keep_sharp")
        if s.keep_sharp:
            col.prop(s, "sharp_angle")
        layout.prop(s, "show_advanced", icon='TRIA_DOWN' if s.show_advanced else 'TRIA_RIGHT', emboss=False)
        if s.show_advanced:
            col = layout.column(align=True)
            col.prop(s, "transfer")
            col.prop(s, "vertex_groups")
            col.prop(s, "hide_source")
            col.prop(s, "mem_limit_mb")


class PLEGMA_PT_queue(_Sub, bpy.types.Panel):
    bl_label = "3. Queue"

    def draw(self, ctx):
        layout = self.layout
        s = ctx.scene.plegma
        ob = jobs.source_of(ctx.active_object)
        n = len(ob.plegma_parts)
        if s.queue_running:
            layout.label(text="Running... press Esc to stop", icon='SORTTIME')
        if n:
            c = jobs.counts(ob)
            row = layout.row()
            row.label(text=f"{c['PENDING'] + c['REDO']} pending", icon='TIME')
            row.label(text=f"{c['DONE']} done", icon='CHECKMARK')
            row.label(text=f"{c['APPROVED']} locked", icon='LOCKED')
        r = layout.row(align=True)
        r.scale_y = 1.5
        r.operator("plegma.process", text="Next part", icon='TRACKING_FORWARDS').which = 'NEXT'
        r.operator("plegma.process", text="All pending", icon='PLAY').which = 'ALL'
        if n and 0 <= ob.plegma_index < len(ob.plegma_parts):
            p = ob.plegma_parts[ob.plegma_index]
            box = layout.box()
            box.label(text=p.name, icon='DOT')
            row = box.row(align=True)
            row.operator("plegma.process", text="Process", icon='MOD_REMESH').which = 'ONE'
            row2 = box.row(align=True)
            if p.status == 'APPROVED':
                row2.operator("plegma.part_action", text="Unlock", icon='UNLOCKED').action = 'UNLOCK'
            else:
                row2.operator("plegma.part_action", text="Approve", icon='CHECKMARK').action = 'APPROVE'
            row2.operator("plegma.part_action", text="Redo", icon='FILE_REFRESH').action = 'REDO'
            if p.res_faces:
                box.label(text=f"{p.res_faces:,} faces")
        row = layout.row(align=True)
        row.operator("plegma.part_action", text="Approve all done", icon='CHECKMARK').action = 'APPROVE_ALL'
        layout.operator("plegma.build", icon='OUTLINER_COLLECTION')
        row = layout.row(align=True)
        row.label(text=f"Cache: {jobs.cache_size_mb():.1f} MB", icon='DISK_DRIVE')
        row.operator("plegma.clear_cache", text="Clear", icon='TRASH')


classes = (PLEGMA_UL_parts, PLEGMA_PT_main, PLEGMA_PT_split, PLEGMA_PT_remesh, PLEGMA_PT_queue)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
