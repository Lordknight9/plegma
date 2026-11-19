# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

from .engine import quadwild


class PLEGMA_UL_parts(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        row.prop(item, "mark", text="")
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
    bl_label = "Πλέγμα"
    bl_idname = "PLEGMA_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Πλέγμα"

    def draw(self, ctx):
        layout = self.layout
        ob = ctx.active_object
        if ob is None or ob.type != 'MESH':
            layout.label(text="Διάλεξε ένα mesh object", icon='INFO')
            return
        layout.label(text=f"{ob.name}: {len(ob.data.polygons):,} faces", icon='MESH_DATA')


class PLEGMA_PT_split(bpy.types.Panel):
    bl_label = "1. Χωρισμός σε μέρη"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Πλέγμα"
    bl_parent_id = "PLEGMA_PT_main"

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH'

    def draw(self, ctx):
        layout = self.layout
        s = ctx.scene.plegma
        ob = ctx.active_object
        col = layout.column(align=True)
        col.prop(s, "split_method", text="")
        if s.split_method == 'SMART':
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
                box = layout.box()
                box.prop(p, "density")
            row = layout.row(align=True)
            row.operator("plegma.merge", icon='AUTOMERGE_ON')
            row.operator("plegma.refresh", icon='FILE_REFRESH')
            row = layout.row(align=True)
            row.operator("plegma.preview", text="Χρώματα", icon='COLOR').on = True
            row.operator("plegma.preview", text="Κανονικά", icon='SHADING_SOLID').on = False
        row = layout.row(align=True)
        row.operator("plegma.sculpt", icon='BRUSH_DATA')
        row.operator("plegma.assign", icon='ADD')

        box = layout.box()
        box.label(text="Τμήματα (οδηγοί για τις ακμές)", icon='MOD_EDGESPLIT')
        col = box.column(align=True)
        col.prop(s, "seg_detail", slider=True)
        col.prop(s, "seg_max")
        col.prop(s, "seg_min")
        box.operator("plegma.segments", icon='MOD_TRIANGULATE')
        if ob.data.attributes.get("plegma_segment") is not None:
            row = box.row(align=True)
            row.operator("plegma.preview_segments", text="Δείξε τμήματα", icon='COLOR')
            row.operator("plegma.clear_segments", text="Καθάρισε", icon='X')


class PLEGMA_PT_remesh(bpy.types.Panel):
    bl_label = "2. Remesh"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Πλέγμα"
    bl_parent_id = "PLEGMA_PT_main"

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH'

    def draw(self, ctx):
        layout = self.layout
        s = ctx.scene.plegma
        layout.row().prop(s, "mode", expand=True)
        col = layout.column(align=True)
        col.prop(s, "target_faces")
        if s.mode == 'QUAD':
            col.prop(s, "engine", text="Μηχανή")
            if s.engine == 'QUADWILD' and not quadwild.is_installed():
                box = layout.box()
                box.label(text="QuadWild: δεν έχει κατέβει", icon='ERROR')
                box.operator("plegma.install_engine", icon='IMPORT')
        else:
            col.prop(s, "tri_iterations")
        box = layout.box()
        row = box.row(align=True)
        row.label(text="Συμμετρία")
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
        r = layout.row()
        r.scale_y = 1.6
        r.operator("plegma.remesh", icon='MOD_REMESH')


classes = (PLEGMA_UL_parts, PLEGMA_PT_main, PLEGMA_PT_split, PLEGMA_PT_remesh)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
