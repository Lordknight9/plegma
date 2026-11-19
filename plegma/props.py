# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

PART_MODES = [
    ('GLOBAL', "Γενική", "Ακολουθεί τη γενική επιλογή (Quad ή Triangle)"),
    ('QUAD', "Quad", "Remesh σε τετράγωνα"),
    ('TRI', "Triangle", "Remesh σε ισόπλευρα τρίγωνα"),
    ('KEEP', "Ως έχει", "Κράτα την αρχική γεωμετρία (π.χ. βλεφαρίδες)"),
    ('DELETE', "Αφαίρεση", "Να μη μπει στο τελικό object"),
]


class PlegmaPart(bpy.types.PropertyGroup):
    fs_id: IntProperty()
    name: StringProperty(name="Όνομα", default="Μέρος")
    color: FloatVectorProperty(name="Χρώμα", subtype='COLOR', size=3, min=0, max=1, default=(0.6, 0.6, 0.6))
    mode: EnumProperty(name="Τύπος", items=PART_MODES, default='GLOBAL')
    density: FloatProperty(name="Συντελεστής πυκνότητας", min=0.05, soft_max=5.0, default=1.0,
                           description="Πολλαπλασιαστής για τα faces αυτού του μέρους (2 = διπλάσια)")
    mark: BoolProperty(name="Επιλογή", description="Σημείωσε για συγχώνευση", default=False)
    faces: IntProperty()
    area_pct: FloatProperty()


class PlegmaSettings(bpy.types.PropertyGroup):
    # split
    split_method: EnumProperty(name="Χωρισμός με", items=[
        ('SMART', "Έξυπνο (σχήμα)", "Από το σχήμα: τα μέρη χωρίζονται στις εσοχές/πτυχώσεις (μαλλιά-μέτωπο, βλέφαρο-μάτι, γιακάς-λαιμός). Το χρώμα βοηθάει μόνο αν το θέλεις"),
        ('COLOR', "Χρώμα (texture)", "Αυτόματα από το χρώμα του texture ή του vertex color - για AI μοντέλα"),
        ('LOOSE', "Ξεχωριστά κομμάτια", "Κάθε loose part γίνεται μέρος"),
        ('MATERIAL', "Υλικά", "Κάθε υλικό γίνεται μέρος"),
        ('ANGLE', "Γωνία ακμών", "Περιοχές που χωρίζονται από αιχμηρές ακμές"),
        ('FACESETS', "Face Sets", "Χρησιμοποίησε τα Face Sets που ήδη έχεις ζωγραφίσει"),
    ], default='SMART')
    detail: FloatProperty(name="Λεπτομέρεια", min=0.0, max=1.0, default=0.5, subtype='FACTOR',
                          description="Πιο ψηλά = περισσότερα, μικρότερα μέρη (πιο ρηχές εσοχές μετράνε σαν όριο)")
    color_weight: FloatProperty(name="Βοήθεια από χρώμα", min=0.0, max=2.0, default=0.3,
                                description="0 = μόνο σχήμα. Πόσο να βοηθάει το χρώμα του texture στα όρια")
    max_parts: IntProperty(name="Μέγιστα μέρη", min=0, max=200, default=0, description="0 = αυτόματα")
    color_count: IntProperty(name="Πλήθος χρωμάτων", min=2, max=24, default=6,
                             description="Σε πόσες ομάδες χρωμάτων να χωριστεί (μαλλιά, δέρμα, μάτια...)")
    smooth_iters: IntProperty(name="Εξομάλυνση ορίων", min=0, max=30, default=6)
    min_region: FloatProperty(name="Ελάχιστο μέγεθος %", min=0.0, max=10.0, default=0.15, precision=2,
                              description="Νησίδες μικρότερες από αυτό το % της επιφάνειας ενώνονται με τη γειτονική")
    also_loose: BoolProperty(name="+ ξεχωριστά κομμάτια", default=True,
                             description="Χώρισε επιπλέον και τα κομμάτια που ήδη είναι ξεχωριστά")
    angle: FloatProperty(name="Γωνία", subtype='ANGLE', min=0.01, max=3.1, default=0.7)

    # segments (flow guides inside each part)
    seg_detail: FloatProperty(name="Λεπτομέρεια τμημάτων", min=0.0, max=1.0, default=0.4, subtype='FACTOR',
                              description="Πιο ψηλά = περισσότερα τμήματα (και πιο ήπιες ακμές γίνονται όρια)")
    seg_max: IntProperty(name="Μέγιστα τμήματα", min=0, max=500, default=0, description="0 = αυτόματα")
    seg_min: FloatProperty(name="Ελάχιστο τμήμα %", min=0.0, max=10.0, default=0.3, precision=2)

    # remesh
    mode: EnumProperty(name="Τύπος", items=[
        ('QUAD', "Quad", "Τετράγωνα (sculpt, animation, subdivision)"),
        ('TRI', "Triangle", "Ισόπλευρα τρίγωνα (3D printing, games)"),
    ], default='QUAD')
    engine: EnumProperty(name="Μηχανή quad", items=[
        ('AUTO', "Αυτόματα", "Η μηχανή του Πλέγματος (πεδίο ροής)"),
        ('GUIDED', "Πλέγμα (πεδίο ροής)", "100% quads σε λωρίδες που ακολουθούν την καμπυλότητα, τις αιχμηρές ακμές και τα τμήματα"),
        ('QUADRIFLOW', "Quadriflow", "Ενσωματωμένο στο Blender, σταθερό"),
        ('QUADWILD', "QuadWild (πειραματικό)", "Καλύτερη ροή ακμών, αλλά ακόμα δεν δουλεύει σωστά σε αυτό το PC"),
    ], default='AUTO')
    target_faces: IntProperty(name="Στόχος faces", min=50, soft_max=200000, default=8000,
                              description="Πόσα faces περίπου θα έχει το τελικό μοντέλο (μοιράζονται ανά εμβαδό)")
    sym_x: BoolProperty(name="X", default=False, description="Συμμετρία στον άξονα X")
    sym_y: BoolProperty(name="Y", default=False, description="Συμμετρία στον άξονα Y")
    sym_z: BoolProperty(name="Z", default=False, description="Συμμετρία στον άξονα Z")
    sym_space: EnumProperty(name="Άξονες", items=[
        ('LOCAL', "Τοπικοί (object)", "Οι άξονες και το origin του ίδιου του object"),
        ('GLOBAL', "Global (κόσμος)", "Οι άξονες του κόσμου, με κέντρο το 0,0,0"),
        ('CURSOR', "3D Cursor", "Η θέση και η περιστροφή του 3D cursor"),
        ('OBJECT', "Άλλο object", "Η θέση και η περιστροφή ενός άλλου object, π.χ. ενός Empty"),
    ], default='LOCAL')
    sym_object: PointerProperty(name="Object άξονα", type=bpy.types.Object,
                                description="Το object (π.χ. Empty) που ορίζει το επίπεδο συμμετρίας")
    sharp_angle: FloatProperty(name="Αιχμηρές ακμές °", min=0, max=180, default=35)
    keep_sharp: BoolProperty(name="Κράτα αιχμηρές ακμές", default=True)
    tri_iterations: IntProperty(name="Επαναλήψεις τριγώνων", min=1, max=20, default=6)
    transfer: BoolProperty(name="Μεταφορά UV & χρωμάτων", default=True)
    vertex_groups: BoolProperty(name="Vertex groups ανά μέρος", default=True)
    hide_source: BoolProperty(name="Κρύψε το αρχικό", default=True)
    show_advanced: BoolProperty(name="Περισσότερα", default=False)


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
