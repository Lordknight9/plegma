# Plegma

Add-on για το **Blender 5.2** που κάνει **quad remesh** σε μοντέλα (κυρίως AI, π.χ. Meshy) και τα χωρίζει σε **καθαρά τμήματα**, σαν να βγήκαν από το Fusion. Κάθε τμήμα είναι ξεχωριστό νησί στο ίδιο object (στο Edit Mode το πιάνεις με `L`), με ομοιόμορφα quads που ακολουθούν τη ροή του.

Το UI είναι στα αγγλικά. Για προσωπική χρήση.

## Τι κάνει (σήμερα, v0.5)

- **Parts:** αυτόματος χωρισμός σε μέρη από σχήμα, χρώμα, υλικά, Face Sets ή ξεχωριστά κομμάτια, με διόρθωση από τον χρήστη (merge, ζωγράφισμα).
- **Segments:** οδηγοί για τις ακμές μέσα σε κάθε μέρος.
- **Remesh:** quads με πεδίο ροής (μέθοδος τύπου Instant Meshes) ή τρίγωνα, συμμετρία X/Y/Z, μεταφορά UV/χρωμάτων.

Ο σχεδιασμός της επόμενης έκδοσης (πυρήνας C++, τμηματική επεξεργασία, cache, ιστορικό) είναι στο [DESIGN.md](DESIGN.md).

## Δομή

```
plegma/            <- το ίδιο το add-on (Blender extension)
  __init__.py, props.py, ui.py, operators.py
  segment.py, smart.py         χωρισμός σε μέρη / τμήματα
  fieldquad.py, guided.py      μηχανές remesh
  remesh_core.py               βοηθητικά (συμμετρία, μεταφορά UV)
  engine/                      QuadWild (πειραματικό)
DESIGN.md          <- σχεδιασμός (Βήμα 3)
```

## Πώς τρέχει (σε κάθε PC, μία φορά)

1. Κλωνοποίηση του repo στο `Documents\Claude`:
   ```powershell
   cd $HOME\Documents\Claude
   git clone https://github.com/Lordknight9/plegma
   ```
2. Στο Blender: **Edit → Preferences → Get Extensions → Repositories (▾) → + → Add Local Repository**, διαδρομή `Documents\Claude\plegma`.
3. Ενεργοποίηση του **Plegma** στη λίστα. Το πάνελ είναι στο **N → Plegma**.

Όταν αλλάζει ο κώδικας (`git pull`), αρκεί **Reload** στο add-on. Δεν χρειάζεται νέα εγκατάσταση.

## Άδεια

GPL-3.0-or-later (λόγω του προαιρετικού QuadWild). Η μέθοδος του πεδίου ροής βασίζεται στο Instant Meshes (Jakob et al., 2015).
