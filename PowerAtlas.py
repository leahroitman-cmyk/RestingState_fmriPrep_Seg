"""
Power-264 tidsserie- och korrelations-extraktion från fMRIPrep (MNI)

Författare: Lea Roitman
Datum: 2025-10-27
Projekt: fMRIPrep 
kontakt: leah.roitman@gmail.com

Kort beskrivning:
- Läser preproc BOLD + confounds + mask
- Extraherar ROI-tidsserier (labels/spheres), regresserar confounds
- Sparar tidsserier, korrelationsmatris och metadata

Indata (förväntade filer)
-------------------------
- <DERIV>/func/*space-MNI*desc-preproc_bold.nii[.gz]
- <DERIV>/func/*desc-confounds_timeseries.tsv
- <DERIV>/func/*space-MNI*desc-brain_mask.nii[.gz]
(alt. i ses-*/func/)

Viktiga parametrar
------------------
- PIPELINE: "labels" eller "spheres"
- LABELS_SMOOTH_FWHM (labels) / SMOOTH_FWHM (spheres): smoothing i mm (None = ingen)
- SUB, BASE/DERIV: var fMRIPrep-utdata finns

Utdata
------
Skapas i: ~/MRI3/out/analysis/<SUB>/sm{FWHM}mm/
- power264_timeseries[_labels].csv/.tsv   (T × 264)
- power264_corr[_labels].csv               (264 × 264)
- run_info.json                             (metadata)
- power264_labels.nii.gz (om saknas; labels-pipeline)

Anmärkningar
------------
- TR hämtas från NIfTI-headern; fallback till BIDS-JSON.
- Volymer med FD > 0.5 mm “scrubbas” (sätts till 0 efter fillna).
- Bandpass: 0.009–0.08 Hz; standardisering och detrend sker i maskern.
- Mappenamnet (sm0/4/8mm) följer vald smoothing i aktuell pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_coords_power_2011
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
from scipy.spatial import cKDTree

# =========================
# STEG 0 — hitta fMRIPrep-filer
# =========================
BASE = Path("/Users/leahanna/MRI3/out") # <-- byt till din faktiska bas
SUB  = "sub-pa1372" # <-- byt till ditt subject, t.ex. "sub-01"
DERIV = BASE / "fmriprep-25.2.2" / SUB  #<-- byt "fmriprep-25.2.2" om din mapp heter annat
SMOOTH_FWHM = 8.0  # För sfärer

PIPELINE = "labels"  # "labels" eller "spheres"
LABELS_SMOOTH_FWHM = 4.0  # För labels

OUT_BASE = Path.home() /"MRI3" / "out" / "analysis" / SUB
OUT_BASE.mkdir(parents=True, exist_ok=True)
sm_tag = int(LABELS_SMOOTH_FWHM or 0) if PIPELINE == "labels" else int(SMOOTH_FWHM)
OUT = OUT_BASE / f"sm{sm_tag}mm"
OUT.mkdir(parents=True, exist_ok=True)


FUNC_CANDIDATES = (
    list((DERIV / "func").glob("*space-MNI*desc-preproc_bold.nii*"))
    or list(DERIV.glob("ses-*/func/*space-MNI*desc-preproc_bold.nii*"))
    or list(DERIV.glob("func/*space-MNI*desc-preproc_bold.nii*"))
)

CONF_CANDIDATES = (
    list((DERIV / "func").glob("*desc-confounds_timeseries.tsv"))
    or list(DERIV.glob("ses-*/func/*desc-confounds_timeseries.tsv"))
    or list(DERIV.glob("func/*desc-confounds_timeseries.tsv"))
)

MASK_CANDIDATES = (
    list((DERIV / "func").glob("*space-MNI*desc-brain_mask.nii*"))
    or list(DERIV.glob("ses-*/func/*space-MNI*desc-brain_mask.nii*"))
    or list(DERIV.glob("func/*space-MNI*desc-brain_mask.nii*"))
)

if not FUNC_CANDIDATES:
    raise FileNotFoundError(f"Ingen preproc_bold hittad under {DERIV}")
if not CONF_CANDIDATES:
    raise FileNotFoundError(f"Inga confounds hittade under {DERIV}")
if not MASK_CANDIDATES:
    raise FileNotFoundError(f"Ingen brain_mask hittad under {DERIV}")

FUNC = FUNC_CANDIDATES[0]
CONF = CONF_CANDIDATES[0]
MASK = MASK_CANDIDATES[0]

print("Hittat BOLD:      ", FUNC)
print("Hittat CONFOUNDS: ", CONF)
print("Hittat MASK:      ", MASK)

# =========================
# STEG 1 — läs BOLD + TR
# =========================
img = nib.load(str(FUNC))
zooms = img.header.get_zooms()
space_u, time_u = img.header.get_xyzt_units()  # t.ex. ('mm', 'sec') eller ('mm', 'msec')
TR = float(zooms[3]) if len(zooms) > 3 else None
if TR is not None and time_u == 'msec':
    TR /= 1000.0  # konvertera till sekunder om headern säger millisekunder

# Fallback: läs från BIDS-JSON om TR saknas/ser fel ut
if (TR is None or TR <= 0):
    import json, pathlib
    jpath = pathlib.Path(str(FUNC)).with_suffix('').with_suffix('.json')
    if jpath.exists():
        try:
            j = json.loads(jpath.read_text())
            TR = float(j.get("RepetitionTime", TR))
        except Exception:
            pass

print("\n[STEG 1]")
print("Shape (x,y,z,t):", img.shape)
print("Voxelstorlek (mm):", zooms[:3])
print("TR (s):", TR)


# =========================
# STEG 2 — confounds
# =========================
conf_df = pd.read_csv(CONF, sep="\t")

base_cands = [
    "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
    "csf","white_matter","global_signal","framewise_displacement"
]
cols = [c for c in base_cands if c in conf_df.columns]

# frivilligt: aCompCor (ta 5 första om de finns)
acc_cols = sorted([c for c in conf_df.columns if c.startswith("a_comp_cor_")])[:5]
use_cols = cols + acc_cols
X = conf_df[use_cols].copy()

n_bad = 0
if "framewise_displacement" in X.columns:
    bad = X["framewise_displacement"] > 0.5
    n_bad = int(bad.sum())
    X.loc[bad, :] = np.nan

X = X.fillna(0.0).values

print("\n[STEG 2]")
print("Confounds valda:", use_cols)
print("Volymer med hög FD (>0.5 mm):", n_bad)
print("Confounds-shape:", X.shape)

# --- sanity checks ---
T = img.shape[-1]
assert X.shape[0] == T, f"Confounds ({X.shape[0]}) matchar inte antal volymer ({T})"
import numpy as np
assert np.isfinite(X).all(), "NaN/Inf i confounds efter fillna"


# =========================
# STEG 3 — Power-264 seeds
# =========================

def make_power264_labels_nifti(reference_img, out_path: Path):
    # referens som 3D
    ref = reference_img
    if ref.ndim == 4:
        ref = ref.slicer[..., 0]
    aff = ref.affine
    shape = ref.shape
    data_ref = np.asarray(ref.get_fdata())
    brain_mask = (data_ref > 0) if (np.unique(data_ref).size <= 5 and data_ref.max() <= 1.0) else np.isfinite(data_ref)

    # Power-koordinater
    pow_df = fetch_coords_power_2011()["rois"]
    seeds_xyz = pow_df[["x","y","z"]].to_numpy(dtype=float)  # (264,3)
    ids = np.arange(1, seeds_xyz.shape[0] + 1, dtype=np.int16)  # 1..264

    # voxel->värld, KD-tree närmaste centrum
    ijk = np.array(np.nonzero(brain_mask)).T
    xyz = nib.affines.apply_affine(aff, ijk)
    tree = cKDTree(seeds_xyz)
    _, idx = tree.query(xyz, k=1)  # 0..263

    labels = np.zeros(shape, dtype=np.int16)
    labels[brain_mask] = ids[idx]
    out_img = nib.Nifti1Image(labels, ref.affine, ref.header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(out_path))
    return out_path

def save_timeseries_and_corr(ts: np.ndarray, labels: list[str], out_dir: Path, suffix: str):
    """
    Sparar tidsserier (CSV + TSV) och korrelationsmatris (CSV).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ts, columns=labels)

    # Tidsserier
    csv_path = out_dir / f"power264_timeseries{suffix}.csv"
    tsv_path = out_dir / f"power264_timeseries{suffix}.tsv"
    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, sep="\t", index=False)

    # Korrelationsmatris
    corr = df.corr(method="pearson")
    corr_path = out_dir / f"power264_corr{suffix}.csv"
    corr.to_csv(corr_path, index=True, float_format="%.6f")

    print("Sparat tidsserier (CSV):", csv_path)
    print("Sparat tidsserier (TSV):", tsv_path)
    print("Sparat korrelation (CSV):", corr_path)
    return csv_path, tsv_path, corr_path


# =========================
# STEG 4 — extrahera tidsserier för labels/sfärer
# =========================
mask_img = load_img(str(MASK))
bold_img = load_img(str(FUNC))

print("\n[STEG 4] Extraktion startar…")
if PIPELINE == "spheres":
    radius_mm = 5
    use_mask = True
    ts_list, kept_labels, skipped = [], [], 0
    for idx, (seed, lab) in enumerate(zip(seeds_all, labels_all)):
        try:
            m = NiftiSpheresMasker(
                seeds=[seed],
                radius=radius_mm,
                mask_img=(mask_img if use_mask else None),
                detrend=True, standardize=True,
                low_pass=0.08 if TR else None,
                high_pass=0.009 if TR else None,
                t_r=TR,
                smoothing_fwhm=SMOOTH_FWHM,
            )
            ts_i = m.fit_transform(bold_img, confounds=X)
            ts_list.append(ts_i[:, 0]); kept_labels.append(lab)
        except ValueError:
            skipped += 1; print(f"Skip ROI {idx:03d} ({lab}): empty sphere")
    if not ts_list:
        raise RuntimeError("Inga ROI-tidsserier (spheres). Testa mindre radie/utan mask.")
    ts = np.column_stack(ts_list)
    labels = kept_labels
    suffix = ""  # filnamn utan _labels
    print("Antal borttagna (empty) ROIs:", skipped)

elif PIPELINE == "labels":
    # 4a) Skapa eller ladda label-NIfTI
    OUT_BASE = Path.home() / "MRI3" / "out" / "analysis" / SUB
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_BASE / "power264_labels.nii.gz"
    if not labels_path.exists():
        print("Skapar Power264 label-NIfTI…")
        make_power264_labels_nifti(bold_img, labels_path)
    labels_img = load_img(str(labels_path))
    if bold_img.shape[:3] != labels_img.shape[:3]:
        labels_img = resample_to_img(labels_img, bold_img, interpolation="nearest")

    # 4b) Extrahera med labels
    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize=True,
        detrend=True,
        low_pass=0.08,
        high_pass=0.009,
        t_r=TR,
        smoothing_fwhm=LABELS_SMOOTH_FWHM, # None = ingen smoothing
    )
    ts = masker.fit_transform(bold_img, confounds=X)  # (T, 264)
    labels = [f"ROI_{i+1:03d}" for i in range(ts.shape[1])]
    suffix = "_labels"  # skilj filnamn
else:
    raise SystemExit("PIPELINE måste vara 'spheres' eller 'labels'.")

print("[STEG 4] KLART ✅  Timeserier shape (T, ROIs):", ts.shape)
_ = save_timeseries_and_corr(ts, labels, OUT, suffix)
import pandas as pd
pd.DataFrame(ts, columns=labels).to_csv(OUT / f"power264_timeseries{suffix}.tsv", index=False)
print("Sparat tidsserier:", OUT / f"power264_timeseries{suffix}.tsv")

import pandas as pd
out_file = OUT / f"power264_timeseries{suffix}.tsv"
pd.DataFrame(ts, columns=labels).to_csv(out_file, index=False)
print("Sparat tidsserier:", out_file)

# --- spara metadata ---
run_info = {
    "subject": SUB,
    "bold_path": str(FUNC),
    "confounds_path": str(CONF),
    "mask_path": str(MASK),
    "TR_sec": float(TR) if TR is not None else None,
    "voxel_size_mm": [float(zooms[0]), float(zooms[1]), float(zooms[2])],
    "pipeline": PIPELINE,
    "smoothing_fwhm_mm": float(SMOOTH_FWHM if PIPELINE=="spheres" else (LABELS_SMOOTH_FWHM or 0.0)),
    "n_volumes": int(ts.shape[0]),
    "n_rois_total": int(ts.shape[1]),
    "n_rois_kept": int(len(labels)),
    "confounds_used": list(map(str, use_cols)),
    "fd_over_0p5": int(n_bad),
}

if PIPELINE == "labels":
    run_info["labels_path"] = str(labels_path)
    run_info["labels_smoothing_fwhm_mm"] = float(LABELS_SMOOTH_FWHM or 0.0)


import json
(OUT / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False))