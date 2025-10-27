"""
Power-264: Läs nätverksindelning från Excel, beräkna segregation och plotta

Författare: Lea Roitman
Datum: 2025-10-27
Projekt: fMRIPrep 
kontakt: leah.roitman@gmail.com

Beskrivning
--------
1) Läser Power/Neuron-Excel (tvåradig header) och plockar ut ROI-ID (1..264)
   samt "Suggested System".
2) Bygger en mapping {ROI_ID -> System} och sparar till
   OUTDIR/power264_node_to_system.csv.
3) Läser 264×264 korrelationsmatrisen (CSV med index), säkerställer numerik,
   sätter rader/kolumner till 1..264 och diagonal = 1.0.
4) Grupperar ROI per System och beräknar segregation per system:
     S = (mean_within - mean_between) / mean_within
5) Sparar resultat till OUTDIR/segregation_by_system.csv och skriver ut toppresultat.
6) Plottar:
   - Stapeldiagram för segregation per system
   - Grupp×grupp-heatmap (medelkorrelation)
   - Reorderad hel korrelationsheatmap (blockad enligt systemordning)
   Figurer sparas i OUTDIR/segregation/.

Indata (vägar)
--------------
- EXCEL:  Neuron_consensus_264.xlsx (innehåller "Suggested System")
- CORR:   power264_corr_labels.csv (264×264, sparad MED indexkolumn)
- OUTDIR: mål-mapp för CSV och figurer (t.ex. .../analysis/<SUB>/sm8mm)

Utdata
------
- OUTDIR/power264_node_to_system.csv
- OUTDIR/segregation_by_system.csv
- OUTDIR/segregation/segregation_per_group_bar.png
- OUTDIR/segregation/heatmap_group_by_group.png
- OUTDIR/segregation/heatmap_reordered_full.png

Noter
-----
- ROI som saknar system i Excel får etiketten "MISSING" och exkluderas i analysen.
- Plot kräver matplotlib; annars hoppar skriptet över figur-delarna.
- Se till att CORR och OUTDIR pekar på samma smoothing-mapp (sm0/4/8mm).
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

EXCEL = "/Users/leahanna/2011Neurondata/Neuron_consensus_264.xlsx"
CORR  = "/Users/leahanna/MRI3/out/analysis/sub-pa1372/sm8mm/power264_corr_labels.csv"
OUTDIR= "/Users/leahanna/MRI3/out/analysis/sub-pa1372/sm8mm"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

xl = pd.ExcelFile(EXCEL)

# Läs med två-raders header (MultiIndex)
def read_sheet(sh):
    df = xl.parse(sh, header=[0,1])
    return df

# hitta blad som har "Suggested System"
sheet = None
for sh in xl.sheet_names:
    df_try = read_sheet(sh)
    if any(str(a).strip().lower()=="suggested system" for a,b in df_try.columns) and len(df_try)>=200:
        sheet = sh; break
df = read_sheet(sheet or xl.sheet_names[0])

# plocka kolumner via headernivåer
id_col  = next(col for col in df.columns if str(col[1]).strip().lower()=="roi")
sys_col = next(col for col in df.columns if str(col[0]).strip().lower()=="suggested system")

# bygg mapping ROI -> system
df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
node2sys = {int(rid): sys for rid, sys in zip(df[id_col].dropna(), df[sys_col].dropna())
            if 1 <= int(rid) <= 264}

# spara mapping
map_df = pd.DataFrame({"ROI_ID": range(1,265),
                       "System": [node2sys.get(i,"MISSING") for i in range(1,265)]})
map_df.to_csv(f"{OUTDIR}/power264_node_to_system.csv", index=False)
print("Sparat mapping →", f"{OUTDIR}/power264_node_to_system.csv")

# --- (valfritt) räkna segregation från din 264x264-korrelationsmatris ---
# Din korrelations-CSV är sparad MED index → läs med index_col=0
df_corr = pd.read_csv(CORR, index_col=0)

# Om någon "Unnamed: 0"-kolumn råkar följa med, rensa bort den
df_corr = df_corr.loc[:, ~df_corr.columns.astype(str).str.startswith("Unnamed")]

# Tvinga numeriskt
df_corr = df_corr.apply(pd.to_numeric, errors="coerce")

# Sätt rader/kolumner till ROI-ID 1..264 (nu ska det vara 264×264)
df_corr.columns = range(1, 265)
df_corr.index   = range(1, 265)

# Sätt diagonal till 1.0 (säkerhetsbälte)
vals = df_corr.values
np.fill_diagonal(vals, 1.0)
df_corr.iloc[:, :] = vals


R = df_corr.to_numpy(dtype=float)
assert R.shape == (264, 264), f"Korrelationsmatrisen måste vara 264x264 (fick {R.shape})"


# Bygg grupper: system -> lista av ROI-id
GROUPS = defaultdict(list)
for rid in range(1, 265):
    GROUPS[node2sys.get(rid, "MISSING")].append(rid)
# Ta bort ev. MISSING
if "MISSING" in GROUPS:
    del GROUPS["MISSING"]

def mean_without_diag(M: np.ndarray):
    triu = M[np.triu_indices_from(M, k=1)]
    return np.nanmean(triu) if triu.size else np.nan

rows = []
for sys, roi_ids in GROUPS.items():
    idx = np.array([i-1 for i in roi_ids])
    if idx.size < 2:
        rows.append((sys, np.nan, np.nan, np.nan, idx.size))
        continue
    within  = mean_without_diag(R[np.ix_(idx, idx)])
    other   = np.setdiff1d(np.arange(264), idx)
    between = np.nanmean(R[np.ix_(idx, other)]) if other.size else np.nan
    S = (within - between) / within if (np.isfinite(within) and within != 0) else np.nan
    rows.append((sys, within, between, S, idx.size))

seg_df = (
    pd.DataFrame(rows, columns=["System","mean_within","mean_between","Segregation_S","n_ROI"])
      .sort_values("Segregation_S", ascending=False)
)

seg_path = f"{OUTDIR}/segregation_by_system.csv"
seg_df.to_csv(seg_path, index=False)
print("Sparat segregation →", seg_path)

# Visa lite resultat i terminalen
print("\nToppsnitt:")
print(seg_df.head(10).to_string(index=False))


# --- PLOTTAR från redan beräknade data ---
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("[INFO] matplotlib saknas – inga figurer ritas.")
    raise SystemExit

PLOT_DIR = Path(OUTDIR) / "segregation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Stapeldiagram: segregation per system
plt.figure()
plt.bar(seg_df["System"], seg_df["Segregation_S"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Segregation S = (W - B) / W")
plt.title("Segregation per system")
plt.tight_layout()
plt.savefig(PLOT_DIR / "segregation_per_group_bar.png", dpi=150)

# 2) Heatmap: grupp × grupp (medelkorrelation)
#    Bygg ordning efter seg_df (mest segregerade först, snyggare figur)
group_order = seg_df["System"].tolist()
G = len(group_order)

def _mean_within(M, idx0):
    if len(idx0) < 2:
        return np.nan
    sub = M[np.ix_(idx0, idx0)]
    iu = np.triu_indices(len(idx0), k=1)
    return np.nanmean(sub[iu]) if sub.size else np.nan

def _mean_between(M, a0, b0):
    if not a0 or not b0:
        return np.nan
    return np.nanmean(M[np.ix_(a0, b0)])

# GROUPS innehåller ROI som 1..264 -> gör 0-baserat för indexering i R
GROUPS_0 = {k: [i-1 for i in v] for k, v in GROUPS.items()}
GB = np.full((G, G), np.nan)
for i, gi in enumerate(group_order):
    for j, gj in enumerate(group_order):
        ai = GROUPS_0.get(gi, [])
        bj = GROUPS_0.get(gj, [])
        GB[i, j] = _mean_within(R, ai) if i == j else _mean_between(R, ai, bj)

plt.figure()
plt.imshow(GB, interpolation="nearest")
plt.title("Grupp × Grupp – medelkorrelation")
plt.xticks(range(G), group_order, rotation=45, ha="right")
plt.yticks(range(G), group_order)
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(PLOT_DIR / "heatmap_group_by_group.png", dpi=150)

# 3) Reorderad full korrelationsheatmap (blockordning efter group_order)
order = []
for g in group_order:
    order.extend(GROUPS_0.get(g, []))
if order and len(order) == R.shape[0]:
    R_ord = R[np.ix_(order, order)]
    plt.figure()
    plt.imshow(R_ord, interpolation="nearest")
    plt.title("Reorderad korrelationsmatris")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "heatmap_reordered_full.png", dpi=150)
else:
    print("[INFO] Full heatmap hoppad – ordningen täcker inte alla ROI.")

print(f"\nFigurer sparade i: {PLOT_DIR}")
