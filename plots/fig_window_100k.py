#!/usr/bin/env python3
"""100k full-window-ladder figure: GloVe / SG / CBOW across w5-w70, d128.
All cold/plain (architecture comparison). CBOW = NS, ep05. GloVe N=1 (seed-
stable); SG, CBOW N=3. Okabe-Ito: GloVe #0072B2, CBOW #D55E00, SG #009E73."""
import os
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

M = ["STS-A", "SICK-A", "SL-999", "WS-353", "Analogy", "Outlier", "OPP"]
ML = {"STS-A":"STS","SICK-A":"SICK","SL-999":"SimLex-999","WS-353":"WS-353",
      "Analogy":"Analogy","Outlier":"Outlier","OPP":"OPP"}
COL = {"GloVe":"#0072B2","CBOW":"#D55E00","SG":"#009E73","PPMI-SVD":"#CC79A7"}

def arch(n):
    if "glove" in n: return "GloVe"
    if "ppmisvd" in n: return "PPMI-SVD"
    if "skgr" in n: return "SG"
    return "CBOW"
rows=[]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "results_100k.txt")) as f:
    r=csv.reader(f,delimiter="\t"); h=next(r)
    for a in r: rows.append(dict(zip(h,a)))

def curve(af,m):
    d={int(x["w"]):float(x[m]) for x in rows if arch(x["Model"])==af}
    xs=sorted(d); return xs,[d[w] for w in xs]

fig,axes=plt.subplots(2,4,figsize=(780/72.,380/72.))
fig.suptitle("Window-size acquisition at 100k articles (d=128, cold start)",
             fontsize=12,y=0.985)
axes=axes.ravel()
for k,m in enumerate(M):
    ax=axes[k]
    ax.grid(True,color="#E6E6E6",lw=0.6); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#B0B0B0")
    ax.tick_params(colors="#606060",labelsize=8)
    for af in ("GloVe","SG","CBOW","PPMI-SVD"):
        xs,ys=curve(af,m)
        if xs: ax.plot(xs,ys,color=COL[af],lw=2,marker="o",ms=4,mew=0)
    ax.set_title(ML[m],fontsize=10,color="#303030")
    ax.set_xlabel("window",fontsize=8)
axes[-1].axis("off")
axes[-1].legend(handles=[
    Line2D([],[],color=COL["GloVe"],lw=2,marker="o",ms=4,label="GloVe (N=1)"),
    Line2D([],[],color=COL["PPMI-SVD"],lw=2,marker="o",ms=4,label="PPMI-SVD (N=1)"),
    Line2D([],[],color=COL["CBOW"],lw=2,marker="o",ms=4,label="CBOW-NS (ep05)"),
    Line2D([],[],color=COL["SG"],lw=2,marker="o",ms=4,label="SG")],
    loc="center",frameon=False,fontsize=9,handlelength=2.4)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig("fig_window_100k.pdf")
print("wrote fig_window_100k.pdf")
