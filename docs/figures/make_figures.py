"""Regenerates the figures and the numbers shown in the README.

    pip install numpy pandas scikit-learn matplotlib pillow
    python docs/figures/make_figures.py

Part 1 re-executes the value-iteration cell of entregas/Groba/TP2-RL.ipynb; Part 2 re-fits the
final model of TP2-ML.ipynb; Parts 3 and 4 re-plot outputs saved in TP2-CV.ipynb and
TP2-NLP.ipynb (training a CNN or downloading BERT/Word2Vec is not needed).
"""
import contextlib
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUB = ROOT / "notebooks"
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name, **kw):
    fig.savefig(HERE / name, metadata={"Date": None} if name.endswith(".svg") else {}, **kw)
    plt.close(fig)


def cell(nb, i):
    return "".join(json.loads((SUB / nb).read_text())["cells"][i]["source"])


# ---- Figure 1: the nursery MDP and its optimal value function / policy
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(cell("TP2-RL.ipynb", 0), ns)  # defines value_iteration(gamma, tol=0.1)
GAMMAS = (0.1, 0.99, 1.0)
sol = {g: ns["value_iteration"](g) for g in GAMMAS}

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 3], wspace=0.1, hspace=0.55)
ax = fig.add_subplot(gs[:, 0])
pos = {0: (0, 0), 1: (0.75, 1.0), 2: (0.75, -1.0), 3: (1.5, 0)}
desc = {0: "start", 1: "market", 2: "fair", 3: "month\nend"}
for a, b in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (2, 3)]:
    p0, p1 = np.array(pos[a], float), np.array(pos[b], float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    off = 0.06 * np.array([d[1], -d[0]]) if b != 3 else 0  # two-way edges drawn side by side
    ax.add_patch(FancyArrowPatch(p0 + off, p1 + off, arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                 color="#1a1a1a", shrinkA=11, shrinkB=11))
for k, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), 0.2, fc="white", ec="#1a1a1a", lw=0.9, zorder=3))
    ax.text(x, y, str(k), ha="center", va="center", zorder=4)
    lx, ly, ha, va = {0: (0, -0.27, "center", "top"), 1: (0, 0.27, "center", "bottom"),
                      2: (0, -0.27, "center", "top"), 3: (0, -0.27, "center", "top")}[k]
    ax.text(x + lx, y + ly, desc[k], ha=ha, va=va, fontsize=8)
ax.set_xlim(-0.4, 1.9)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Graph", y=0.97)

cmap = plt.get_cmap("Blues")
for r, (g, letter) in enumerate(zip(GAMMAS, "bcd")):
    V, pi = sol[g]
    ax = fig.add_subplot(gs[r, 1])
    M = np.array([[V[(n, d)] for d in range(11)] for n in range(3)])
    ax.imshow(np.where(M < -100, np.nan, M), cmap=cmap, vmin=-2, vmax=6, aspect="auto")
    for n in range(3):
        for d in range(11):
            v = M[n, d]
            if v < -100:
                ax.add_patch(plt.Rectangle((d - 0.5, n - 0.5), 1, 1, fc="#f2d4cf", ec="none"))
            acts = ",".join(map(str, pi[(n, d)]))
            txt = ("$-2000$" if v < -100 else f"{v:.2f}") + f"\n→{acts}"
            ax.text(d, n, txt, ha="center", va="center", fontsize=6.3, linespacing=0.95,
                    color="white" if v > 3.5 else "#1a1a1a")
    ax.set_xticks(range(11))
    ax.set_yticks(range(3), ["node 0", "node 1", "node 2"])
    ax.tick_params(length=0, top=False, right=False)
    ax.set_title(f"({letter}) $V^*$ and $\\pi^*$, $\\gamma = {g:g}$", fontsize=9, pad=3)
    if r == 2:
        ax.set_xlabel("money [\\$]")
save(fig, "fig1-mdp.svg")

# ---- Figure 2: iris classifier and the "secret flower"
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

iris = pd.read_csv(SUB / "IRIS.csv")
secret = pd.read_csv(SUB / "secret_flower.csv")
FEATS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
knn = KNeighborsClassifier(n_neighbors=5, weights="uniform")  # final model of TP2-ML.ipynb
cv_acc = cross_val_score(knn, iris[FEATS], iris["species"], cv=3).mean()
knn.fit(iris[FEATS], iris["species"])
pred = knn.predict(secret[FEATS])
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
for s, c in zip(SPECIES, C):
    g = iris[iris.species == s]
    ax1.plot(g.petal_length, g.petal_width, "o", ms=3.2, color=c, alpha=0.55, mec="none",
             label=s.replace("Iris-", "I. "))
    h = secret[pred == s]
    if len(h):
        ax1.plot(h.petal_length, h.petal_width, "*", ms=8, color=c, mec="#1a1a1a", mew=0.5)
ax1.plot([], [], "*", ms=8, color="white", mec="#1a1a1a", mew=0.5, label="hybrid flower")
ax1.set_xlabel("petal length [cm]")
ax1.set_ylabel("petal width [cm]")
ax1.set_title("(a) Development set and hybrid flowers")
ax1.legend(loc="upper left", handletextpad=0.2)
cent = iris.groupby("species")[FEATS].mean()
x = np.arange(4)
for s, c in zip(SPECIES, C):
    ax2.plot(x, cent.loc[s], "o-", color=c, lw=1, ms=3.5, label=s.replace("Iris-", "I. "))
ax2.plot(x, secret[FEATS].mean(), "s--", color="#1a1a1a", lw=1, ms=3.5, label="hybrid")
ax2.set_xticks(x, ["sepal\nlength", "sepal\nwidth", "petal\nlength", "petal\nwidth"])
ax2.set_ylabel("mean [cm]")
ax2.set_title("(b) Class centroids")
ax2.legend(loc="lower left")
ax2.tick_params(top=False)
fig.tight_layout()
save(fig, "fig2-iris.svg")

# ---- Figure 3: CNN learning curves, parsed from the training log saved in TP2-CV.ipynb
nb = json.loads((SUB / "TP2-CV.ipynb").read_text())
log = "".join("".join(o.get("text", "")) for o in nb["cells"][5]["outputs"])
ep = np.array(re.findall(r"Ep (\d+)/\d+ \| [\d.]+s train_loss=([\d.]+) val_loss=([\d.]+) val_acc=([\d.]+)",
                         log), float)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax1.plot(ep[:, 0], ep[:, 1], color=C[0], lw=1, label="train (dropout on)")
ax1.plot(ep[:, 0], ep[:, 2], color=C[1], lw=1, label="validation")
best = ep[ep[:, 2].argmin()]
ax1.plot(best[0], best[2], "o", color=C[1], ms=4)
ax1.annotate(f"min. {best[2]:.3f} (epoch {best[0]:.0f})", (best[0], best[2]), xytext=(95, 0.75),
             fontsize=8, color=C[1])
ax1.set_xlabel("epoch")
ax1.set_ylabel("cross-entropy")
ax1.set_title("(a) Loss")
ax1.legend(loc="upper right")
ax2.plot(ep[:, 0], ep[:, 3] * 100, color=C[1], lw=1)
ax2.axhline(80, color="#8c8c8c", ls=":", lw=0.8)
ax2.text(228, 79, "80 % target", ha="right", va="top", fontsize=8, color="#4d4d4d")
ax2.set_xlabel("epoch")
ax2.set_ylabel("validation accuracy [%]")
ax2.set_title("(b) Validation accuracy")
fig.tight_layout()
save(fig, "fig3-cnn-training.svg")

# ---- Figure 4: what the CNN sees (dataset patches) and the three photos (PNG, raster content)
CLASSES = ["Coleus neochilus", "Salvia microphylla", "Salvia uliginosa"]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.2)
top = gs[0].subgridspec(3, 8, wspace=0.04, hspace=0.06)
rng = np.random.default_rng(0)
for r, cls in enumerate(CLASSES):
    files = sorted((ROOT / "data" / "patches" / cls).glob("*.jpg"))
    for j, fpath in enumerate(rng.choice(files, 8, replace=False)):
        ax = fig.add_subplot(top[r, j])
        ax.imshow(Image.open(fpath))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if j == 0:
            ax.set_ylabel(cls.replace(" ", "\n"), fontsize=8, fontstyle="italic", rotation=0,
                          ha="right", va="center", labelpad=4)
fig.text(0.5, 0.905, "(a) Random 96 × 96 training patches", ha="center")
bottom = gs[1].subgridspec(1, 3, wspace=0.04)
VOTES = [("Coleus neochilus", 125, 140), ("Coleus neochilus", 105, 165), ("Salvia uliginosa", 104, 165)]
for k, (cls, v, tot) in enumerate(VOTES):
    ax = fig.add_subplot(bottom[k])
    img = ImageOps.exif_transpose(Image.open(SUB / "mis_fotos" / f"foto{k + 1}" / f"foto{k + 1}.jpg"))
    img.thumbnail((700, 700))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"photo {k + 1}: {cls}\n{v}/{tot} patches ({100 * v / tot:.1f} %)", fontsize=8)
fig.text(0.5, 0.452, "(b) Own photographs, majority vote over patches", ha="center")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=170)
plt.close(fig)
Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE).save(
    HERE / "fig4-patches.png", optimize=True)  # palette PNG keeps the file small

# ---- Figure 5: sentiment classifiers, confusion matrices as saved in TP2-NLP.ipynb
CM = {"(a) BERT (nlptown, 1–5 stars)": np.array([[19, 1], [3, 18]]),
      "(b) Word2Vec centroids": np.array([[14, 6], [9, 12]])}
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
for ax, (title, cm) in zip(axes, CM.items()):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=21)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 12 else "#1a1a1a")
    ax.set_xticks([0, 1], ["pos.", "neg."])
    ax.set_yticks([0, 1], ["pos.", "neg."])
    ax.tick_params(length=0, top=False, right=False)
    ax.set_xlabel("predicted")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\naccuracy {acc:.1%}", fontsize=9)
axes[0].set_ylabel("true")
fig.tight_layout()
save(fig, "fig5-nlp.svg")

# ---- numbers quoted in the README
print(f"KNN(k=5) 3-fold CV accuracy: {cv_acc:.4f}")
print("hybrid predictions:", pd.Series(pred).value_counts().to_dict())
print("hybrid centroid:", secret[FEATS].mean().round(2).to_dict())
print("CNN: min val loss", best, "| last epoch", ep[-1], "| max val acc", ep[ep[:, 3].argmax()])
