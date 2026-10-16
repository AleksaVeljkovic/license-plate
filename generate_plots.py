"""
Cilj: Iz evaluation_results.csv (rezultat evaluate_models.py) napraviti
preglednu grafiku.

Generise tri vrste izlaza:

1) plots/po_modelu/<model>_<metrika>.png
   Za jedan model: sve degradacije na istom grafikonu (koja degradacija je
   najgora). Y-osa je FIKSNA (0-1), pa se grafikoni razlicitih modela mogu
   direktno porediti.

2) plots/poredjenje/<degradacija>_<metrika>.png
   Za jednu degradaciju: svih 5 modela na istom grafikonu (koji model je
   otporniji). Ovo je glavni grafikon za poglavlje o rezultatima.

3) plots/pregled_<metrika>.png
   Jedna slika sa 5 podgrafikona (po jedan za svaku degradaciju), svi modeli
   u svakom - sazetak za rad.

Dodatno: plots/prag_pada.csv - SSIM nivo na kom svaki model prvi put padne
ispod zadatog praga mAP50 (PRAG_MAP50), po tipu degradacije.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "evaluation_results_clean.csv"
OUTPUT_DIR = Path("plots_clean")

METRICS = ["mAP50", "mAP50-95", "precision", "recall"]

# Metrike za koje se prave pregledne slike (sa 5 podgrafikona)
OVERVIEW_METRICS = ["mAP50", "mAP50-95"]

# Prag za tabelu "prag pada" (mAP50 ispod ove vrednosti = detekcija se raspala)
PRAG_MAP50 = 0.5

DEGRADATION_COLORS = {
    "mrak": "#2c3e50",
    "magla": "#95a5a6",
    "sneg": "#3498db",
    "blur": "#e67e22",
    "kompresija": "#9b59b6",
}

DEGRADATION_LABELS = {
    "mrak": "Mrak",
    "magla": "Magla",
    "sneg": "Sneg",
    "blur": "Zamućenje pokreta",
    "kompresija": "Kompresija/rezolucija",
}

# Boje i stilovi za modele (razlicite boje + razlicite oznake, da se razlikuju
# i u crno-beloj stampi)
MODEL_ORDER = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
MODEL_STYLE = {
    "yolo11n": {"color": "#1b9e77", "marker": "o", "linestyle": "-"},
    "yolo11s": {"color": "#d95f02", "marker": "s", "linestyle": "-"},
    "yolo11m": {"color": "#7570b3", "marker": "^", "linestyle": "-"},
    "yolo11l": {"color": "#e7298a", "marker": "D", "linestyle": "-"},
    "yolo11x": {"color": "#555555", "marker": "v", "linestyle": "--"},
}

# -----------------------------------------------------------


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")

    # ISPRAVKA: SSIM ponovo racunamo iz kolone 'nivo' (broj iz imena / 100),
    # jer je stariji evaluate_models.py jednocifrene nivoe pogresno citao
    # (ssim_2 -> 0.2 umesto 0.02). Baseline ostaje 1.0.
    from_name = pd.to_numeric(df["nivo"].astype(str).str.extract(r"ssim_(\d+)")[0], errors="coerce") / 100.0
    df["ssim"] = from_name.where(from_name.notna(), df["ssim"])
    return df


def ordered_models(df: pd.DataFrame):
    present = list(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present] + [m for m in present if m not in MODEL_ORDER]


def ordered_degradations(df: pd.DataFrame):
    present = [t for t in df["tip_degradacije"].unique() if t != "cisto"]
    order = list(DEGRADATION_COLORS.keys())
    return sorted(present, key=lambda t: order.index(t) if t in order else 999)


def curve(df: pd.DataFrame, model: str, tip: str, metric: str):
    """Vraca (x, y) listu za jedan model i jednu degradaciju, sa baseline
    tackom na pocetku (SSIM=1.0), sortirano od cistog ka degradiranom."""
    model_df = df[df["model"] == model]
    base = model_df[model_df["tip_degradacije"] == "cisto"]
    if base.empty:
        return None
    tip_df = model_df[model_df["tip_degradacije"] == tip].sort_values("ssim", ascending=False)
    x = [float(base["ssim"].iloc[0])] + tip_df["ssim"].tolist()
    y = [float(base[metric].iloc[0])] + tip_df[metric].tolist()
    return x, y


def style_axes(ax, metric: str):
    ax.set_ylim(-0.02, 1.02)  # fiksna Y-osa - grafikoni su medjusobno uporedivi
    ax.invert_xaxis()  # 1.0 levo, opada udesno
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("SSIM (1.0 = čista slika, niže = jača degradacija)")
    ax.set_ylabel(metric)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sačuvano: {path}")


# ---------- 1) po modelu: sve degradacije ----------

def plot_per_model(df, model, metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    for tip in ordered_degradations(df):
        c = curve(df, model, tip, metric)
        if c is None:
            plt.close(fig)
            return
        ax.plot(*c, marker="o", markersize=3, label=DEGRADATION_LABELS.get(tip, tip),
                color=DEGRADATION_COLORS.get(tip))
    style_axes(ax, metric)
    ax.set_title(f"{model} — {metric} u zavisnosti od SSIM-a")
    ax.legend(loc="lower left")
    save(fig, OUTPUT_DIR / "po_modelu" / f"{model}_{metric}.png")


# ---------- 2) po degradaciji: svi modeli ----------

def draw_models(ax, df, tip, metric, models, markersize=3.5):
    for model in models:
        c = curve(df, model, tip, metric)
        if c is None:
            continue
        ax.plot(*c, markersize=markersize, label=model.replace("yolo11", "YOLO11"),
                linewidth=1.6, **MODEL_STYLE.get(model, {}))


def plot_per_degradation(df, tip, metric, models):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_models(ax, df, tip, metric, models)
    style_axes(ax, metric)
    ax.set_title(f"{DEGRADATION_LABELS.get(tip, tip)} — poređenje modela ({metric})")
    ax.legend(loc="lower left")
    save(fig, OUTPUT_DIR / "poredjenje" / f"{tip}_{metric}.png")


# ---------- 3) pregled: 5 podgrafikona na jednoj slici ----------

def plot_overview(df, metric, models):
    types = ordered_degradations(df)
    ncols = 3
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, tip in zip(axes, types):
        draw_models(ax, df, tip, metric, models, markersize=3)
        ax.set_ylim(-0.02, 1.02)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(DEGRADATION_LABELS.get(tip, tip))
        ax.set_xlabel("SSIM")
        ax.set_ylabel(metric)

    # Prazni podgrafikon (6. polje) koristimo za zajednicku legendu
    for ax in axes[len(types):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if len(types) < len(axes):
        axes[len(types)].legend(handles, labels, loc="center", fontsize=12, title="Model")
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(models))

    fig.suptitle(f"{metric} u zavisnosti od SSIM-a — svi modeli, sve degradacije", fontsize=14)
    fig.tight_layout()
    save(fig, OUTPUT_DIR / f"pregled_{metric}.png")


# ---------- tabela praga pada ----------

def breakpoint_table(df, models, types):
    rows = []
    for tip in types:
        row = {"degradacija": DEGRADATION_LABELS.get(tip, tip)}
        for model in models:
            sub = df[(df["model"] == model) & (df["tip_degradacije"] == tip)].sort_values("ssim", ascending=False)
            below = sub[sub["mAP50"] < PRAG_MAP50]
            row[model] = f"{below['ssim'].iloc[0]:.2f}" if not below.empty else f"nije pao (min SSIM {sub['ssim'].min():.2f})"
        rows.append(row)
    out = pd.DataFrame(rows)
    path = OUTPUT_DIR / "prag_pada.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n  Prag pada (prvi SSIM sa mAP50 < {PRAG_MAP50}):")
    print(out.to_string(index=False))
    print(f"  Sačuvano: {path}")


def main():
    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        print(f"GRESKA: {csv_path.resolve()} ne postoji. Pokreni prvo evaluate_models.py.")
        return

    df = load_results(str(csv_path))
    models = ordered_models(df)
    types = ordered_degradations(df)

    print(f"Pronadjeno modela: {models}")
    print(f"Tipovi degradacije: {types}")
    print(f"Ukupno redova u CSV-u: {len(df)}\n")

    print("1) Grafikoni po modelu (sve degradacije)")
    for model in models:
        for metric in METRICS:
            plot_per_model(df, model, metric)

    print("\n2) Poredjenje modela po degradaciji")
    for tip in types:
        for metric in METRICS:
            plot_per_degradation(df, tip, metric, models)

    print("\n3) Pregledne slike")
    for metric in OVERVIEW_METRICS:
        plot_overview(df, metric, models)

    breakpoint_table(df, models, types)

    print(f"\nGotovo. Grafikoni sačuvani u: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()