import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import NetworkAnalyzer
from laticceTopology import LatticeTopologyNetwork
from linearTopology import LinearTopologyNetwork
from treeTopology import TreeTopologyNetwork

OUT_DIR = "lab4_results"
N_COL = "N (кількість процесорів)"
METRICS = ["D", "D̄", "S", "C", "T"]
METRIC_NAMES = {
    "D": "Діаметр D",
    "D̄": "Середній діаметр D̄",
    "S": "Ступінь S",
    "C": "Вартість C",
    "T": "Трафік T",
}

TOPOLOGIES = {
    "Лінійна (ЛР1)": (LinearTopologyNetwork,
                      sorted(set(range(1, 18)) | set(range(20, 161, 10)))),
    "Бінарне дерево (ЛР2)": (TreeTopologyNetwork,
                             sorted(set([1] + list(range(6, 126, 6))))),
    "Решітка (ЛР3)": (LatticeTopologyNetwork, [k * k for k in range(1, 12)]),
}

STYLES = {
    "Лінійна (ЛР1)": dict(color="#2a78d6", marker="o", linestyle="-"),
    "Бінарне дерево (ЛР2)": dict(color="#eb6834", marker="s", linestyle="--"),
    "Решітка (ЛР3)": dict(color="#1baf7a", marker="^", linestyle="-."),
}

ZONE_BOUNDS = [100, 450]
ZONES = [
    ("Зона I", 0, ZONE_BOUNDS[0]),
    ("Зона II", ZONE_BOUNDS[0], ZONE_BOUNDS[1]),
    ("Зона III", ZONE_BOUNDS[1], np.inf),
]


def load_tables():
    os.makedirs(OUT_DIR, exist_ok=True)
    tables = {}
    for i, (name, (cls, counts)) in enumerate(TOPOLOGIES.items(), start=1):
        path = os.path.join(OUT_DIR, f"lab{i}_table.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig")
        else:
            print(f"Обчислення: {name} ...")
            df = NetworkAnalyzer.generate_scaling_table(None, cls, cluster_counts=counts)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        tables[name] = df
    return tables


def find_crossings(tables):
    names = list(tables)
    rows = []
    for m in METRICS:
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ta, tb = tables[names[a]], tables[names[b]]
                lo = max(ta[N_COL].min(), tb[N_COL].min())
                hi = min(ta[N_COL].max(), tb[N_COL].max())
                grid = np.arange(lo, hi + 1)
                diff = (np.interp(grid, ta[N_COL], ta[m]) - np.interp(grid, tb[N_COL], tb[m]))
                sign = np.sign(diff)
                for k in np.where(sign[:-1] * sign[1:] < 0)[0]:
                    rows.append({"Характеристика": m, "Пара": f"{names[a]} / {names[b]}",
                                 "N перетину": int(grid[k + 1])})
    return pd.DataFrame(rows)


def plot_metrics(tables):
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, df in tables.items():
            ax.plot(df[N_COL], df[m], label=name, linewidth=2, markersize=6, **STYLES[name])
        for x in ZONE_BOUNDS:
            ax.axvline(x, color="#888888", linewidth=1, linestyle=":")
        for zname, lo, hi in ZONES:
            hi = min(hi, 1000)
            ax.text((lo + hi) / 2, 1.01, zname, ha="center", va="bottom", color="#52514e",
                    transform=ax.get_xaxis_transform())
        ax.set_title(f"{METRIC_NAMES[m]} залежно від кількості вузлів", pad=22)
        ax.set_xlabel("Кількість вузлів N")
        ax.set_ylabel(METRIC_NAMES[m])
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        fig.tight_layout()
        safe = {"D̄": "aD"}.get(m, m)
        fig.savefig(os.path.join(OUT_DIR, f"plot_{safe}.png"), dpi=150)
        plt.close(fig)


def zone_means(tables):
    result = {}
    for zname, lo, hi in ZONES:
        rows = {}
        for name, df in tables.items():
            part = df[(df[N_COL] > lo) & (df[N_COL] <= hi)]
            if part.empty:
                raise ValueError(f"{zname}: немає експериментів для '{name}'")
            row = part[METRICS].mean().round(3).astype(object)
            row["К-сть експериментів"] = len(part)
            row["N (діапазон)"] = f"{part[N_COL].min()}–{part[N_COL].max()}"
            rows[name] = row
        result[zname] = pd.DataFrame(rows).T
    return result


def normalize(means):
    vals = means[METRICS].astype(float).copy()
    vals["T"] = vals["T"].where(vals["T"] >= 1, 2 - vals["T"])  # перерахунок трафіку < 1
    norm = (vals / vals.min()).round(3)
    norm["Інтегральна оцінка"] = norm[METRICS].sum(axis=1).round(3)
    return norm


def plot_histogram(integral):
    zones = list(integral.index)
    names = list(integral.columns)
    x = np.arange(len(zones))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    hatches = ["", "//", "\\\\"]
    for i, name in enumerate(names):
        bars = ax.bar(x + (i - 1) * width, integral[name], width * 0.92, label=name,
                      color=STYLES[name]["color"], hatch=hatches[i], edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.axhline(5, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xlim(-0.5, x[-1] + 0.85)
    ax.text(x[-1] + 0.45, 5.05, "оптимум = 5", va="bottom", ha="left", color="#52514e", fontsize=9)
    ax.set_xticks(x, zones)
    ax.set_ylabel("Інтегральна оцінка")
    ax.set_title("Інтегральні оцінки топологій по зонах")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, integral.values.max() * 1.25)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "histogram_integral.png"), dpi=150)
    plt.close(fig)


def main():
    pd.set_option("display.width", 200)
    tables = load_tables()
    for name, df in tables.items():
        print(f"\n=== {name} ===\n{df.to_string(index=False)}")

    crossings = find_crossings(tables)
    print("\n=== Точки перетину графіків ===\n" + crossings.to_string(index=False))

    plot_metrics(tables)

    means = zone_means(tables)
    normalized = {z: normalize(df) for z, df in means.items()}
    integral = pd.DataFrame({z: n["Інтегральна оцінка"] for z, n in normalized.items()}).T

    for z in means:
        print(f"\n=== {z}: середні значення ===\n{means[z].to_string()}")
        print(f"\n=== {z}: нормалізовані значення ===\n{normalized[z].to_string()}")
    print("\n=== Інтегральні оцінки ===\n" + integral.to_string())
    for z, row in integral.iterrows():
        print(f"{z}: найкраща — {row.idxmin()} ({row.min():.3f}), гірша — {row.idxmax()} ({row.max():.3f})")

    plot_histogram(integral)

    xlsx = os.path.join(OUT_DIR, "lab4_results.xlsx")
    with pd.ExcelWriter(xlsx) as w:
        for i, (name, df) in enumerate(tables.items(), start=1):
            df.to_excel(w, sheet_name=f"ЛР{i}", index=False)
        crossings.to_excel(w, sheet_name="Перетини", index=False)
        for z in means:
            means[z].to_excel(w, sheet_name=f"{z} середні")
            normalized[z].to_excel(w, sheet_name=f"{z} нормалізовані")
        integral.to_excel(w, sheet_name="Інтегральні оцінки")
    print(f"\nРезультати збережено у теці {OUT_DIR}/")


if __name__ == "__main__":
    main()
