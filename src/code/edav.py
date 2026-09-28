import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# INIT:
## constants
out = "./outputs/"
BLUE = "#0072B2"
SOURCE = ("")
styles = {
    "Denmark": ("#666666", "-."),
    "Finland": ("#555555", "-"),
    "Iceland": ("#222222", "-"),
    "Norway": ("#666666", ":"),
    "Sweden": ("#444444", "--"),
}

pd_nordics = pd.read_csv("./data/nordics-lex-1950-2005.csv")

wide = pd_nordics.set_index("name").astype(float).sort_index()
wide.index.name = "country"
wide.columns = wide.columns.astype(int)
wide.columns.name = "year"

years = wide.columns.tolist()
start, end = years[0], years[-1]

endpoints = wide[[start, end]].copy()

summary = endpoints.copy()
summary["gain_years"] = summary[end] - summary[start]

for year in (start, end):
    summary[f"rank_{year}"] = (
        summary[year].rank(ascending=False, method="min").astype(int)
    )

ordered = summary.sort_values("gain_years", ascending=False)

spread = (wide.max() - wide.min()).rename("range_years")

# Find all minimum years.
minimum_years = spread.index[
    np.isclose(spread, spread.min(), rtol=0, atol=1e-10)
]
minimum_text = " and ".join(str(year) for year in minimum_years)

focus = ordered.index[0]
leader = summary[end].idxmax()
focus_gain = summary.loc[focus, "gain_years"]
focus_rank = summary.loc[focus, f"rank_{end}"]

range_reduction = 100 * (1 - spread.loc[end] / spread.loc[start])

# Round display limits (not data).
low = 5 * np.floor(wide.min().min() / 5)
high = 5 * np.ceil(wide.max().max() / 5)


# Summary results
print("\nEndpoint results:")
print(summary.round(1).to_string())

print("\nBetween-country ranges:")
print(spread.round(1).to_string())

print(f"\nLargest gain: {focus}, {focus_gain:.1f} years")
print(f"Highest value in {end}: {leader}, {summary.loc[leader, end]:.1f}")
print(f"Endpoint range reduction: {range_reduction:.1f}%")
print(
    f"Sampled minimum: {spread.min():.1f} years "
    f"in {minimum_text}"
)

# Helper functions
## Shared styling
styles[focus] = (BLUE, "-")

drawing_order = [country for country in wide.index if country != focus]
drawing_order.append(focus)


def save(fig, filename, title=None, note=""):
    """Add headings and export."""
    if title is not None:
        fig.text(
            0.02, 0.97, title,
            fontsize=13, weight="bold", va="top"
        )
        fig.text(
            0.02, 0.02, f"{note}\n{SOURCE}",
            fontsize=8.5, va="bottom"
        )

    for extension in ("png", "svg"):
        fig.savefig(
            out + f"{filename}.{extension}",
            dpi=300, bbox_inches="tight"
        )

    plt.close(fig)


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E2E2E2", linewidth=0.7)


def label_positions(values, gap=1.3):
    """Move labels while keeping data points in their true positions."""
    positions = values.sort_values().copy()

    for i in range(1, len(positions)):
        positions.iloc[i] = max(
            positions.iloc[i],
            positions.iloc[i - 1] + gap
        )

    return positions - (positions - values).mean()

# ----------------------------------------------------------------------------

# default plots: Fig 1

plt.rcdefaults()
pd.options.plotting.backend = "matplotlib"

ax = endpoints.plot()
save(ax.figure, "figure_1_reference")


# styling done after making the default plots.
plt.rcParams.update({
    "font.size": 11,
    "svg.fonttype": "none"
})

# ----------------------------------------------------------------------------

# Fig 2: grouped horizontal bars

fig, ax = plt.subplots(figsize=(7.6, 5.0))
fig.subplots_adjust(left=0.17, right=0.81, bottom=0.05, top=0.76)

y = np.arange(len(ordered))

for year, offset, color, hatch in [
    (start, -0.17, "#BDBDBD", "///"),
    (end, 0.17, "#555555", None),
]:
    bars = ax.barh(
        y + offset, ordered[year], height=0.29,
        color=color, hatch=hatch, label=str(year),
        edgecolor="#444444", linewidth=0.5
    )
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)

ax.set_yticks(y, ordered.index)
ax.invert_yaxis()
ax.set(
    xlim=(0, high + 5),
    xlabel="Life expectancy at birth (years)"
)

ax.get_yticklabels()[0].set_color(BLUE)
ax.get_yticklabels()[0].set_weight("bold")

clean_axes(ax, "x")
ax.legend(
    loc="lower left", bbox_to_anchor=(0, 1.01),
    ncol=2, frameon=False
)

ax.text(
    1.04, 1.03, "Gain\n(years)",
    transform=ax.transAxes, weight="bold", va="bottom"
)

for row, country in enumerate(ordered.index):
    ax.text(
        1.04, row, f"{ordered.loc[country, 'gain_years']:+.1f}",
        transform=ax.get_yaxis_transform(), va="center",
        color=BLUE if country == focus else "#333333"
    )

save(
    fig, "figure_2_grouped_bars",
    "",
    ""
)
