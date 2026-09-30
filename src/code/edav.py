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

# ----------------------------------------------------------------------------

# Fig 3: slopegraph

fig, ax = plt.subplots(figsize=(8.2, 5.7))
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.05, top=0.83)

left_labels = label_positions(endpoints[start])
right_labels = label_positions(endpoints[end])

for country in drawing_order:
    first, last = endpoints.loc[country]
    gain = summary.loc[country, "gain_years"]
    color, line_style = styles[country]

    ax.plot(
        [0, 1], [first, last],
        color=color, linestyle=line_style, marker="o",
        linewidth=2.8 if country == focus else 1.4,
        markersize=5
    )

    ax.plot(
        [-0.04, 0], [left_labels[country], first],
        color=color, linewidth=0.7
    )
    ax.plot(
        [1, 1.04], [last, right_labels[country]],
        color=color, linewidth=0.7
    )

    ax.text(
        -0.06, left_labels[country],
        f"{country}  {first:.1f}",
        ha="right", va="center", color=color, fontsize=10
    )
    ax.text(
        1.06, right_labels[country],
        f"{country}  {last:.1f}  ({gain:+.1f})",
        ha="left", va="center", color=color, fontsize=10
    )

ax.set(
    xlim=(-0.70, 1.95), ylim=(low, high),
    ylabel="Life expectancy at birth (years)"
)
ax.set_xticks([0, 1], [str(start), str(end)])
ax.set_yticks(np.arange(low, high + 1, 5))
ax.tick_params(axis="x", length=0)

clean_axes(ax)
ax.spines["bottom"].set_visible(False)

save(
    fig, "figure_3_slopegraph",
    "",
    ""
)

# ----------------------------------------------------------------------------

# Fig 4: annotated heatmap
temporary_low = low
temporary_high = high

# override:
low = 64
high = 82
bottom=0.05

fig, ax = plt.subplots(figsize=(7.6, 4.8))
fig.subplots_adjust(left=0.18, right=0.90, bottom=bottom, top=0.79)

matrix = ordered[[start, end]].to_numpy()

image = ax.pcolormesh(
    matrix, cmap="Greys", vmin=low, vmax=high,
    edgecolors="white", linewidth=0.8
)

# The third column has no fill.
ax.set(xlim=(0, 3.1), ylim=(len(ordered), 0))
ax.set_xticks([0.5, 1.5], [str(start), str(end)])
ax.set_yticks(np.arange(len(ordered)) + 0.5, ordered.index)

ax.tick_params(
    axis="x", top=True, labeltop=True,
    bottom=False, labelbottom=False, length=0
)
ax.tick_params(axis="y", length=0, pad=8)

for spine in ax.spines.values():
    spine.set_visible(False)

for row, country in enumerate(ordered.index):
    for column, year in enumerate((start, end)):
        value = ordered.loc[country, year]
        shade = image.cmap(image.norm(value))[0]

        ax.text(
            column + 0.5, row + 0.5, f"{value:.1f}",
            ha="center", va="center",
            color="white" if shade < 0.5 else "black"
        )

    ax.text(
        2.6, row + 0.5,
        f"{ordered.loc[country, 'gain_years']:+.1f}",
        ha="center", va="center",
        color=BLUE if country == focus else "#333333"
    )

ax.text(
    2.6, -0.05, "Gain\n(years)",
    ha="center", va="bottom", weight="bold"
)
ax.get_yticklabels()[0].set_color(BLUE)
ax.get_yticklabels()[0].set_weight("bold")

colorbar = fig.colorbar(image, ax=ax, pad=0.05, fraction=0.05)
colorbar.set_ticks(np.arange(low, high + 1, 3))  #############################
colorbar.set_label("Life expectancy (years)")

save(
    fig, "figure_4_heatmap",
    "",
    ""
)

low = temporary_low
high = temporary_high

# ----------------------------------------------------------------------------

# Fig 5: more time-detail

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(8.2, 7.2), sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1]}
)
fig.subplots_adjust(
    left=0.13, right=0.95, bottom=0.05, top=0.84, hspace=0.30
)

for country in drawing_order:
    color, line_style = styles[country]

    top.plot(
        years, wide.loc[country],
        color=color, linestyle=line_style,
        marker="o", markersize=3,
        linewidth=2.4 if country == focus else 1.4
    )
    top.plot(
        [end, end + 2],
        [wide.loc[country, end], right_labels[country]],
        color=color, linewidth=0.7
    )
    top.text(
        end + 2.3, right_labels[country], country,
        color=color, va="center", fontsize=10
    )

top.set(
    ylim=(low, high),
    ylabel="Life expectancy at birth\n(years)"
)
top.set_title(
    "A. Selected five-yearly values", loc="left", fontsize=11
)
clean_axes(top)

bottom.plot(
    spread.index, spread,
    color="#333333", marker="o", markersize=4
)
bottom.scatter(
    minimum_years, spread.loc[minimum_years],
    color=BLUE, zorder=3
)

bottom.annotate(
    f"Sampled minimum: {spread.min():.1f} years\n"
    f"in {minimum_text}",
    xy=(minimum_years[0], spread.loc[minimum_years[0]]),
    xytext=(0.28, 0.85), textcoords="axes fraction",
    va="top", fontsize=10,
    arrowprops={"arrowstyle": "->", "color": "#555555"}
)

for year in (start, end):
    bottom.text(
        year, spread.loc[year] + 0.25,
        f"{spread.loc[year]:.1f}", ha="center", fontsize=10
    )

bottom.set(
    xlim=(start - 2, end + 12),
    ylim=(0, spread.max() + 1),
    xlabel="Year",
    ylabel="Highest–lowest range\n(years)"
)
bottom.set_xticks(sorted(set(years[::2] + [end])))
bottom.set_title(
    "B. Between-country range", loc="left", fontsize=11
)
clean_axes(bottom)

save(
    fig, "figure_5_temporal",
    "",
    ""
)