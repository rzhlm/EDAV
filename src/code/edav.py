import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# INIT:
## constants
out = "./outputs/"

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

