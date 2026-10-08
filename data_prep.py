"""
data_prep.py
Turns the raw WA DOL "Electric Vehicle Title and Registration Activity" CSV
into the used-resale table the agent uses (used_resales_clean.csv).
Same cleaning rules as our CP3 report.
Run:  python data_prep.py "Electric_Vehicle_Title_and_Registration_Activity.csv"
"""
import sys
import numpy as np
import pandas as pd

raw_path = sys.argv[1]
ev = pd.read_csv(raw_path, low_memory=False)
print("All transactions:", len(ev))

# purchases only, Jan 2019 - Jun 2026, passenger vehicles and trucks
ev = ev[ev["Transaction Type"].isin(["Original Title", "Transfer Title"])].copy()
ev["sale_date"] = pd.to_datetime(ev["Sale Date"], errors="coerce")
ev = ev[(ev["sale_date"] >= "2019-01-01") & (ev["sale_date"] <= "2026-06-30")]
ev = ev[ev["Primary Use"].isin(["Passenger", "Truck"])]
ev["price"] = pd.to_numeric(ev["Sale Price"], errors="coerce")
ev["model_year"] = pd.to_numeric(ev["Model Year"], errors="coerce")
ev = ev[(ev["sale_date"].dt.year - ev["model_year"]) >= -1]
print("Purchases kept:", len(ev))

# original new price of each vehicle (first sale as New in WA)
new = ev[(ev["New or Used Vehicle"] == "New") & (ev["price"] >= 1000)]
orig = new.sort_values("sale_date").groupby("DOL Vehicle ID")["price"].first()

# used resales within WA
used = ev[(ev["Transaction Type"] == "Transfer Title") & (ev["New or Used Vehicle"] == "Used")].copy()
used = used[used["Odometer Reading Description"] == "Actual Mileage"]
used["odometer"] = pd.to_numeric(used["Odometer Reading"], errors="coerce")
used.loc[(used["odometer"] < 0) | (used["odometer"] > 300000), "odometer"] = np.nan

# prices: $0 / under $1,000 = not a market sale
used = used[used["price"] >= 1000]

# make/model price check (replaces the flat $250k cap): flag > 3x the model median
med = used.groupby(["Make", "Model"])["price"].transform("median")
cnt = used.groupby(["Make", "Model"])["price"].transform("count")
bad = (cnt >= 30) & (used["price"] > 3 * med)
print("Prices flagged by make/model check:", int(bad.sum()))
used = used[~bad]

used["fuel"] = np.where(used["Clean Alternative Fuel Vehicle Type"].str.contains("Battery", na=False), "BEV", "PHEV")
used["original_price"] = used["DOL Vehicle ID"].map(orig)

out = used.rename(columns={"DOL Vehicle ID": "vehicle_id", "Make": "make", "Model": "model"})
out = out[["vehicle_id", "make", "model", "fuel", "model_year", "sale_date", "price", "odometer", "original_price"]]
before = len(out)
out = out.drop_duplicates(subset=["vehicle_id", "sale_date"])
print("Duplicate resales removed:", before - len(out))
out = out.dropna(subset=["odometer"])
out.to_csv("used_resales_clean.csv", index=False)
print("Wrote used_resales_clean.csv:", out.shape)
