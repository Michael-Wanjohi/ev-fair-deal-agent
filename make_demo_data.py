"""
make_demo_data.py
Creates FAKE used-EV resale data with the same columns as our cleaned WA table,
so the prototype runs before the real data is plugged in.
DO NOT use these numbers in any report - they are made up.
Run:  python make_demo_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(28)

# make, model, fuel, typical new price
cars = [
    ("TESLA", "MODEL 3", "BEV", 45000), ("TESLA", "MODEL Y", "BEV", 52000),
    ("TESLA", "MODEL S", "BEV", 85000), ("NISSAN", "LEAF", "BEV", 32000),
    ("CHEVROLET", "BOLT EV", "BEV", 34000), ("KIA", "NIRO", "BEV", 41000),
    ("FORD", "MUSTANG MACH-E", "BEV", 50000), ("HYUNDAI", "IONIQ 5", "BEV", 47000),
    ("VOLKSWAGEN", "ID.4", "BEV", 44000), ("BMW", "I3", "BEV", 46000),
    ("CHEVROLET", "VOLT", "PHEV", 34000), ("TOYOTA", "PRIUS PRIME", "PHEV", 30000),
    ("JEEP", "WRANGLER", "PHEV", 55000), ("RIVIAN", "R1T", "BEV", 75000),
]
weights = np.array([30, 20, 4, 10, 8, 4, 5, 4, 4, 2, 4, 3, 2, 1], dtype=float)
weights /= weights.sum()

n = 30000
idx = rng.choice(len(cars), size=n, p=weights)
sale_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 2737, n), unit="D")
age = rng.integers(1, 10, n)
model_year = sale_date.year - age
miles_per_year = rng.normal(11000, 3500, n).clip(2000, 30000)
odometer = (age * miles_per_year).round()

rows = []
for i in range(n):
    make, model, fuel, new_price = cars[idx[i]]
    keep = 0.86 ** age[i] * (1 - 0.0000012 * (odometer[i] - age[i] * 11000))
    price = new_price * keep * rng.lognormal(0, 0.12)
    rows.append((f"V{i:06d}", make, model, fuel, int(model_year[i]), sale_date[i].date(),
                 round(price, -1), odometer[i], round(new_price * rng.lognormal(0, 0.05), -1)))

df = pd.DataFrame(rows, columns=["vehicle_id", "make", "model", "fuel", "model_year",
                                 "sale_date", "price", "odometer", "original_price"])
df.loc[rng.random(n) < 0.3, "original_price"] = np.nan   # ~30% unknown, like the real data
df.to_csv("used_resales_clean.csv", index=False)
print("Wrote used_resales_clean.csv (DEMO DATA):", df.shape)
