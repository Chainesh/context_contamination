import os
import pandas as pd

root = os.path.join(os.path.dirname(__file__), "..")
out = os.path.join(root, "data", "raw", "us_births.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)

cdc = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994-2003_CDC_NCHS.csv")
ssa = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_2000-2014_SSA.csv")

df = pd.concat([cdc[cdc.year < 2000], ssa], ignore_index=True)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.date_of_month))
df = df.sort_values("date")[["date", "births"]]
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, {df.date.min().date()} to {df.date.max().date()}")