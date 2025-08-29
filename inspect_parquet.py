import pandas as pd 
# read /home/skooiker/ml_macro_at_risk/outputs/forecasts/q=0.1/h=1/rolling_window.parquet
df = pd.read_parquet("/home/skooiker/ml_macro_at_risk/outputs/progress/claimed_tasks.parquet")

# get unique rows
df = df.drop_duplicates()

# # store df 
# df.to_parquet("/home/skooiker/ml_macro_at_risk/outputs/progress/claimed_tasks.parquet")

# order by COUNTRY and WINDOW_START
df = df.sort_values(by=["COUNTRY", "WINDOW_START"])

# remove lqr from MODEL
df = df[df["MODEL"] != "lqr"]

# write to readable file with proper indentation
with open("/home/skooiker/ml_macro_at_risk/outputs/progress/progress_claimed.txt", "w") as f:
    f.write(df.to_string())
    f.write("\n")


# count rows per country and MODEL
for (country, model), group in df.groupby(["COUNTRY", "MODEL"]):
    print(f"Country: {country}, Model: {model}, Rows: {len(group)}")


# read /home/skooiker/ml_macro_at_risk/outputs/forecasts/q=0.1/h=1/rolling_window.parquet
df = pd.read_parquet("/home/skooiker/ml_macro_at_risk/outputs/progress/completed_tasks.parquet")

# unique
df = df.drop_duplicates()

# # store df
# df.to_parquet("/home/skooiker/ml_macro_at_risk/outputs/progress/completed_tasks.parquet", index=False)

# order by COUNTRY and WINDOW_START
df = df.sort_values(by=["COUNTRY", "WINDOW_START"])

# remove lqr from MODEL
df = df[df["MODEL"] != "lqr"]

# write to readable file with proper indentation
with open("/home/skooiker/ml_macro_at_risk/outputs/progress/completed_claimed.txt", "w") as f:
    f.write(df.to_string())
    f.write("\n")


# count rows per country and MODEL
for (country, model), group in df.groupby(["COUNTRY", "MODEL"]):
    print(f"Country: {country}, Model: {model}, Rows: {len(group)}")

# # drop rows where STATUS is CLAIMED
# df = df[df["STATUS"] != "claimed"]

# # store again
# df.to_parquet("/home/skooiker/ml_macro_at_risk/outputs/progress/progress.parquet")



