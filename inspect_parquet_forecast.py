import pandas as pd 
# read /home/skooiker/ml_macro_at_risk/outputs/forecasts/q=0.1/h=1/rolling_window.parquet
df = pd.read_parquet("/home/skooiker/ml_macro_at_risk/outputs/forecasts/q=0.1/h=1/rolling_window.parquet")
print(df)

# store df
df.to_parquet("/home/skooiker/ml_macro_at_risk/outputs/forecasts/q=0.1/h=1/rolling_window.parquet")

# drop duplicates
df = df.drop_duplicates()

# print number of entries per country per model
print(df.groupby(["COUNTRY", "MODEL"]).size())

# remove 