import pandas as pd

df = pd.read_csv("data/nyc_raw/yellow_tripdata_2019-02.csv", nrows=5)
print(df.columns)
