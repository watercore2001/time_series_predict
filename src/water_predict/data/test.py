import pandas as pd
data = pd.read_csv('../../../data/weekly_land.csv')
grouped = data.groupby('Station')
for name, group in grouped:
    data = group
pass
