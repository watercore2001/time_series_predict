import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv('../../../dataset/weekly_land_backup.csv')

    # 1. Station length 147
    data["Station"] = pd.factorize(data["Station"], sort=True)[0]
    print(data["Station"].value_counts())
    # 2. Watershed length 9
    data["Watershed"] = pd.factorize(data["Watershed"], sort=True)[0]
    print(data["Watershed"].value_counts())
    # 3. Week length 53
    # Day of Year is insufficient, So use Week of Year
    date = pd.to_datetime(data["Date"])
    data = data.drop("Date", axis=1)
    data["WeekOfYear"] = date.apply(lambda d: d.weekofyear)
    data["WeekOfYear"] = pd.factorize(data["WeekOfYear"], sort=True)[0]
    print(data["WeekOfYear"].value_counts())
    # 4. Features
    features = data[["Longitude", "Latitude", "CODMn", "DO", "NH4N", "pH"]]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    features.fillna(0, inplace=True)
    data[["Longitude", "Latitude", "CODMn", "DO", "NH4N", "pH"]] = pd.DataFrame(features)
    # save
    data.to_csv("weekly_land.csv", index=False)


if __name__ == "__main__":
    main()
