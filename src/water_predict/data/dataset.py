from torch.utils.data import Dataset
from typing import Literal
import pandas as pd
import numpy as np
# Station, Watershed, WeekOfYear, Longitude, Latitude, CODMn, DO, NH4N, pH,


class StationDataset(Dataset):
    TRAIN_PERCENT = 0.9

    def __init__(self, flag: Literal["train", "val"], data: pd.DataFrame,
                 x_length: int, share_length: int, y_length: int):
        super().__init__()
        # 1. save metadata
        self.station_id = int(data["Station"].mean())
        self.lat_lon = np.array([data["Latitude"].mean(), data["Longitude"].mean()])
        self.watershed_id = int(data["Watershed"].mean())
        self.x_length = x_length
        self.share_length = share_length
        self.y_length = y_length

        # 2. Split data
        length = data.shape[0]
        train_border = int(length * self.TRAIN_PERCENT)
        if flag == "train":
            border1 = 0
            border2 = train_border
        elif flag == "val":
            border1 = train_border - x_length
            border2 = length
        else:
            raise ValueError(f"Invalid flag {flag}")
        self.length = border2 - border1
        self.data = data.loc[border1:border2].reset_index(drop=True)

    def __len__(self):
        return self.length - self.x_length - self.y_length + 1

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.x_length
        y_begin = x_end - self.share_length
        y_end = y_begin + self.share_length + self.y_length

        x_feature = self.data.loc[x_begin:x_end, ["CODMn", "DO", "NH4N", "pH"]].values
        x_week_of_years = self.data.loc[x_begin:x_end, ["WeekOfYear"]].values
        y_feature = self.data.loc[y_begin:y_end, ["CODMn", "DO", "NH4N", "pH"]].values
        y_week_of_years = self.data.loc[y_begin:y_end, ["WeekOfYear"]].values

        return {"x": x_feature, "x_week_of_years": x_week_of_years,
                "y": y_feature, "y_week_of_years": y_week_of_years,
                "station_id": self.station_id, "watershed_id": self.watershed_id,
                "lat_lng": self.lat_lon}


class OverallDataset(Dataset):
    def __init__(self, file_path: str, flag: Literal["train", "val"],
                 x_length: int, share_length: int, y_length: int):
        super().__init__()

        self.station_datasets = []
        data = pd.read_csv(file_path)
        for _, station_df in data.groupby("Station"):
            self.station_datasets.append(StationDataset(flag, station_df,
                                                        x_length, share_length, y_length))

        self.cumulative_lengths = [0]
        for dataset in self.station_datasets:
            # print(len(dataset))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int):
        dataset_index = 0
        while self.cumulative_lengths[dataset_index + 1] <= idx:
            dataset_index += 1
        inner_idx = idx - self.cumulative_lengths[dataset_index]
        return self.station_datasets[dataset_index][inner_idx]


if __name__ == "__main__":
    # minimal station length is 64

    overall_dataset = OverallDataset("/mnt/code/course/time_series_predict/data/weekly_land.csv",
                                     "val", x_length=32, share_length=8, y_length=8)
    # print(overall_dataset[0])
    # print(overall_dataset[1])
    from torch.utils.data import DataLoader
    loader = DataLoader(overall_dataset, batch_size=2, shuffle=False)
    for batch in loader:
        print(batch)
        break
    print(len(loader))
    pass

