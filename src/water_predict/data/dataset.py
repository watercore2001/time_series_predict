import torch
from torch.utils.data import Dataset
from typing import Literal
import pandas as pd
import numpy as np

STATION_NUM = 147
WATERSHED_NUM = 9
WEEK_NUM = 53

STATION = "Station"
WATERSHED = "Watershed"
WEEK_OF_YEAR = "WeekOfYear"
LONGITUDE = "Longitude"
LATITUDE = "Latitude"
FEATURE_NAMES = ["CODMn", "DO", "NH4N", "pH"]


def int_to_tensor(x):
    return torch.as_tensor(x, dtype=torch.int32).contiguous()


def float_to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32).contiguous()


class StationDataset(Dataset):
    TRAIN_PERCENT = 0.9

    def __init__(self, flag: Literal["train", "val"], data: pd.DataFrame,
                 x_length: int, y_length: int):
        super().__init__()
        # 1. save metadata
        self.station_id = int(data[STATION].mean())
        self.lat_lon = np.array([data[LATITUDE].mean(), data[LONGITUDE].mean()])
        self.watershed_id = int(data[WATERSHED].mean())
        self.x_length = x_length
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
        self.data = data[border1:border2]

    def __len__(self):
        return self.length - self.x_length - self.y_length + 1

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.x_length
        y_begin = x_end
        y_end = y_begin + self.y_length

        x_feature = self.data[x_begin:x_end][FEATURE_NAMES].values
        x_week_of_years = self.data[x_begin:x_end][WEEK_OF_YEAR].values
        y_feature = self.data[y_begin:y_end][FEATURE_NAMES].values
        y_week_of_years = self.data[y_begin:y_end][WEEK_OF_YEAR].values

        return {"x": float_to_tensor(x_feature),
                "y": float_to_tensor(y_feature),
                "x_week_of_years": int_to_tensor(x_week_of_years),
                "y_week_of_years": int_to_tensor(y_week_of_years),
                "station_id": int_to_tensor(self.station_id),
                "watershed_id": int_to_tensor(self.watershed_id),
                "lat_lng": float_to_tensor(self.lat_lon)}


class FitDataset(Dataset):
    def __init__(self, file_path: str, flag: Literal["train", "val"], watershed_ids: list[int],
                 x_length: int, y_length: int):
        super().__init__()
        data = pd.read_csv(file_path)
        data = data[data[WATERSHED].isin(watershed_ids)]

        self.station_datasets = []

        for _, station_df in data.groupby(STATION):
            # Important reset index
            self.station_datasets.append(StationDataset(flag, station_df.reset_index(drop=True),
                                                        x_length, y_length))

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


class TestDataset(Dataset):
    def __init__(self, file_path: str, watershed_ids: list[int]):
        data = pd.read_csv(file_path)
        data = data[data[WATERSHED].isin(watershed_ids)]
        self.station_groups = []
        for _, station_data in data.groupby(STATION):
            self.station_groups.append(station_data)

    def __len__(self):
        return len(self.station_groups)

    def __getitem__(self, idx: int):
        data = self.station_groups[idx]
        station_id = int(data[STATION].mean())
        lat_lon = np.array([data[LATITUDE].mean(), data[LONGITUDE].mean()])
        watershed_id = int(data[WATERSHED].mean())
        week_of_years = data[WEEK_OF_YEAR].values
        feature = data[FEATURE_NAMES].values

        return {"feature": float_to_tensor(feature),
                "week_of_years": int_to_tensor(week_of_years),
                "station_id": int_to_tensor(station_id),
                "watershed_id": int_to_tensor(watershed_id),
                "lat_lng": float_to_tensor(lat_lon)}


if __name__ == "__main__":
    # minimal station length is 64

    overall_dataset = FitDataset("../../../dataset/weekly_land.csv",
                                 watershed_ids=list([0, 1]),
                                 flag="train", x_length=32, y_length=8)
    #print(overall_dataset[0])
    #print(overall_dataset[1])
    data = overall_dataset[172]
    for i in range(len(overall_dataset)):
        assert overall_dataset[i]["x"].shape == (32, 4), i
    from torch.utils.data import DataLoader
    loader = DataLoader(overall_dataset, batch_size=2, shuffle=False)
    for batch in loader:
        print(batch)
        break
    print(len(loader))
    pass

