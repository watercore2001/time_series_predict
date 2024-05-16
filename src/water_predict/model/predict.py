import torch
from torch.utils.data import DataLoader
from water_predict.model.transformer import Transformer
from water_predict.data.dataset import PredictDataset
from einops import rearrange

ckpt_path = "/mnt/code/course/time_series_predict/dataset/transformer.ckpt"
file_path = "/mnt/code/course/time_series_predict/dataset/weekly_land.csv"
watershed_ids = [0]
PREDICT_LENGTH = 32

def add_thickness(tensor, thickness=2):
    # 获取原始 Tensor 的形状
    original_shape = tensor.shape

    # 创建一个厚度为 thickness 的零 Tensor
    zeros = torch.zeros(thickness, *original_shape[1:]).to(tensor.device)

    # 将原始 Tensor 和零 Tensor 拼接起来
    new_tensor = torch.cat((zeros, tensor), dim=0)

    return new_tensor

def predict(model, batch):
    output = torch.zeros_like(batch["feature"])
    output[:, :PREDICT_LENGTH] = batch["feature"][:, :PREDICT_LENGTH]
    b, length, c = output.shape

    for i in range(PREDICT_LENGTH, length):
        batch["x"] = batch["feature"][:, i - PREDICT_LENGTH:i]
        batch["x_week_of_years"] = batch["week_of_years"][:, i - PREDICT_LENGTH:i]
        batch["y"] = torch.zeros(b, 1, c).to(batch["x"].device)
        batch["y_week_of_years"] = batch["week_of_years"][:, i].reshape(-1, 1)

        batch["x"] = add_thickness(batch["x"])
        batch["x_week_of_years"] = add_thickness(batch["x_week_of_years"]).to(torch.int32)
        batch["y"] = add_thickness(batch["y"])
        batch["y_week_of_years"] = add_thickness(batch["y_week_of_years"]).to(torch.int32)

        y_hat = model(batch)
        y_hat = rearrange(y_hat, "b 1 c -> b c")
        output[:, i] = y_hat


def remove_model_prefix(d):
    new_dict = {}
    for key, value in d.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            new_dict[new_key] = value
    return new_dict


if __name__ == "__main__":
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))

    model = Transformer(c_in=4, c_out=4, d_model=128,
        encoder_depth=2,
        decoder_depth=1,
        use_station=True,
        use_watershed= True,
        use_latlng=False)



    msg = model.load_state_dict(remove_model_prefix(ckpt_dict['state_dict']))
    print(msg)

    model.eval()

    dataset = PredictDataset(file_path=file_path, watershed_ids=[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        predict(model, batch)




