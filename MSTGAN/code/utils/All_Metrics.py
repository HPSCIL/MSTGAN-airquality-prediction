
import torch

def All_Metrics(pred, true):
   # print(pred.shape,true.shape)
    mae = MAE_torch(pred, true)
    rmse = RMSE_torch(pred, true)
    mape = MAPE_torch(pred, true)
    r2 = r2_torch(pred, true)
    return mae, rmse, mape,r2


def MAE_torch(pred, true):
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def MAPE_torch(pred, true):
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def r2_torch(pred, true):
    target_mean = torch.mean(true)
    ss_tot = torch.sum((true - target_mean) ** 2)
    ss_res = torch.sum((true - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

