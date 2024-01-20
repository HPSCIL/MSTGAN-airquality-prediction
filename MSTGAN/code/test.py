import torch
from visualizer import get_local
get_local.activate()
from MSTAN.code.model.MSTAN import MSTAN
import pandas as pd
import numpy as np
from code.utils.All_Metrics import All_Metrics
from code.utils.scaled_Laplacian import scaled_Laplacian,cheb_polynomial

model_name = 'MSTAN'
window_size = 24
pre_len = 12
best_path = '../experiments/in_{}h_out_{}h/best_model.pth'.format(window_size,pre_len)

#load dataset
dataframe = pd.read_excel('E:\大电脑\补充实验\预测实验代码\TSTGCRN\预测统计/Beijing_PM25.xlsx')
dataset_aqi = dataframe.values
dataset_aqi = dataset_aqi.astype('float32')
dataset = dataset_aqi[:, 1:11]
dataset = np.reshape(dataset, (35, -1, 10))   #[35,10806,9]
data = dataset.transpose(1,0,2)   #[10806,35,9]
train_size = int(len(data)*0.8)
test_data = data[train_size:]

device = torch.device('cuda')


def data_loader(X, Y,date,sta,change, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X,Y,date,sta,change= TensorFloat(X),TensorFloat(Y),TensorFloat(date),TensorFloat(sta),TensorFloat(change)
    data = torch.utils.data.TensorDataset(X,Y,date,sta,change)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    return dataloader



def pro_data(data,lag,pre_len):
    trainX,trainY,week_train,sta_train,change= [],[],[],[],[]
    # 训练数据的划分
    for i in range(0,(len(data)-lag-pre_len)):
        x_train = data[i:i+lag:,:,2:8]   #[48,35,6]
        target_train = data[i+lag:i+lag+pre_len,:,7:8]  #[24,35,1]
        week_pre_train = data[i + lag:i + lag + pre_len, :, 8:9]  # [24,35,1]
        sta_pre_train = data[i + lag:i + lag + pre_len, :, 0:1]
        change_train = data[i + lag:i + lag + pre_len, :, 9:10]
        sta_train.append(sta_pre_train)
        week_train.append(week_pre_train)
        change.append(change_train)
        trainX.append(x_train)
        trainY.append(target_train)
    trainX = np.array(trainX).transpose(0, 2, 3, 1)  #[8614,35,6,24]
    trainY = np.array(trainY).transpose(0,2,3,1)   # [8614,35,1,24]
    week_train = np.array(week_train).transpose(0, 2, 3, 1)  # [8614,35,1,24]
    sta_train = np.array(sta_train).transpose(0, 2, 3, 1) #[8614,35,1,24]
    change = np.array( change).transpose(0, 2, 3, 1)

    return trainX,trainY,week_train,sta_train,change
def input_data(seq,ws,pre_len):
    X = []
    Y = []
    L = len(seq)
    for i in range(0,(L-ws-pre_len),pre_len):
        lag = seq[i:i+ws]
        target = seq[i+ws:i+ws+pre_len,:,5]
        X.append(lag)
        Y.append(target)
    X = np.array(X).transpose(0,2,3,1) #[32,24,35,1]
    Y = np.array(Y).transpose(0,2,1)  #[32,12,35,1]
    return X,Y


testX, testY, week_test, sta_test, change_test = pro_data(test_data,window_size,pre_len)
test_loader = data_loader(testX, testY,week_test,sta_test,change_test,batch_size=32, shuffle=False, drop_last=True)

adj = pd.read_csv('E:\大电脑\实验\图数据/dis.csv', header=None)
adj_mx = np.mat(adj).astype(float)
L_tilde = scaled_Laplacian(adj_mx)
cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L_tilde, 3)]
#Test

model = MSTAN(input_dim=6,hiden_dim=64,out_channels=64,device=device,num_nodes=35,num_of_timesteps=window_size,num_for_predict=pre_len,K=3,dropout=0.1,d_model=512).to(device)
model.load_state_dict(torch.load(best_path))
model.eval()
test_outputs = torch.empty([0, 35, 1, pre_len]).to(device)
test_target = torch.empty([0, 35, 1, pre_len]).to(device)
test_Date = torch.empty([0, 35, 1, pre_len]).to(device)
test_Sta = torch.empty([0, 35, 1, pre_len]).to(device)
test_Change = torch.empty([0, 35, 1, pre_len]).to(device)
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        test_X, test_Y,test_week,test_sta,test_change= batch_data
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)
        test_week = test_week.to(device)
        test_sta = test_sta.to(device)
        test_change =test_change.to(device)
        test_output = model(test_X,cheb_polynomials)
        cache = get_local.cache
        attention_maps = cache['Transform.forward']
        test_outputs = torch.cat((test_outputs, test_output), 0)  #[?,35,1,6]
        test_target = torch.cat((test_target, test_Y), 0)  #[?,35,1,6]
        test_Date = torch.cat((test_Date,test_week),0)   #[?,35,1,6]
        test_Sta = torch.cat((test_Sta,test_sta),0)   #[?,35,1,6]
        test_Change = torch.cat((test_Change, test_change), 0)  # [?,35,1,6]
    mae, rmse, mape, r2 = All_Metrics(test_outputs, test_target)
    a = torch.cat((test_Sta,test_Date,test_Change,test_outputs,test_target),dim=2)  #[2112,35,4,6]
    attention_map = np.array(attention_maps)
    np.save('../Prediction_results/{}_{}h_attention_map.npy'.format(model_name, pre_len), attention_map)
    np.save('../Prediction_results/{}_{}h_test.npy'.format(model_name,pre_len), a.cpu().numpy())
    for t in range(test_target.shape[2]):
        maes, rmses, mapes, r2_s = All_Metrics(test_outputs[:,:, t], test_target[:,:, t])
        print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, R2:{:.4f}".format(t + 1, maes, rmses, mapes, r2_s))
print('loss1:{},rmse1:{},mape1:{},r2_1:{}'.format(mae, rmse, mape, r2))


# np.save('E:\大电脑\补充实验\预测实验代码\保存预测真实文件/TSTGCRN_true_2018.npy'.format(pre_len),test_target.cpu().numpy())
# np.save('E:\大电脑\补充实验\预测实验代码\保存预测真实文件/TSTGCRN_pre_2018.npy'.format(pre_len),test_outputs.cpu().numpy())