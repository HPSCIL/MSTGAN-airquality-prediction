
import numpy as np
import torch

def pro_data(data,args):
    trainX,trainY,week_train,sta_train,change= [],[],[],[],[]
    # Divide the training set and validation set
    for i in range(len(data)-args.lag-args.pre_len):
        x_train = data[i:i+args.lag:,:,2:8]   #[48,35,6]
        target_train = data[i+args.lag:i+args.lag+args.pre_len,:,7:8]  #[24,35,1]
        week_pre_train = data[i + args.lag:i + args.lag + args.pre_len, :, 8:9]  # [24,35,1]
        sta_pre_train = data[i + args.lag:i + args.lag + args.pre_len, :, 0:1]
        change_train = data[i + args.lag:i + args.lag + args.pre_len, :, 9:10]
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





def data_loader(X, Y,date,sta,change, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X,Y,date,sta,change= TensorFloat(X),TensorFloat(Y),TensorFloat(date),TensorFloat(sta),TensorFloat(change)
    data = torch.utils.data.TensorDataset(X,Y,date,sta,change)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    return dataloader

