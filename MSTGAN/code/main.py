import os
import copy
import time
import torch
import argparse
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from MSTAN.code.utils.init_seed import  init_seed
from MSTAN.code.utils.pro_data import pro_data,data_loader
from MSTAN.code.model.MSTAN import MSTAN
from MSTAN.code.utils.get_logger import get_logger
from MSTAN.code.utils.All_Metrics import All_Metrics
from MSTAN.code.utils.scaled_Laplacian import scaled_Laplacian
from MSTAN.code.utils.scaled_Laplacian import cheb_polynomial
from adjust_learning_rate import adjust_learning_rate



Model = 'MSTAN'
device = torch.device('cuda')

args = argparse.ArgumentParser(description='arguments')

args.add_argument('--train_rate', default=0.8, type=float,help='rate of train set.')
args.add_argument('--test_rate', default=0.2, type=float,help='rate of test set.')
args.add_argument('--lag', default=24, type=int,help='time length of inputs.')
args.add_argument('--pre_len', default=18, type=int)
args.add_argument('--num_nodes', default=35, type=int)
args.add_argument('--batch_size', default=32, type=int)
args.add_argument('--input_dim', default=6, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--learning_rate', default=0.001, type=float)
args.add_argument('--cheb_k', default=3, type=int)
args.add_argument('--block1_hidden',default=64,type=int)
args.add_argument('--block2_hiden',default=64,type=int)
args.add_argument('--time_strides',default=1,type=int)
args.add_argument('--nb_block',default=2,type=int)
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--debug', default=False, type=eval)
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=10, type=int)
args.add_argument('--seed', default=20, type=int)
args.add_argument('--device', type=eval, default=device)
args.add_argument('--dropout', type=eval, default=0.1)
args.add_argument('--d_model', type=int, default=512)
args = args.parse_args()
init_seed(args.seed)





#load dataset
dataframe = pd.read_excel('../Data/Beijing_PM25.xlsx')
dataset_aqi = dataframe.values
dataset_aqi = dataset_aqi.astype('float32')
dataset = dataset_aqi[:, 1:11]
dataset = np.reshape(dataset, (35, -1, 10))
data = dataset.transpose(1,0,2)



train_size = int(data.shape[0] * args.train_rate)
train_data = data[:train_size]
test_data = data[train_size:]


trainX,trainY,week_train,sta_train,change_train = pro_data(train_data,args)
testX,testY,week_test,sta_test,change_test = pro_data(test_data,args)
train_loader = data_loader(trainX, trainY,week_train,sta_train,change_train,args.batch_size, shuffle=True, drop_last=True)
test_loader = data_loader(testX, testY,week_test,sta_test,change_test,args.batch_size, shuffle=True, drop_last=True)


adj = pd.read_csv('../adj/dis.csv', header=None)
adj_mx = np.mat(adj).astype(float)
L_tilde = scaled_Laplacian(adj_mx)
cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L_tilde, args.cheb_k)]


# initial model
net =MSTAN(args.input_dim,args.block1_hidden,args.block2_hidden,device,args.num_nodes,args.lag,args.pre_len,args.cheb_k,args.dropout,args.d_model)
for p in net.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
net = net.to(device)
# Define loss function and Optimization Function
criterion = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)



# Define store file path
result_filename = 'in_'+str(args.lag)+'h_out_'+str(args.pre_len)+'h'
print(result_filename)
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, '../experiments', result_filename)
print(log_dir)
if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)
logger = get_logger(log_dir, name=Model, debug=args.debug)
logger.info('Experiment log path in: {}'.format(args.log_dir))


# Define best model path
best_path = os.path.join(log_dir, 'best_model.pth')
print(best_path)


# Train and val
best_loss = np.inf
start_time = time.time()
for epoch in range(1,args.epochs+1):
    train_outputs = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    test_outputs = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    test_target = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    test_Date = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    test_Sta = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    test_Change = torch.empty([0, args.num_nodes,args.output_dim, args.pre_len]).to(device)
    total_loss = 0

    #Train
    for batch_index, batch_data in enumerate(train_loader):
        train_per_epoch = len(train_loader)
        train_X,train_Y,train_week,train_sta,train_change= batch_data
        train_X = train_X.to(device)
        train_Y = train_Y.to(device)
        optimizer.zero_grad()
        train_output = net(train_X,cheb_polynomials)  #[32,35,12]
        loss = criterion(train_output, train_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_index % args.log_step == 0:
            logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_index,train_per_epoch, loss.item()))
        train_outputs = torch.cat((train_outputs,train_output),0)
    train_epoch_loss = total_loss / train_per_epoch
    logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

    # val
    val_loader = None
    if val_loader == None:
        val_dataloader = test_loader
    else:
        val_dataloader = val_loader
    net.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            val_X, val_Y,val_week,val_sta,val_change= batch_data
            val_X = val_X.to(device)
            val_Y = val_Y.to(device)
            output = net(val_X,cheb_polynomials)
            loss = criterion(output, val_Y)
            if not torch.isnan(loss):
                total_val_loss += loss.item()
    val_epoch_loss = total_val_loss / len(val_dataloader)
    logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_epoch_loss))
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        not_improved_count = 0
        best_state = True
    else:
        not_improved_count += 1
        best_state = False
    if args.early_stop:
        if not_improved_count == args.early_stop_patience:
            print("Validation performance didn\'t improve for {} epochs. "
                  "Training stops.".format(args.early_stop_patience))
            break
    if best_state == True:
        print('*********************************Current best model saved!')
        best_model = copy.deepcopy(net.state_dict())
    adjust_learning_rate(optimizer, epoch, args)
training_time = time.time() - start_time
logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
torch.save(best_model, best_path)
logger.info("Saving current best model to " + best_path)



# Test
net.load_state_dict(torch.load(best_path))
net.eval()
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        test_X, test_Y,test_week,test_sta,test_change= batch_data
        test_X = test_X.to(device)
        test_Y = test_Y.to(device)
        test_week = test_week.to(device)
        test_sta = test_sta.to(device)
        test_change =test_change.to(device)
        test_output = net(test_X,cheb_polynomials)

        test_outputs = torch.cat((test_outputs, test_output), 0)
        test_target = torch.cat((test_target, test_Y), 0)
        test_Date = torch.cat((test_Date,test_week),0)
        test_Sta = torch.cat((test_Sta,test_sta),0)
        test_Change = torch.cat((test_Change, test_change), 0)
    mae, rmse, mape, r2 = All_Metrics(test_outputs, test_target)
    predict_result = torch.cat((test_Sta,test_Date,test_Change,test_outputs,test_target),dim=2)
    np.save('../Prediction_results/{}_{}h.npy'.format(Model,args.pre_len), predict_result.cpu().numpy())
    for t in range(test_target.shape[3]):
        maes1, rmses1, mapes1, r2_s1 = All_Metrics(test_outputs[:, :,:, t],test_target[:, :,:, t])
        print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, R2:{:.4f}".format(t + 1, maes1, rmses1, mapes1, r2_s1))
    print('loss1:{},rmse1:{},mape1:{},r2_1:{}'.format(torch.mean(mae), torch.mean(rmse), torch.mean(mape), torch.mean(r2)))
