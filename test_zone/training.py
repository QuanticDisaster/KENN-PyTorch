# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:37:04 2021

@author: Tidop
"""

#import Pytorch
import torch
from torch import Tensor, LongTensor

# import PyTorch Geometric
import torch_geometric as torchgm
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.utils import accuracy, precision, f1_score, recall
import torch_geometric.transforms as T
from torch_geometric.nn import DataParallel

# import other libraries
import time
import pickle
import argparse
import numpy as np 
import os
import copy
import platform

# libraries only for visualization at the end
import pandas
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#import from other files
from dataloader import get_dataloader
#from model import Net
#from model_PointTransformer import Net
from point_transformer_segmentation_main import Net


# Print package versions
print("\n \n Package Versions \n -------------------------------------")
print("Python version :", platform.python_version())
print(f"Numpy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Geometric: {torchgm.__version__}")


# parsing command line arguments. Example : python training.py --train train.txt --test test.txt --labels 6 --output output_directory
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', type=str, help='training point cloud')
parser.add_argument('--test', type=str, help='test point cloud')
parser.add_argument('--features', type=str, help='features index to use separated by comma, starting from 0 included. Ex : 1,2,4. Default : None', default=None)
parser.add_argument('--labels', type=int, help='index of the column with labels', default=None)
parser.add_argument('--output', type=str, help='directory where to save the result', default='output')
parser.add_argument('--size', type=int, help='numbers of points processed at the same time. Higher values need more memory. Default=48000', default=48000)
parser.add_argument('--epoch', type=int, help='numbers of epochs (learning rate is divided by 10 regularly). Default=100', default=100)
args = parser.parse_args()

#split features list from '3,4,5' to ['3','4','5']
feat = args.features.split(',') if args.features is not None else None
file_model = args.output
labs = args.labels
feat = feat
feat_train = feat
batch_size = int(args.size / 1500)


#load training and test data thanks to the get_dataloader function
print('found train : ', os.path.isfile(args.train))
if os.path.isfile(args.train):
    data_root, train_filename = os.path.split(args.train)
else:
    data_root, train_filename = args.train, None
train_loader, weights = get_dataloader(data_root, train_filename, label_index = labs, features_index = feat_train, sample_size = 1500, batch_size=batch_size, log=True)
print('found test : ', os.path.isfile(args.test))
if os.path.isfile(args.test):
    data_root, test_filename = os.path.split(args.test)
else:
    data_root, test_filename = args.test, None
test_loader, _        = get_dataloader(data_root, test_filename,  label_index = labs, features_index = feat, sample_size = 1500, batch_size=batch_size)



NUM_FEATURES = len(feat) if feat is not None else 0
NUM_CLASSES = len(weights)

#Try to work on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("let's work on :", device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if device.type == 'cpu':
    print("no GPU was detected. Ensure CUDA drivers are properly installed")
    print("\n \n########### WARNING : You are currently working on CPU which is up to 10x slower than GPU ! It is strongly advised to work on GPU if possible !\n \n ")

#define data_augmentation and general transformations
data_augmentation = T.Compose([
    T.RandomRotate(360, axis=2),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=0),
    T.RandomFlip(0),
    T.RandomFlip(1)
    #T.RandomTranslate([0.1, 0.1, 0.03])
])

transform = T.Compose([
])

def train():
    epoch(train_loader,True)

def test():
    with torch.no_grad():
        epoch(test_loader, False)

def epoch(loader,is_training):
    """
    

    Parameters
    ----------
    loader : torch_geometric.Data.dataloader
        data loader.
    is_training : bool
        boolean indicating if it is a training epoch.

    Returns
    -------
    None.

    """
    
    model.train(mode=is_training)  #switch to model.eval() or model.train(). mode=False mean model is in test mode
    
    #store predictions and corresponding ground truth to compute metrics later
    all_pred = []
    all_labels = []
    epoch_loss = 0
    
    for k, data in enumerate(loader):

        #In case of training mode, augment data and reset optimizer gradients
        if is_training :
            optimizer.zero_grad()
            data = data_augmentation(data)
        data = transform(data)
        data = data.to(device)
        out = model(data.to_data_list()) # due to the use of DataParallel to use multiple GPUs, we need to pass the batch as a list of samples instead of directly giving the batch
        y_mask = data.y != -1  # unclassified points are labelled as -1
        loss = criterion(out[y_mask], data.y[y_mask])
        
        out, data = out.cpu(), data.cpu()
        
        #if training, update gradients
        if is_training:
            loss.backward()
            optimizer.step()
            
        #store predictions and corresponding ground truth to compute metrics later
        epoch_loss += loss.item()
        pred = out.argmax(dim=1)
        all_pred += list(pred[y_mask].numpy())
        all_labels += list(data.y[y_mask].numpy())
    
        
    test_or_train = 'train' if is_training else 'test'
    
    #track loss
    metrics[test_or_train]['loss'].append(epoch_loss)
    
    #compute per class metrics : IoUs
    all_pred = LongTensor(all_pred)
    all_labels = LongTensor(all_labels)    
    i, u = i_and_u(all_pred, all_labels, NUM_CLASSES)
    iou = torch.true_divide(i, u)
    iou[torch.isnan(iou)] = -1
    metrics[test_or_train]["IoUs"].append(iou.numpy())
    
        
    #per_class other metrics : accuracy
    metrics[test_or_train]["accuracy"].append(accuracy(all_pred, all_labels))
    
    
    

#define a list of learning rate
lr = [np.round(0.1 ** i,8) for i in range(3,4)]

#store models and metrics
models = []
models_metrics = []

for i,l in enumerate(lr):
    
    best_model = None
    best_model_metric = -1
    best_model_index = None
    
    #load the architecture from pt file.
    architecture = torch.load('PTransformer_seg.pt')
    model = Net(50, 3, dim_model=[
            32, 64], k=16)
    #Load neurons weights from architecture
    model.load_state_dict(architecture, strict=False)
    
    #cut first and last layer
    from torch_geometric.nn import Linear as Lin
    model.mlp_input[0] = Lin(max(1,NUM_FEATURES),32)
    model.lin[4] = Lin(64, NUM_CLASSES)
    
    
    model = DataParallel(model)
    model = model.to(device)
    
    print(" trainable parameters :",sum(p.numel() for p in model.parameters() if p.requires_grad) )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=l)#,weight_decay=0.95)
    criterion = torch.nn.CrossEntropyLoss(weight=Tensor(weights).to(device))    #use weights to take class imbalance into account
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch //  4,  2 * args.epoch // 4, 3 * args.epoch // 4], gamma=0.1) #reduce learning rate by factor 0.1 at 20th epoch

    metrics =  {"train" : {"accuracy" : [], "IoUs": [], 'loss' : []},
                "test" : {"accuracy" : [], "IoUs": [], 'loss' : []}}
    train_loss_history = []
    test_loss_history = []
    

    print("-------------------- initial learning rate : " + str(l) + "-------------------------------")
    for epo in range(1, args.epoch):
        #Each iteration, do a training and a test on whole data
        start_time = time.time()
        train()
        test()
        end_time = time.time()
        scheduler.step()
        print('Epoch: {:02d} / {:02d}, Time: {:.2f}s, mIoU train : {}, mIoU test : {}, IoU: {}'.format(epo, args.epoch,(end_time-start_time), metrics["train"]["IoUs"][-1].mean(), metrics["test"]["IoUs"][-1].mean(), metrics["test"]["IoUs"][-1]))
        
        #if IoU of last model is better, save model and the metric
        if metrics['test']["IoUs"][-1].mean() > best_model_metric:
            best_model = copy.deepcopy(model)
            best_model_metric = metrics['test']["IoUs"][-1].mean()
            best_model_index = epo
    
    print("\nSelecting best model with mIoU: ", best_model_metric, " at epoch: ", best_model_index, '\n\n')
    models.append(best_model)
    models_metrics.append(metrics)
    
    
    
    


#save model and other things
dir_path = args.output
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for i in range(len(models)):  
    torch.save(models[i].state_dict(), os.path.join(dir_path, "model" + str(i) + ".pt"))

with open(os.path.join(dir_path, 'models_metrics.pickle'), 'wb') as handle:
    pickle.dump(models_metrics, handle)

#for unknown reasons, causes error when trying to iterate trough the loader once it has been saved
#torch.save(test_loader, os.path.join(dir_path, 'test_loader.pth'))


#-----------------LOSS and IOU curves----------------
matplotlib.use('Agg')

models_train_loss = []
models_test_loss = []
for mod in models_metrics:
    models_train_loss.append(mod['train']['loss'])
    models_test_loss.append(mod['test']['loss'])

display_loss = np.array(models_train_loss), np.array(models_test_loss)
print(len(models_train_loss))
#plt.rcParams['figure.dpi'] = 150
n = max(2,len(models_train_loss))
fig, axs = plt.subplots(n, 2,figsize=(8, 5*n), sharex=True)#, sharey=True)
#fig.tight_layout(pad=3.0)

axs[0, 0].set_title('loss')
for i in range(len(models_train_loss)):
    axs[i, 0].plot(np.log(display_loss[0][i]), label="test")  #training loss
    axs[i, 0].plot(np.log(display_loss[1][i]))   #test loss
    axs[i, 0].grid(True)
    axs[i, 1].plot(np.mean(models_metrics[i]["train"]["IoUs"], axis=1))#, 'b',  alpha=1 - 0.15 * i)
    axs[i, 1].plot(np.mean(models_metrics[i]["test"]["IoUs"], axis=1))#, 'g', alpha=1 - 0.15 * i)
    axs[i, 1].locator_params(axis="y", nbins=14)
    axs[i, 1].grid(True)
    
plt.legend()
axs[0, 1].set_title('mean IoU')
plt.savefig(os.path.join(dir_path, 'loss_and_IoU_curves.jpg'))




#-----------------classification reports----------------*

precisions = []
recalls = []
f_score = []
IoUs = []
accuracies = []

for i,model in enumerate(models):
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = transform(data)
            data = data.to(device)
            out = model(data.to_data_list())
            out, data = out.cpu(), data.cpu()
            pred = out.argmax(dim=1)

            all_preds += np.array(pred).tolist()
            all_labels += np.array(data.y).tolist()

    all_preds = LongTensor(all_preds)
    all_labels = LongTensor(all_labels)   
    
    precisions.append( precision(all_preds, all_labels, NUM_CLASSES).mean().numpy())
    recalls.append( recall(all_preds, all_labels, NUM_CLASSES).mean().numpy())
    f_score.append( f1_score(all_preds, all_labels, NUM_CLASSES).mean().numpy() )
    
    IoUs.append(models_metrics[i]['test']['IoUs'][best_model_index-1].mean())
    accuracies.append(models_metrics[i]['test']['accuracy'][best_model_index-1])
    
    results = classification_report(all_labels, all_preds)
    print(results)

stats = np.vstack([precisions, recalls, f_score, accuracies, IoUs])

row_labels = ['model loss', 'average class precision', 'average class recall', 'average class accuracy', "average class IoU"]
column_labels = ["model" + str(i) for i in range(len(models))]

df = pandas.DataFrame(stats, columns=column_labels, index=row_labels)
print(df)
    
#-----------------per class accuracy figures----------------*

for i in range(len(models_metrics)):
    plt.figure()
    for k in range(NUM_CLASSES):
        plt.plot( np.array(models_metrics[i]["test"]["IoUs"])[:,k], label="classe " +str(k))
    plt.legend()
plt.savefig(os.path.join(dir_path, 'per_class_IoU_record.jpg'))



#-----------------superposed loss and accuracies image of the different models----------------*


fig, axs = plt.subplots(1, 2,figsize=(16, 10), sharex=True)#, sharey=True)

font = {'family' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

axs[ 0].set_title('loss')
for i in range(len(models_metrics)):
    axs[ 0].plot(np.log(display_loss[0][i]), label="test")  #training loss
    axs[ 0].plot(np.log(display_loss[1][i]))   #test loss
    axs[ 0].grid(True)
    axs[ 1].plot(np.mean(models_metrics[i]["train"]["IoUs"], axis=1))#, 'b',  alpha=1 - 0.15 * i)
    axs[ 1].plot(np.mean(models_metrics[i]["test"]["IoUs"], axis=1))#, 'g', alpha=1 - 0.15 * i)
    axs[ 1].locator_params(axis="y", nbins=14)
    axs[ 1].grid(True)
    
plt.legend()

axs[ 1].set_title('mean IoU')
plt.savefig(os.path.join(dir_path, 'superposed_loss_and_iou.jpg'))