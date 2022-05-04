# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:43:02 2021

@author: Tidop
"""

#import PyTorch
import torch
from torch import Tensor, LongTensor

# import PyTorch Geometric
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_geometric.transforms as T
from torch_geometric.nn import DataParallel


# import other libraries
import argparse
import numpy as np 
import platform
import time
import os
import pandas as pd

#import other files
from dataloader import get_dataloader
#from model import Net
from point_transformer_segmentation_main import Net

# Print package versions
print("\n \n Package Versions \n -------------------------------------")
print("Python version :", platform.python_version())
print(f"Numpy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Geometric: {torch_geometric.__version__}")

# parsing command line arguments. Example : python inference.py --input cloud.txt --output result.txt --model model.pt
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--input', type=str, help='path and name of the point cloud to process')
parser.add_argument('--output', type=str, help='where to save the processed point cloud')
parser.add_argument('--features', type=str, help='features index to use separated by comma. Ex : 1,2,4. Default : None', default=None)
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--size', type=int, help='numbers of points processed at the same time. Higher values need more memory. Default=48000', default=48000)
args = parser.parse_args()

feat = args.features.split(',') if args.features is not None else None
output_filename = args.output
file_model = args.model

#load data and create dataloader
data_root, cloud_filename = os.path.split(args.input)
final_loader, _ = get_dataloader(data_root, cloud_filename,  features_index = feat, sample_size = args.size, batch_size=1)



#work on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n lets work on :", device, "\n")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  
#load the architecture from pt file.
architecture = torch.load(file_model)

#Get number of features and number of classes automatically by looking ar architecture dimensions
NUM_FEATURES = architecture[[*architecture.keys()][ 0]].shape[1] #XYZ ne sont pas considérés comme features
NUM_CLASSES  = architecture[[*architecture.keys()][-1]].shape[0]

print("number of features detected : ", NUM_FEATURES)
print("number of classes detected : ", NUM_CLASSES)

#Create the model
model = Net(NUM_CLASSES, NUM_FEATURES, dim_model=[
            32, 64], k=16)#, 128, 256, 512], k=16)
#Load neurons weights from architecture
model = DataParallel(model)
model.load_state_dict(architecture)

model = model.to(device)

transform = T.Compose([
#T.Constant(1)
])


def final(loader):
    """
    

    Parameters
    ----------
    loader : torch_geometric.Data.Dataloader
        data loader of cloud to classify.

    Returns
    -------
    cloud_output : numpy.array, dim N x (2 + C)
        Numpy array, contains id_points, 
    """
    model.eval()
    cloud_output = []
    n = len(loader)
    with torch.no_grad():
        for i,data in enumerate(loader):
            start_time = time.time()
        
            data_tf = transform(data)
            data_tf = data_tf.to(device)
            out = model(data_tf.to_data_list())  # Perform a single forward pass.
            
            scores = out #.argmax(dim=1)  
            data = data.cpu()
            labels = scores.argmax(dim=1).type(torch.LongTensor) # Use the class with highest probability.
            scores = scores.cpu()     
            
            ids = data.ids.reshape((-1,1))
            tupl = [ids, labels.reshape((-1,1)), scores]
            result = np.concatenate(tupl,axis=1) #convert to array
            cloud_output.append( result )   #save sub_data result
            
            end_time = time.time()
            
            if i < 5 or i % max(1,(n // 10)) == 0 :
                print('time per iteration :', end_time - start_time)
                print(f'progress : {i} / {n}')  #print progress
        print("Done")
        
    cloud_output = np.vstack(cloud_output) #convert to array
    
    return cloud_output

result = final(final_loader)
sorted_result = result[ np.argsort( result[:,0])]

arr = pd.read_csv(args.input, sep=" ")

formats = []
for c in arr.columns:
    i = arr[c].first_valid_index()
    liste = str( arr[c][i] ).split('.')
    prec = len( liste[1] ) if len(liste) != 1 else 0
    formats.append('%1.' + str(prec) + 'f')

arr['prediction'] = sorted_result[:,1]
formats.append('%1d')
for i in range(2, sorted_result.shape[1]):
    arr['class_' + str(i-1) + '_score'] = sorted_result[:,i]
    formats.append('%1.3f')

print("saving") 
np.savetxt(output_filename, arr.values, fmt=formats, delimiter=' ',comments='', header=' '.join(arr.columns))
print("done")

'''
#-----------------SMOOTHING----------------
final_result = arr.to_numpy()
try:
    import numpy_indexed as npi
    
    r_smoothing = 0.1 # in meters
    print("smoothing the result on a ", r_smoothing, "m wide neighboorhood")
    edges = radius_graph(Tensor(final_result[:,0:3]), r=0.1, max_num_neighbors=32, loop=True).T
    pred = final_result[:,-1]
    
    def pred_smoothing(cloud, pred,edges):
        """
        
    
        Parameters
        ----------
        cloud : numpy array
            cloud, dim N x (3 + f + 1) 
        pred : LongTensor, dim N x 1 
            Class predicted by the model.
        edges : Tensor, dim E x 2, where E is number of edges
            torch_geometric sparse representation of edges.
    
        Returns
        -------
        output : numpy array
            smoothed cloud, same dimension as input cloud
    
        """
        #assigns to each point the main class of neighboordhood
        smoothed_cloud = []
        smoothed_pred = []
        
        n = pred.shape[0]
        
        #We work here in bulks of 1500000 points, to avoid out of memory errors
        for i in range(1,(n // 1500000) + 2):
            
            #selecting edges of sub data
            sub_edges = edges[ ((i-1) * 1500000 <= edges[:,1]) & (edges[:,1] < i*1500000)]
            #numpy index library allows to group edges by id of the point where they originate of with npi.group_by(). mode() allow to return the main class amongts the neighbors.
            index, main_class = npi.group_by(sub_edges[:,1]).mode(pred[sub_edges[:,0]])
            
            smoothed_cloud.append(cloud[(i-1) * 1500000:i*1500000][index])
            smoothed_pred.append(main_class)
        
        smoothed_cloud = np.vstack(smoothed_cloud)
        smoothed_pred = np.hstack(smoothed_pred)
        
        output = np.concatenate([smoothed_cloud[:,:-1], smoothed_pred.reshape((-1,1))], axis=1)
        
        return output
    
    #smooth result
    smoothed_cloud = pred_smoothing(final_result,pred,edges)
    
    #save
    print("saving")
    np.savetxt(output_filename, smoothed_cloud, fmt='%1.5f')
    print("done")

except:
    print("saving")
    np.savetxt(output_filename, final_result, fmt='%1.5f')
    print("done")
    
'''
