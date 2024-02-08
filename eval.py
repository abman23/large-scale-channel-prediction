from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled

from skimage import img_as_ubyte

from tqdm import tqdm
import time
import os
import argparse



class RadioMapTestset(Dataset):

    def __init__(self,
                 ind1=0,ind2=83, 
                 input_dir="./RadioMapChallenge_Test/png/",
                 gt_dir="./RadioMapChallenge_Test/gain/",
                 numTx=80,                  
                 antenna = 'height',
                 cityMap="height",
                 transform= transforms.ToTensor()):

        
        self.ind1=ind1
        self.ind2=ind2
        
        self.numTx = numTx 
        
        self.input_dir = input_dir
        self.dir_gain=gt_dir
        
        # set buildings directory
        self.cityMap=cityMap
        if cityMap=="complete":
            self.dir_buildings=os.path.join(self.input_dir, "buildings_complete/")
        elif cityMap=='height':
            self.dir_buildings = os.path.join(self.input_dir, "buildingsWHeight/")
        
        # set antenna directory
        self.antenna=antenna
        if antenna=='complete':
            self.dir_Tx = os.path.join(self.input_dir, "antennas/")
        elif antenna=='height':
            self.dir_Tx = os.path.join(self.input_dir, "antennasWHeight/")
        elif antenna=='building':
            self.dir_Tx = os.path.join(self.input_dir, "antennasBuildings/")

        self.transform= transform
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=idxr+self.ind1
        #names of files that depend only on the map:
        building_name = str(dataset_map_ind)
        Tx_name = str(idxc)
        name1 = building_name + ".png"
        #names of files that depend on the map and the Tx:
        name2 = building_name + "_" + Tx_name + ".png"
        
        #Load buildings:

        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:

        img_name_gain = os.path.join(self.dir_gain, name2)  
        image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255

        # Building complete for post processing
        self.post_buildings=os.path.join(self.input_dir, "buildings_complete/")
        build_comp_name = os.path.join(self.post_buildings, name1)
        build_comp = np.asarray(io.imread(build_comp_name))   

        inputs=np.stack([image_buildings, image_Tx], axis=2) 

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            build_comp = self.transform(build_comp).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!


        return [inputs, image_gain, build_comp, building_name, Tx_name]

def load_model(model_fpath:str, network_type: str ='pmnet'):
    
    # init model 
    if network_type=='pmnet':
        from network.pmnet import PMNet as Model
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,)
    elif network_type=='pmnet_v3':
        from network.pmnet_v3 import PMNet as Model
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,)

    model.cuda()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_fpath))
    model.to(device)

    print(f'Load {model_fpath}.')

    return model

  
def RMSE(pred, target, metrics=None):
  loss = (((pred-target)**2).mean())**0.5
  return loss


def remove_outlier(preds, threshold_low=1/255):
    mask = preds.clone()
    mask[preds < threshold_low] = 0
    
    mask = mask.cuda()

    return mask



def eval_model(model, test_loader):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    for inputs, targets, build_comp, _ , _ in tqdm(test_loader, desc="Evaluating the model.."):
        inputs = inputs.cuda()
        targets = targets.cuda()


        with torch.set_grad_enabled(False):

            preds = model(inputs)  
            preds = torch.clip(preds, 0, 1)

            loss = RMSE(preds, targets) 

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)

    return avg_loss

def inference_all_images(model, test_loader, output_dir: str):
    '''
    :output_dir: directory path where predicted images will be saved.
    '''
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("Error: Failed to create the directory.")

    for inputs, _, _, building_names, Tx_names in tqdm(test_loader, desc="Saving estimated radio map images.."):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):

            preds = model(inputs)
            preds = torch.clip(preds, 0, 1)

            for i in range(len(preds)):
                pred = preds[i]
                pred = pred.reshape((256,256))

                # save predicted image
                io.imsave(os.path.join(output_dir, f'{building_names[i]}_{Tx_names[i]}.png'), img_as_ubyte(pred.cpu()))

    print('All predicted radio maps are saved.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_dir', type=str, help='directory of input images - ~/png/' )
    parser.add_argument('-g', '--gt_dir', type=str, help='directory of ground truth images - ~/gain/')
    parser.add_argument('-f', '--first_index', type=int, help='The first index of map indices')
    parser.add_argument('-l', '--last_index', type=int, help='The last index of map indices')
    parser.add_argument('-m','--pretrained_model', type=str, help='full path of pretrained model.')
    parser.add_argument('-n','--network_type', default='pmnet', type=str, help='Types of network. pmnet, pmnet_v3')
    parser.add_argument('-o','--output_dir', default='./outputs/', type=str, help='directory where predicted images are saved.')

    
    args = parser.parse_args()
    print(args)

    input_dir = args.input_dir
    gt_dir = args.gt_dir
    first_index = args.first_index
    last_index = args.last_index
    pretrained_model = args.pretrained_model
    output_dir = args.output_dir
    network_type = args.network_type

    start = time.time()
    
    # initialize test dataloader
    radio_testset = RadioMapTestset(input_dir=input_dir, gt_dir=gt_dir, ind1=first_index, ind2=last_index)
    test_loader =  DataLoader(radio_testset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False, generator=torch.Generator(device='cuda'))

    # load models
    model = load_model(model_fpath=pretrained_model, network_type=network_type)

    # evaluation
    eval_score = eval_model(model=model, test_loader=test_loader)

    end = time.time()
    
    print(f'RMSE: {eval_score}')
    print(f'Run-time: {end-start:.5f} sec')
    with open('./results.txt', 'w') as f:
        f.write(f'RMSE: {eval_score}\n')
        f.write(f'Run-time: {end-start:.5f} sec\n')
    

    # inference images
    # Uncomment below if you want to save predicted radio maps.
    # inference_all_images(model=model, test_loader=test_loader, output_dir=output_dir)

