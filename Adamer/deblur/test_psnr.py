## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
# mac 跑不动哈


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import concurrent.futures
from natsort import natsorted
from glob import glob
from model import Adamer # 这个是训练的要的名字
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from pdb import set_trace as stx


def proc(filename):
    tar, prd = filename
    tar_img = utils.load_img(tar)
    prd_img = utils.load_img(prd)
    # PSNR = utils.calculate_psnr(tar_img, prd_img, crop_border=0, test_y_channel=False)
    # SSIM = utils.calculate_ssim(tar_img, prd_img, crop_border=0, test_y_channel=False)
    PSNR = psnr_loss(prd_img, tar_img)
    SSIM = ssim_loss(prd_img, tar_img, multichannel=True)  #multichannel=true 替换成   =2,如果不指定则为灰度图
    # SSIM = ssim_loss(prd_img, tar_img, channel_axis=2)
    return PSNR, SSIM

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='../../datasets/Deblurring', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/option4微调_88000.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/Deblurring_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Adamer.Adamer(**x['network_g'])
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
datasets = ['HIDE']
# datasets = ['RealBlur_J','RealBlur_R']


for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)
    inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()
            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))

            
#################################################       batch 预测              ############################################################               
# batch_size = 2  # 每次预测时将多少张图片打包成一个batch
# for dataset in datasets:
#     result_dir  = os.path.join(args.result_dir, dataset)
#     os.makedirs(result_dir, exist_ok=True)
#     inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
#     files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
#     with torch.no_grad():
        
#          for ids in tqdm(range(0, len(files) // batch_size)):
#             torch.cuda.ipc_collect()
#             torch.cuda.empty_cache()
#             img_list = []
#             for img_path in files[ids * batch_size: (ids + 1) * batch_size]:
#                 assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
#                 img = np.float32(utils.load_img(img_path))/255.
#                 img = torch.from_numpy(img).permute(2,0,1)
#                 input_ = img.unsqueeze(0).cuda()
#                 img_list.append(img)
                
                
            
#             batch_img = torch.stack(img_list, dim=0)
#             restored = model_restoration(batch_img.cuda()) 
#             for idx, im in enumerate(restored):
#                 im = torch.clamp(im,0,1).cpu().detach().permute(1,2,0).numpy()
#                 utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(files[ids * batch_size + idx])[-1])[0]+'.png')), img_as_ubyte(im))
###########################################################################################################################################              
            
            
            
            
            
            
            
                   
for dataset in datasets:
     gt_path = os.path.join(args.input_dir, 'test', dataset, 'target')
     gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')) + glob(os.path.join(gt_path, '*.jpg')))
     assert len(gt_list) != 0, "Target files not found"
     file_path = os.path.join(args.result_dir, dataset)
     path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')) + glob(os.path.join(file_path, '*.jpg')))
     assert len(path_list) != 0, "Predicted files not found"
     psnr, ssim = [], []
     img_files = [(i, j) for i, j in zip(gt_list, path_list)]
     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
         for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
             # psnr.append(PSNR_SSIM)
            
             psnr.append(PSNR_SSIM[0])
             ssim.append(PSNR_SSIM[1])

     avg_psnr = sum(psnr) / len(psnr)
     avg_ssim = sum(ssim) / len(ssim)

print('For {:s} dataset  avg_PSNR: {:f} avg_ssim: {:f}\n'.format(dataset, avg_psnr, avg_ssim))