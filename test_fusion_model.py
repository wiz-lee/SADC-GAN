""" test SADC-GAN model """

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from data_loader.test_dataset import TestDataset
from model.common import YCbCr2RGB, RGB2YCbCr
from model.fusion_model import GeneratorTestPhase
from train_fusion_model import use_cuda


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SADC-GAN by PyTorch')
    parser.add_argument('--dataset_path', metavar='DIR', default='./test_images/MSRS',     
                        help='path of test dataset.')         
    # change this arg, 'Vis' for MSRS dataset, 'No' for TNO dataset, and 'Inf' for Harvard medical dataset.
    parser.add_argument('--hasRGB', default='Vis',                          
                        help='If have RGB source image, the fused image should be RGB. This arg can be Inf, Vis, or No.')                 
    parser.add_argument('--save_path', default='./fusion_results/MSRS')                                       
    parser.add_argument('--checkpoint', default='./checkpoint/MSRS/fusion_model_G_MSRS.pth',            
                        help='path of trained model.')                                                               
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers.')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
                                                          


    args = parser.parse_args()
    init_seeds(args.seed)

    test_dataset = TestDataset(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    result_path_Gray = os.path.join(args.save_path, 'Gray')
    result_path_RGB = os.path.join(args.save_path, 'RGB')

    if not os.path.exists(result_path_Gray):
        os.makedirs(result_path_Gray, exist_ok=True)
    if not os.path.exists(result_path_RGB):
        os.makedirs(result_path_RGB, exist_ok=True)

    test_model = GeneratorTestPhase()
    if use_cuda:
        test_model = test_model.cuda()
    
    state_dict = torch.load(args.checkpoint)
    test_model.extractor_fuser.load_state_dict(state_dict['extractor_fuser'])
    test_model.fr.load_state_dict(state_dict['fr'])
    test_model.eval()

    test_tqdm = tqdm(test_loader, total=len(test_loader))
    running_times = []
    for cur_batch, (inf_batch, vis_batch, batch_file_names) in enumerate(test_tqdm):
        if use_cuda:
            inf_batch = inf_batch.cuda()
            vis_batch = vis_batch.cuda()

        start_time = time.time()
        fused_image = test_model(inf_batch, vis_batch)
        end_time = time.time()  
        running_times.append(end_time - start_time)

        # save as gray fused image 
        if args.hasRGB == 'No':
            fused_image = transforms.ToPILImage()(fused_image.squeeze())
            fused_image.save(f'{result_path_Gray}/{batch_file_names[0]}')
        # save as rgb fused image
        elif args.hasRGB in ['Inf', 'Vis']:
            folder = 'Inf' if args.hasRGB == 'Inf' else 'Vis'
            rgb = Image.open(os.path.join(args.dataset_path, folder, batch_file_names[0]))
            assert len(rgb.split()) == 3, 'check channels of RGB image'
            
            rgb = transforms.ToTensor()(rgb)
            if use_cuda:
                rgb = rgb.cuda()
            _, cb, cr = RGB2YCbCr(rgb)
            fused_RGB = YCbCr2RGB(fused_image[0], cb, cr)

            fused_image = transforms.ToPILImage()(fused_RGB)
            fused_image.save(f'{result_path_RGB}/{batch_file_names[0]}')
        else:
            raise ValueError(args.hasRGB)
    
    running_mean = np.mean(running_times)
    running_std = np.std(running_times)
    print(f'The average running time is {running_mean} +- {running_std} seconds!')    
        
