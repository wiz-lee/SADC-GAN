""" train SADC-GAN model """

import argparse
import os
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader.train_dataset import TrainDataset
from model.common import get_gradient_by_sobel, SAM
from model.fusion_model import SADC_GAN



use_cuda = torch.cuda.is_available()


def init_random_seeds(seed=0):
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


def train_discriminator(train_model, inf_batch, vis_batch, opt_discriminator_inf, opt_discriminator_vis, args):

    fused_image, fake_inf, fake_vis, predicted_value_inf_real, predicted_value_inf_fake, \
            predicted_value_vis_real, predicted_value_vis_fake = train_model(inf_batch, vis_batch)

    # soft-lable of D
    inf_confidence_real = torch.empty_like(predicted_value_inf_real, dtype=torch.float32).uniform_(0.9, 1.0) # b
    vis_confidence_real = torch.empty_like(predicted_value_vis_real, dtype=torch.float32).uniform_(0.9, 1.0) # b

    inf_confidence_fake = torch.empty_like(predicted_value_inf_fake, dtype=torch.float32).uniform_(0.0, 0.1) # c
    vis_confidence_fake = torch.empty_like(predicted_value_vis_fake, dtype=torch.float32).uniform_(0.0, 0.1) # c    

    # LSGAN loss of D
    d_loss_inf_real = torch.mean(torch.square(predicted_value_inf_real - inf_confidence_real))
    d_loss_inf_fake= torch.mean(torch.square(predicted_value_inf_fake - inf_confidence_fake))
    d_loss_inf = d_loss_inf_real + d_loss_inf_fake

    d_loss_vis_real = torch.mean(torch.square(predicted_value_vis_real - vis_confidence_real))
    d_loss_vis_fake= torch.mean(torch.square(predicted_value_vis_fake - vis_confidence_fake))
    d_loss_vis = d_loss_vis_real + d_loss_vis_fake

    d_loss_total = d_loss_inf + d_loss_vis

    opt_discriminator_inf.zero_grad()
    d_loss_inf.backward(retain_graph=True)
    opt_discriminator_inf.step()

    opt_discriminator_vis.zero_grad()
    d_loss_vis.backward(retain_graph=False)
    opt_discriminator_vis.step()

    return d_loss_inf, d_loss_vis, d_loss_total



def train_generator(train_model, inf_batch, vis_batch, opt_generator):

    fused_image, fake_inf, fake_vis, predicted_value_inf_real, predicted_value_inf_fake, \
            predicted_value_vis_real, predicted_value_vis_fake = train_model(inf_batch, vis_batch)

    # soft-lable of D
    inf_confidence_fake = torch.empty_like(predicted_value_inf_fake, dtype=torch.float32).uniform_(0.9, 1.0) # a
    vis_confidence_fake = torch.empty_like(predicted_value_vis_fake, dtype=torch.float32).uniform_(0.9, 1.0) # a

    # *** losses of G ***
    
    # 1. intensity loss
    w_int_inf, w_int_vis = 0.3, 0.7
    g_intensity_loss_inf = F.l1_loss(fused_image, inf_batch)
    g_intensity_loss_vis = F.l1_loss(fused_image, vis_batch)
    g_intensity_loss = w_int_inf * g_intensity_loss_inf + w_int_vis * g_intensity_loss_vis 

    # 2. gradient saliency loss
    gradient_fused = get_gradient_by_sobel(fused_image)
    target_saliency_map, grad_saliency_map_inf, grad_saliency_map_vis, \
                            gradient_inf, gradient_vis = SAM(inf_batch, vis_batch, 0.3)

    g_grad_saliency_loss_inf = torch.mean(grad_saliency_map_inf * torch.abs(gradient_fused - gradient_inf))
    g_grad_saliency_loss_vis = torch.mean(grad_saliency_map_vis * torch.abs(gradient_fused - gradient_vis))
    g_grad_saliency_loss = g_grad_saliency_loss_inf + g_grad_saliency_loss_vis

    # 3. target saliency loss
    g_target_saliency_loss = torch.mean(target_saliency_map * torch.abs(fused_image - inf_batch))

    # 4. LSGAN loss
    g_adv_loss_inf = torch.mean(torch.square(predicted_value_inf_fake - inf_confidence_fake))
    g_adv_loss_vis = torch.mean(torch.square(predicted_value_vis_fake - vis_confidence_fake))
    g_adv_loss = g_adv_loss_inf + g_adv_loss_vis


    # *** decomposition consitency loss of DcNet ***
    int_dc_loss_inf = F.l1_loss(fake_inf, inf_batch)
    int_dc_loss_vis = F.l1_loss(fake_vis, vis_batch)
    int_dc_loss = w_int_inf * int_dc_loss_inf + w_int_vis * int_dc_loss_vis

    grad_dc_loss_inf = F.l1_loss(get_gradient_by_sobel(fake_inf), gradient_inf)
    grad_dc_loss_vis = F.l1_loss(get_gradient_by_sobel(fake_vis), gradient_vis)
    grad_dc_loss = grad_dc_loss_inf + grad_dc_loss_vis


    w_int = 10.
    w_tar = 20.
    w_grad = 25. 
    w_adv = 5.
    w_int_dc = 5.
    w_grad_dc = 5.

    fusion_loss = w_int * g_intensity_loss + w_tar * g_target_saliency_loss + w_grad * g_grad_saliency_loss + w_adv * g_adv_loss
    dc_loss = w_int_dc * int_dc_loss + w_grad_dc * grad_dc_loss
    g_loss_total = fusion_loss + dc_loss                                              
    

    opt_generator.zero_grad()
    g_loss_total.backward()
    opt_generator.step()

    return w_int * g_intensity_loss, w_tar * g_target_saliency_loss, w_grad * g_grad_saliency_loss, w_adv * g_adv_loss, \
            dc_loss, g_loss_total




def main():
    parser = argparse.ArgumentParser(description='Train SADC-GAN by PyTorch.')
    parser.add_argument('--dataset_file', metavar='DIR', default='./train_datasets/MSRS_vis_inf_64.h5',
                        help='path of trainning dataset')                                                                                   
    parser.add_argument('--checkpoint_path', default='trained_models/MSRS')    
    parser.add_argument('--log_path', default='./logs/MSRS', 
                        help='path for logging.')                                                                                      
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')                                       
    parser.add_argument('-b', '--batch_size', default=96, type=int,
                        metavar='N',
                        help='mini-batch size')                 
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')                                                 
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')   
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers')                              
                                                                                           
    args = parser.parse_args()

    init_random_seeds(args.seed)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    train_dataset = TrainDataset(args.dataset_file)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    train_model = SADC_GAN()
    if use_cuda:
        train_model = train_model.cuda()

    optimizer_generator = optim.RMSprop(train_model.generator.parameters(), lr=args.lr)
    optimizer_discriminator_inf = optim.SGD(train_model.discriminator_inf.parameters(), lr=args.lr)
    optimizer_discriminator_vis = optim.SGD(train_model.discriminator_vis.parameters(), lr=args.lr)

    log_writter = SummaryWriter(log_dir=args.log_path)

    # start training
    cur_step = 0
    for epoch in range(args.epochs):
        train_model.train()

        # modify lr during training
        if epoch < args.epochs // 2:
            lr = args.lr
        else:
            lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

        for param_group in optimizer_generator.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_discriminator_inf.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_discriminator_vis.param_groups:
            param_group['lr'] = lr

        # train every mini-batch
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        for cur_batch, (inf_batch, vis_batch) in enumerate(train_tqdm):
            if use_cuda:
                inf_batch = inf_batch.cuda()
                vis_batch = vis_batch.cuda()

            # train D and G in turn
            d_loss_inf, d_loss_vis, d_loss_total = \
                train_discriminator(train_model, inf_batch, vis_batch, optimizer_discriminator_inf, optimizer_discriminator_vis, args)

            g_intensity_loss, g_target_saliency_loss, g_grad_saliency_loss, g_adv_loss, \
            dc_loss, g_loss_total = \
                train_generator(train_model, inf_batch, vis_batch, optimizer_generator)


            train_tqdm.set_postfix(epoch=epoch, int_loss=g_intensity_loss.item(),
                                tar_loss=g_target_saliency_loss.item(),
                                grad_loss=g_grad_saliency_loss.item(),
                                g_adv = g_adv_loss.item(),

                                dc_loss=dc_loss.item(),
                                G_total=g_loss_total.item(),

                                D_total=d_loss_total.item()
                                )
            
            # logging to file
            log_writter.add_scalar('G_total', g_loss_total.item(), cur_step)
            log_writter.add_scalar('G_int', g_intensity_loss.item(), cur_step)
            log_writter.add_scalar('G_target', g_target_saliency_loss.item(), cur_step)
            log_writter.add_scalar('G_grad', g_grad_saliency_loss.item(), cur_step)
            log_writter.add_scalar('G_adv', g_adv_loss.item(), cur_step)
            log_writter.add_scalar('dc_loss', dc_loss.item(), cur_step)
            log_writter.add_scalar('D_total', d_loss_total.item(), cur_step)
            cur_step += 1

        G_state_dict = {'extractor_fuser': train_model.generator.extractor_fuser.state_dict(), 'fr': train_model.generator.fr.state_dict()}
        torch.save(G_state_dict, f'{args.checkpoint_path}/fusion_model_G_epoch_{epoch}.pth')

    log_writter.close()



if __name__ == '__main__':
    ...
    main()
