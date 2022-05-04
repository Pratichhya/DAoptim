import gc
import ot
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import wandb
import json
import tifffile as tiff
import torch
from torch import nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
import torchmetrics
import os
from scipy.spatial.distance import cdist

# files
from data_loader.dataset import Dataset
from utils.criterion import DiceLoss
from utils import eval_metrics



# reading config file
with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)

Dice = DiceLoss()
itr = config["num_iterations"]

class Train:
    # def train_epoch(net, optimizer,source_dataloader,target_dataloader, scaler):
    def train_epoch(net, optimizer,source_dataloader,target_dataloader,alpha,lambda_t,reg,reg_m):
        len_train_source = len(source_dataloader)       #training steps
        len_train_target = len(target_dataloader)
        f1_source,acc,IoU,K = 0.0,0.0,0.0,0.0
        f1_tr,acc_tr,IoU_tr,K_tr=0.0,0.0,0.0,0.0
        training_losses = classifier_losses = transfer_losses = target_losses = 0.0
        net.train()
        iter_ = 0

        for i in tqdm(range(itr), total=itr):
            #zero optimizer
            optimizer.zero_grad()

            if i % (len_train_source-1)== 0:
                iter_source = iter(source_dataloader)
            if i % (len_train_target-1) == 0:
                iter_target = iter(target_dataloader)
            xs, ys = iter_source.next()  # source minibatch
            xt,yt = iter_target.next()  # target minibatch
            xs, xt, ys,yt = Variable(xs).cuda(), Variable(xt).cuda(), Variable(ys).cuda(), Variable(yt).cuda()

            # forward
            # with torch.cuda.amp.autocast():
            g_xs, f_g_xs = net(xs)  # source embedded data
            g_xt, f_g_xt = net(xt)  # target embedded data
            del xs, xt
            # segmentation loss
            classifier_loss = Dice(f_g_xs, ys)

            #target loss term on labels
            """loss_target = loss_fn(ys, f_g_xt)"""
            # target_loss = Dice(f_g_xt, ys)
            v_ys = ys.view(ys.size(0),-1)
            v_f_g_xt = f_g_xt.view(f_g_xt.size(0),-1)

            # target_loss = (torch.cdist(v_ys,v_f_g_xt)**2)#/v_ys.size(1)
            target_loss = cdist(v_ys.detach().cpu().numpy(),v_f_g_xt.detach().cpu().numpy(), metric='sqeuclidean')
            target_loss = torch.Tensor(target_loss).cuda()
            target_loss = target_loss/65536

            #transportation cost matrix
            # M_embed = (torch.cdist(g_xs, g_xt) ** 2)#/g_xs.size(1) #Term on embedded data
            M_embed = torch.Tensor(cdist(g_xs.detach().cpu().numpy(),g_xt.detach().cpu().numpy(), metric='sqeuclidean'))
            M_embed = M_embed.cuda()
            M_embed = M_embed/262144
            #computed total ground cost
            M = M_embed*alpha + lambda_t * target_loss

            #OT computation
            a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
            del M_embed
            gamma_ot = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),reg, reg_m=reg_m) 
            gamma = torch.from_numpy(gamma_ot).float().cuda()  # Transport plan
            transfer_loss = torch.sum(gamma * M)

            # total training loss
            total_loss= classifier_loss + transfer_loss
            del gamma,M,gamma_ot#,gamma_emd

            # backward+optimzer
            total_loss.backward()
            optimizer.step()

            #evaluation
            f1_source_step,acc_step,IoU_step,K_step = eval_metrics.f1_score(ys,f_g_xs)
            f1_target, acc_target, IoU_target, K_target = eval_metrics.f1_score(yt, f_g_xt)
            del g_xs, f_g_xs,g_xt, f_g_xt,ys,yt
            #print(acc_step,f1_source_step,IoU_step)
            f1_source+=f1_source_step.detach().cpu().numpy()
            acc+=acc_step.detach().cpu().numpy()
            IoU+=IoU_step.detach().cpu().numpy()
            K+=K_step.detach().cpu().numpy()
            
            f1_tr+=f1_target.detach().cpu().numpy()
            acc_tr+=acc_target.detach().cpu().numpy()
            IoU_tr+=IoU_target.detach().cpu().numpy()
            K_tr+=K_target.detach().cpu().numpy()

            #to calculate average later
            training_losses += total_loss.detach().cpu().numpy()
            classifier_losses += classifier_loss.detach().cpu().numpy()
            transfer_losses += transfer_loss.detach().cpu().numpy()
            target_losses += target_loss.detach().cpu().numpy()

            del f1_source_step,acc_step,IoU_step,K_step , f1_target, acc_target, IoU_target, K_target, classifier_loss,transfer_loss
        return (training_losses/itr),(transfer_losses/itr),[f1_tr/itr,acc_tr/itr,IoU_tr/itr,K_tr/itr]
    
    def val_epoch(net,source_dataloader,target_dataloader,alpha,lambda_t,reg,reg_m):
        len_train_source = len(source_dataloader)       #training steps
        len_train_target = len(target_dataloader)
        f1_source,acc,IoU,K = 0.0,0.0,0.0,0.0
        f1_tr,acc_tr,IoU_tr,K_tr=0.0,0.0,0.0,0.0
        net.train()
        with torch.no_grad():
            # set the model in evaluation mode
            net.eval()

            for i in tqdm(range(itr), total=itr):

                if i % (len_train_source-1)== 0:
                    iter_source = iter(source_dataloader)
                if i % (len_train_target-1) == 0:
                    iter_target = iter(target_dataloader)
                xs, ys = iter_source.next()  # source minibatch
                xt,yt = iter_target.next()  # target minibatch
                xs, xt, ys,yt = Variable(xs).cuda(), Variable(xt).cuda(), Variable(ys).cuda(), Variable(yt).cuda()

                # forward
                g_xs, f_g_xs = net(xs)  # source embedded data
                g_xt, f_g_xt = net(xt)  # target embedded data
                del xs, xt

                #evaluation
                f1_source_step,acc_step,IoU_step,K_step = eval_metrics.f1_score(ys,f_g_xs)
                f1_target, acc_target, IoU_target, K_target = eval_metrics.f1_score(yt, f_g_xt)
                del g_xs, f_g_xs,g_xt, f_g_xt,ys,yt

                f1_source+=f1_source_step.detach().cpu().numpy()
                acc+=acc_step.detach().cpu().numpy()
                IoU+=IoU_step.detach().cpu().numpy()
                K+=K_step.detach().cpu().numpy()

                f1_tr+=f1_target.detach().cpu().numpy()
                acc_tr+=acc_target.detach().cpu().numpy()
                IoU_tr+=IoU_target.detach().cpu().numpy()
                K_tr+=K_target.detach().cpu().numpy()

                del f1_source_step,acc_step,IoU_step,K_step , f1_target, acc_target, IoU_target, K_target
        return [f1_tr/itr,acc_tr/itr,IoU_tr/itr,K_tr/itr]