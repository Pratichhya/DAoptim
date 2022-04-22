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
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
import torchmetrics
import os
from scipy.spatial import distance

# files
from data_loader.dataset_ot import Dataset
from utils.criterion import DiceLoss,FocalTverskyLoss
from utils import eval_metrics
from scipy.spatial.distance import cdist

# reading config file
with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)

# Dice = FocalTverskyLoss()
Dice = DiceLoss()


class Train:
    def train_epoch(net, optimizer,source_dataloader,target_dataloader, scaler):
     # def train_epoch(net, optimizer,source_dataloader,target_dataloader,alpha,lambda_t,reg_m,itr):
        len_train_source = len(source_dataloader)       #training steps
        len_train_target = len(target_dataloader)
        f1_source,acc,IoU,K = 0.0,0.0,0.0,0.0
        f1_tr,acc_tr,IoU_tr,K_tr=0.0,0.0,0.0,0.0
        training_losses = classifier_losses = transfer_losses = target_losses = 0.0
        alpha = config["alpha"]
        lambda_t = config["lambda_t"]
        reg_m = config["reg_m"]
        net.train()
        iter_ = 0

        for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
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
            with torch.cuda.amp.autocast():
                g_xs, f_g_xs = net(xs)  # source embedded data
                g_xt, f_g_xt = net(xt)  # target embedded data
                del xs, xt
                # segmentation loss
                classifier_loss = Dice(f_g_xs, ys)

                #target loss term on labels
                """loss_target = loss_fn(ys, f_g_xt)"""
                # target_loss = Dice(f_g_xt, ys)
                # v_ys = ys.view(ys.size(0),-1)
                # v_f_g_xt = f_g_xt.view(f_g_xt.size(0),-1)
                # # print(f"v_ys{v_ys.size(1)}")
                # # print(f"v_f_g_xt{v_f_g_xt.shape}")
                # # print(f"f_g_xt{f_g_xt.shape}")
                # # print(f"ys{ys.shape}")
                # # print(f"g_xs{g_xs.size(1)}")
                # # print(f"g_xt{g_xt.shape}")

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
                # gamma_emd = ot.emd(a, b, M.detach().cpu().numpy())
                # gamma_ot = ot.sinkhorn(a, b, M.detach().cpu().numpy(), reg_m)
                gamma_ot = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),0.01, reg_m=reg_m) 
                gamma = torch.from_numpy(gamma_ot).float().cuda()  # Transport plan
                transfer_loss = torch.sum(gamma * M)

                # print(f"transfer_loss:{transfer_loss}")

                # total training loss
                total_loss= classifier_loss + transfer_loss
                del gamma,M,gamma_ot#,gamma_emd

            # backward+optimzer
            # total_loss.backward()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()

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
            
            # torch.cuda.empty_cache()
            # wandb.log({'train_Loss': total_loss,'train_F1': f1_source_step,'train_acc':acc_step,'train_IoU':IoU_step,'f1_target': f1_target,'acc_target':acc_target,'IoU_target':IoU_target,'classifier_loss':classifier_loss,'transfer_loss':transfer_loss,'target_loss':target_loss})
            del f1_source_step,acc_step,IoU_step,K_step , f1_target, acc_target, IoU_target, K_target, classifier_loss,transfer_loss
        return (training_losses/config["num_iterations"]),(transfer_losses/config["num_iterations"]),[f1_tr/config["num_iterations"],acc_tr/config["num_iterations"],IoU_tr/config["num_iterations"],K_tr/config["num_iterations"]]
    
    def eval_epoch(e, net, val_source_dataloader, val_target_dataloader,scaler):
     # def eval_epoch(e, net, val_source_dataloader, val_target_dataloader,alpha,lambda_t,reg_m,itr):
        len_val_source = len(val_source_dataloader)  # training steps
        len_val_target = len(val_target_dataloader)
        f1_source, acc, IoU, K = 0.0, 0.0, 0.0, 0.0
        f1_t, acc_t, IoU_t, K_t = 0.0, 0.0, 0.0, 0.0
        training_losses = classifier_losses = transfer_losses = target_losses = 0.0
        alpha = config["alpha"]
        lambda_t = config["lambda_t"]
        reg_m = config["reg_m"]
        # num_iteration = itr
        # for validation set
        with torch.no_grad():
            # set the model in evaluation mode
            net.eval()
            # for i in tqdm(range(len_val_target), total=len_val_target):
            for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
                if i % (len_val_source - 1) == 0:
                    v_iter_source = iter(val_source_dataloader)
                if i % (len_val_target - 1) == 0:
                    v_iter_target = iter(val_target_dataloader)
                val_xs, val_ys = v_iter_source.next()  # source minibatch
                val_xt, val_yt = v_iter_target.next()  # target minibatch
                val_xs, val_xt, val_ys,val_yt = (
                    Variable(val_xs).cuda(),
                    Variable(val_xt).cuda(),
                    Variable(val_ys).cuda(),
                    Variable(val_yt).cuda()
                )
                with torch.cuda.amp.autocast():
                    # forward
                    val_g_xs, val_f_g_xs = net(val_xs)  # source embedded data
                    val_g_xt, val_f_g_xt = net(val_xt)  # target embedded data
                    # pred_xt = torch.argmax(f_g_xt, dim=0)
                    del val_xt, val_xs
                    # segmentation loss
                    eval_classifier_loss = Dice(val_f_g_xs, val_ys)
                    # print(f"classifier loss is:{classifier_loss}")

                    # target loss term on labels
                    """loss_target = loss_fn(ys, f_g_xt)"""
                    # eval_target_loss = Dice(val_f_g_xt, val_ys)
                    # print(f"target segmentation loss is:{target_loss}")
                    eval_val_ys = val_ys.view(val_ys.size(0),-1)
                    eval_val_f_g_xt = val_f_g_xt.view(val_f_g_xt.size(0),-1)

                    eval_target_loss = cdist(eval_val_ys.detach().cpu().numpy(),eval_val_f_g_xt.detach().cpu().numpy(), metric='sqeuclidean')
                    eval_target_loss = torch.Tensor(eval_target_loss).cuda()
                    eval_target_loss = eval_target_loss/65536

                    # transportation cost matrix
                    # eval_M_embed = (torch.cdist(val_g_xs, val_g_xt) ** 2)  
                    eval_M_embed = torch.Tensor(cdist(val_g_xs.detach().cpu().numpy(),val_g_xt.detach().cpu().numpy(), metric='sqeuclidean'))
                    eval_M_embed = eval_M_embed.cuda()
                    eval_M_embed = eval_M_embed/262144
                    # Term on embedded data
                    # print(f"g_xs{g_xs.size()}, g_xt {g_xt.size()}")

                    # computed total ground cost ()
                    eval_M = eval_M_embed * alpha + lambda_t * eval_target_loss

                    # OT computation
                    val_a, val_b = ot.unif(val_g_xs.size()[0]), ot.unif(val_g_xt.size()[0])
                    del eval_M_embed
                    # val_gamma_emd = ot.emd(val_a, val_b, eval_M.detach().cpu().numpy())
                    # val_gamma_ot = ot.sinkhorn(val_a, val_b, eval_M.detach().cpu().numpy(), reg_m )
                    val_gamma_ot = ot.unbalanced.sinkhorn_knopp_unbalanced(val_a, val_b, eval_M.detach().cpu().numpy(),0.01, reg_m=reg_m)
                    val_gamma = (torch.from_numpy(val_gamma_ot).float().cuda()
                    )  
                    # Transport plan
                    eval_transfer_loss = torch.sum(val_gamma * eval_M)
                    eval_total_loss = eval_classifier_loss + eval_transfer_loss
                    # eval_total_loss = eval_transfer_loss
                    # print(f"validation transfer_loss:{eval_transfer_loss}")
                    del val_gamma,eval_M,val_gamma_ot#,gamma_emd
                # evaluation

                f1_source_step,acc_step,IoU_step,K_step = eval_metrics.f1_score(val_ys,val_f_g_xs)
                val_f1_target, val_acc_target, val_IoU_target, val_K_target = eval_metrics.f1_score(val_yt, val_f_g_xt)
                del val_ys, val_f_g_xs
                #print(acc_step,f1_source_step,IoU_step)
                f1_source+=f1_source_step.detach().cpu().numpy()
                acc+=acc_step.detach().cpu().numpy()
                IoU+=IoU_step.detach().cpu().numpy()
                K+=K_step.detach().cpu().numpy()

                f1_t+=val_f1_target.detach().cpu().numpy()
                acc_t+=val_acc_target.detach().cpu().numpy()
                IoU_t+=val_IoU_target.detach().cpu().numpy()
                K_t+=val_K_target.detach().cpu().numpy()

                #to calculate average later
                training_losses += eval_total_loss.detach().cpu().numpy()
                classifier_losses += eval_classifier_loss.detach().cpu().numpy()
                transfer_losses += eval_transfer_loss.detach().cpu().numpy()
                target_losses += eval_target_loss.detach().cpu().numpy()
           
                if e % 10 == 0:
                    # rgb = val_xt.data.cpu().numpy()[0]
                    pred = np.rint(val_f_g_xt.data.cpu().numpy()[0])
                    gt = val_yt.data.cpu().numpy()[0]
                    # tiff.imwrite(
                    #    os.path.join(config["eval_output"], f"rgb_val{i+1}" + ".tif"),
                    #    rgb,
                    # )
                    # tiff.imwrite(
                    #    os.path.join(config["eval_output"], f"pred_val{i+1}" + ".tif"),
                    #    pred,
                    # # )
                    # images_pred = wandb.Image(pred, caption="Top: Output, Bottom: Input")
                    # # images_rgb = wandb.Image(rgb, caption="Top: Output, Bottom: Input")
                    # images_gt = wandb.Image(gt, caption="Top: Output, Bottom: Input")
                    # wandb.log({"Ground truth": images_gt,"Prediction": images_pred})
                    # torch.cuda.empty_cache()

                # wandb.log({'val_train_Loss': eval_total_loss,'val_train_F1': f1_source_step,'val_train_acc':acc_step,'val_train_IoU':IoU_step,'val_f1_target': val_f1_target,'val_acc_target':val_acc_target,'val_IoU_target':val_IoU_target,'val_classifier_loss':eval_classifier_loss,'val_transfer_loss':eval_transfer_loss,'val_target_loss':eval_target_loss})
        
                del f1_source_step,acc_step,IoU_step,K_step , eval_classifier_loss,eval_transfer_loss,eval_target_loss, val_f1_target, val_acc_target,val_IoU_target,val_K_target
        return (training_losses/config["num_iterations"]),(transfer_losses/config["num_iterations"]),[f1_t/config["num_iterations"],acc_t/config["num_iterations"],IoU_t/config["num_iterations"],K_t/config["num_iterations"]]

