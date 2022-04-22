import json
import random
import torch
import torch.optim as optim
import wandb
import numpy as np
import os
import optuna
import joblib
# from model.unet import UNet
from train_tune import Train
from data_loader.dataset_ot import Dataset
import gc


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)
# set network
from seg_model_smp.models_predefined import segmentation_models_pytorch as psmp
net = psmp.Unet( encoder_name="resnet34",        
   encoder_weights=None,    
   in_channels=3,                  
   classes=1,                      
)
net.cuda()

saving_interval = 10
lrs = []

# early stopping patience; how long to wait after last time validation loss improved.
patience = 5
the_last_loss = 100


def main_hyper(trial):
    set_seed(42)
    # seting training and testing dataset

    dsource_loaders = Dataset(config["data_folder"], config["patchsize"], "both")
    dsource_loaders.array_torch()
    source_dataloader = dsource_loaders.source_dataloader
    val_source_dataloader = dsource_loaders.valid_source_dataloader

    #dtarget_loaders = Dataset(config["data_folder"], config["patchsize"], "training_target")
    #dtarget_loaders.array_torch()
    target_dataloader = dsource_loaders.target_dataloader
    val_target_dataloader = dsource_loaders.valid_target_dataloader

    # computing the length
    len_train_source = len(source_dataloader)  # training steps
    len_train_target = len(target_dataloader)
    print(
        f"length of train source:{len_train_source}, lenth of train target is {len_train_target}"
    )

    # computing the length
    len_val_source = len(val_source_dataloader)  # training steps
    len_val_target = len(val_target_dataloader)
    print(
        f"length of validation source:{len_val_source}, lenth of validation target is {len_val_target}"
    )
    #autocast 
    scaler = torch.cuda.amp.GradScaler()
    
    cfg = {
        # # 'n_epochs' : trial.suggest_int('n_epochs',10,50),
        # 'n_epochs' : 2,
        # 'seed' : 0,
        # # 'lr'       : 0.005,
        # # 'momentum' : 0.5,
        # # 'optimizer': optim.SGD,
        # 'n_iter' : trial.suggest_int('n_iter',100,500),
        # 'lr': trial.suggest_loguniform('lr', 1e-3, 1e-1),
        # 'momentum' : trial.suggest_uniform('momentum', 0.4, 0.99),
        # 'optimizer': optim.SGD,
        # # 'optimizer': trial.suggest_categorical('optimizer',[optim.SGD, optim.RMSprop, optim.Adam]),
        # 'weight_decay': 0.0005,
        # 'save_model' : False,
        # 'alpha':trial.suggest_uniform('alpha', 0.01, 0.1),
        # 'lambda_t':trial.suggest_uniform('lambda_t', 0.09, 0.1),
        # 'reg_m':trial.suggest_uniform('reg_m', 0.01, 0.09)
        # 'n_epochs' : trial.suggest_int('n_epochs',10,50),
        "n_epochs":30,
        "seed": 42,
        "lr": trial.suggest_loguniform('lr', 1e-4, 1e-1),
        "momentum": 0.6,
        "optimizer": optim.Adam,
        "save_model": False,
        'alpha':trial.suggest_uniform('alpha', 0.07, 2),
        'lambda_t':trial.suggest_uniform('lambda_t', 0.001, 0.9),
        'reg_m':trial.suggest_uniform('reg_m', 0.01, 0.09)
        
    }

    torch.manual_seed(cfg["seed"])

    parameter_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"The model has {parameter_num:,} trainable parameters")

    model = net
    # optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    # optimizer = cfg["optimizer"](model.parameters(), lr=cfg["lr"], weight_decay=cfg['weight_decay'])
    optimizer=optim.Adam(net.parameters(),lr=cfg["lr"])
    schedule_param = config["lr_param"]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [1, 10, 20], gamma=schedule_param["gamma"]
    )

    for epoch in range(1, cfg["n_epochs"] + 1):
        print(f"in epoch {epoch}")
        print("----------------------Traning phase-----------------------------")
        train_loss,transfer_loss, acc_mat = Train.train_epoch(
            net,
            optimizer,
            source_dataloader,
            target_dataloader,
            cfg["alpha"],
            cfg["lambda_t"],
            cfg["reg_m"],
            scaler            
        )
        f1 = acc_mat[0]
        iou = acc_mat[2]
        loss = train_loss
        
        print(f"Training loss in average for epoch {str(epoch)} is {train_loss}")
        print(f"Training F1 in average for epoch {str(epoch)} is {acc_mat[0]}")
        print(f"Training Accuracy in average for epoch {str(epoch)} is {acc_mat[1]}")
        print(f"Training IOU in average for epoch {str(epoch)} is {acc_mat[2]}")
        del train_loss, acc_mat
        torch.cuda.empty_cache()
        # print("----------------------Evaluation phase-----------------------------")
#         valid_loss, acc_mat = Train.eval_epoch(
#             epoch,
#             net,
#             val_source_dataloader,
#             val_target_dataloader,
#             cfg["alpha"],
#             cfg["lambda_t"],
#             cfg["reg_m"],
#         )
        
#         # print(f"Evaluation loss in average for epoch {str(epoch)} is {valid_loss}")
#         # print(f"Evaluation F1 in average for epoch {str(epoch)} is {acc_mat[0]}")
#         # print(f"Evaluation Accuracy in average for epoch {str(epoch)} is {acc_mat[1]}")
#         # print(f"Evaluation IOU in average for epoch {str(epoch)} is {acc_mat[2]}")
        # del valid_loss, acc_mat

    if cfg["save_model"]:
        torch.save(model.state_dict(), "hyperparam.pt")
    return (1-loss)


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(func=main_hyper, n_trials=6)
    joblib.dump(
        study,
        "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/model/jumbot_optuna.pkl",
    )
