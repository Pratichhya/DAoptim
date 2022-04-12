import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import json
import tifffile as tiff
import shutil

import torch
from torch.autograd import Variable
import gc
from data_loader.dataset_ot import Dataset
from utils import eval_metrics
from model.unet import UNet
# from seg_model_smp.models_predefined import segmentation_models_pytorch as psmp
# model = psmp.UnetPlusPlus(
#      encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#      encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#      in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#      classes=1,                      # model output channels (number of classes in your dataset)
#  )
print("ok till here 1")
with open(
    "/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/utils/config.json",
    "r",
) as read_file:
    config = json.load(read_file)


model_path = config["model_path"]
prediction_path = ("/share/projects/erasmus/pratichhya_sharma/DAoptim/DAoptim/predictions")


# Load
#device = torch.device("cuda")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = UNet(config["n_channel"], config["n_classes"])
# Choose whatever GPU device number you want
model.load_state_dict(torch.load(config["model_path"] + "es_trig1_weinv3.pt"))
# model.load_state_dict(checkpoint['state_dict'], strict=False)

# Make sure to call input = input.to(device) on any input tensors that you feed to the model
model.to(device)
mode = config["mode"]
def predict():
    avg_acs=[]
    avg_iou=[]
    avg_f1=[]
    avg_k=[]

    dataset = Dataset(config["data_folder"], config["patchsize"], mode["3"])
    dataset.array_torch()
    test_dataloader = dataset.test_dataloader
    with torch.no_grad():
        model.eval()
        for test_idx, (data, target) in enumerate(test_dataloader):
            print(f"--------------Testing on patch number: {test_idx+1}----------------- ")
            data, target = Variable(data.to(device)), Variable(target.to(device))
            _, output = model(data)

            pred = np.rint(output.data.cpu().numpy()[0])
            gt = target.data.cpu().numpy()[0]
            # rgb = data.data.cpu().numpy()[0]
            # tiff.imwrite(os.path.join(prediction_path, f"predicted_{test_idx+1}" + ".tif"), pred)
            # tiff.imwrite(os.path.join(prediction_path, f"rgb_{test_idx+1}" + ".tif"), rgb)
            # tiff.imwrite(os.path.join(prediction_path, f"gt_{test_idx+1}" + ".tif"), gt)
            
            images = wandb.Image(pred, caption="Top: Output, Bottom: Input")
            gt = wandb.Image(gt, caption="Top: Output, Bottom: Input")
            wandb.log({"Ground truth": gt,"Prediction": images})
            #evaluation
            f1_test,acc_test,IoU_test,K_test =eval_metrics.f1_score(target,output)
            avg_acs.append(acc_test)
            avg_iou.append(IoU_test)
            avg_f1.append(f1_test)
            avg_k.append(K_test)
            wandb.log({'test_F1': f1_test,'test_acc':acc_test,'test_IoU':IoU_test,'test_Kappa':K_test})
            gc.collect()
            del (data, target, output)
            print(f"f1: {f1_test}\t IOU; {IoU_test} \t Acc: {acc_test}")
            torch.cuda.empty_cache()
        print(f"Average accuracy is: {sum(avg_acs) / len(avg_acs)}")
        print(f"Average IOU is: {sum(avg_iou) / len(avg_iou)}")
        print(f"Average F1 is: {sum(avg_f1) / len(avg_f1)}")
        print(f"Average kappa is: {sum(avg_k) / len(avg_k)}")
        return


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="server")
    predict()
