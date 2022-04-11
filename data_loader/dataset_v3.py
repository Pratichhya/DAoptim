# -*- coding: utf-8 -*-
# importing necessary packages
import os
import rasterio
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import shutil

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

#files
from .test_size import Shape
from .patching import Patch
from .augmentation import Augumentation
from .one_hot import OneHotEncoding



class Dataset():
    def __init__(self, data_folder, patchsize, for_what):
        # connecting to the folder
        print("Buckle up, here with start the journey")
        self.data_folder = data_folder
        self.train_folder = self.data_folder + "train_data"
        self.test_folder = self.data_folder + "test_data"
        self.patchsize = patchsize
        self.for_what = for_what


    # fetching training and testing folder
    def get_data(self):
        self.xs_train_img_files = []
        self.ys_train_label_files = []
        self.xt_train_img_files = []
        self.xt_train_label_files = []
        self.test_img_files = []
        self.test_label_files = []
        # collecting training images files
        if self.for_what == "training_source":
            for image in os.listdir(self.train_folder):
                if image == "source":
                    xs_train_img_folder = os.path.join(self.train_folder, image)
                    for img in os.listdir(xs_train_img_folder):
                        xs_train_img_path = os.path.join(xs_train_img_folder, img)
                        self.xs_train_img_files.append(xs_train_img_path)
                        # print(f"training images:{self.train_img_files}")
                elif image == "target":
                    xt_train_img_folder = os.path.join(self.train_folder, image)
                    for img in os.listdir(xt_train_img_folder):
                        xt_train_img_path = os.path.join(xt_train_img_folder, img)
                        self.xt_train_img_files.append(xt_train_img_path)
                        # print(f"training images:{self.train_img_files}")
                elif image == "source_labels":
                    ys_train_labels_folder = os.path.join(self.train_folder, image)
                    for lab in os.listdir(ys_train_labels_folder):
                        ys_train_label_path = os.path.join(ys_train_labels_folder, lab)
                        self.ys_train_label_files.append(ys_train_label_path)
                        # print(f"training labels:{self.train_label_files}")
        if self.for_what == "training_target":
            for image in os.listdir(self.train_folder):
                if image == "target":
                    xt_train_img_folder = os.path.join(self.train_folder, image)
                    for img in os.listdir(xt_train_img_folder):
                        xt_train_img_path = os.path.join(xt_train_img_folder, img)
                        self.xt_train_img_files.append(xt_train_img_path)
                        # print(f"training images:{self.train_img_files}")
                elif image == "target_labels":
                    yt_train_labels_folder = os.path.join(self.train_folder, image)
                    for lab in os.listdir(yt_train_labels_folder):
                        yt_train_label_path = os.path.join(yt_train_labels_folder, lab)
                        self.xt_train_label_files.append(yt_train_label_path)
                        # print(f"training labels:{self.train_label_files}")        
        elif self.for_what == "testing":
            for image in os.listdir(self.test_folder):
                if image == "images":
                    test_img_folder = os.path.join(self.test_folder, image)
                    for img in os.listdir(test_img_folder):
                        test_img_path = os.path.join(test_img_folder, img)
                        self.test_img_files.append(test_img_path)
                        # print(f"testing images:{self.test_img_files}")
                elif image == "labels":
                    test_labels_folder = os.path.join(self.test_folder, image)
                    for lab in os.listdir(test_labels_folder):
                        test_label_path = os.path.join(test_labels_folder, lab)
                        self.test_label_files.append(test_label_path)
                        # print(f"testing labels:{self.test_label_files}")

    # reading images
    def read_images(self, image_list):
        # append all the images as numpy array after reading
        for val in image_list:
            read_img = rasterio.open(val).read()
            return read_img[:3,:,:]

    # reading labels
    def read_labels(self, label_list):
        # append all the labels as numpy after reading
        for val in label_list:
            read_labels = rasterio.open(val).read()
            return read_labels

    # loading training and testing dataset
    def load_data(self):
        print("Ok just wait here, I will go search for data")
        print("wow, you go girl, you found all the dataset required for the trip")
        self.get_data()
        if self.for_what == "training_source":
            self.Xs_train_main = self.read_images(self.xs_train_img_files)
            self.Ys_train_main = self.read_labels(self.ys_train_label_files)
        elif self.for_what == "training_target":
            self.Xt_train_main = self.read_images(self.xt_train_img_files)
            #self.Yt_train_main = self.read_labels(self.yt_train_label_files)
        elif self.for_what == "testing":
            self.Xtest_main = self.read_images(self.test_img_files)
            self.Ytest_main = self.read_labels(self.test_label_files)

    def save_images(self,path,array,name):
        for i in range(array.shape[0]):
            #Create the output files with each unique file name
            tiff.imwrite(os.path.join(path, f"{name}_{i+1}"+".tif"), array[i,:,:,:])
        print("Congratulations!!! 🎉🎉🎉  for saving all your augmented patches")

    # save the created patches in a different folder
    def create_directory(self):
        self.load_data()
        # self.save_images()
        if self.for_what == "training_source":
            # training images creation
            if os.path.exists(self.train_folder + "/source_img_patches/"):
                print("I will read source image patches from existing directory")
                self.xs_train_patches = []
                for i in os.listdir(self.train_folder + "/source_img_patches/"):
                    im = rasterio.open(self.train_folder + "/source_img_patches/"+i).read()
                    self.xs_train_patches.append(im)
                self.Xs_train = np.asarray(self.xs_train_patches)
                print("read; shape is:", self.Xs_train.shape)
                # shutil.rmtree(self.train_folder + "/source_img_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/source_img_patches")
            else:
                os.makedirs(self.train_folder + "/source_img_patches/")
                print("working on images...grab some food until then")
                self.xs_train_patches = Patch.image_patching(self,self.Xs_train_main)
                self.Xs_train = Augumentation.augumentation(self,self.xs_train_patches)
                xs_file_path = self.train_folder + "/source_img_patches/"
                self.save_images(xs_file_path,self.Xs_train,"xs_train")

            # training labels creation
            if os.path.exists(self.train_folder + "/source_label_patches/"):
                print("I will read source labels from existing directory")
                self.ys_train_patches = []
                for i in os.listdir(self.train_folder + "/source_label_patches/"):
                    im = rasterio.open(self.train_folder + "/source_label_patches/"+i).read()
                    self.ys_train_patches.append(im)
                self.Ys_train = np.asarray(self.ys_train_patches)
                print("read; shape is:", self.Ys_train.shape)
                # shutil.rmtree(self.train_folder + "/source_label_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/source_label_patches")
            else:
                os.makedirs(self.train_folder + "/source_label_patches/")
                print("It's still gonna take a while...have some patience")
                self.ys_train_patches = Patch.image_patching(self,self.Ys_train_main)
                self.Ys_train =Augumentation.augumentation(self,self.ys_train_patches)
                ys_file_path = self.train_folder + "/source_label_patches/"
                self.save_images(ys_file_path,self.Ys_train,"xs_train")
            
        elif self.for_what == "training_target":
            # training images creation
            if os.path.exists(self.train_folder + "/traget_img_patches/"):
                print("I will read target image patches from existing directory")
                self.xt_train_patches = []
                for i in os.listdir(self.train_folder + "/traget_img_patches/"):
                    im = rasterio.open(self.train_folder + "/traget_img_patches/"+i).read()
                    self.xt_train_patches.append(im)
                self.Xt_train = np.asarray(self.xt_train_patches)
                print("read; shape is:", self.Xt_train.shape)
                # shutil.rmtree(self.train_folder + "/traget_img_patches")  # Removes all the subdirectories!
                # os.makedirs(self.train_folder + "/traget_img_patches")
            else:
                os.makedirs(self.train_folder + "/traget_img_patches/")
                print("working on images...grab some food until then")
                self.xt_train_patches = Patch.image_patching(self,self.Xt_train_main)
                self.Xt_train = Augumentation.augumentation(self,self.xt_train_patches)
                xt_file_path = self.train_folder + "/traget_img_patches/"
                self.save_images(xt_file_path,self.Xt_train,"xt_train")

        elif self.for_what == "testing":
            # testing image creation
            if os.path.exists(self.test_folder + "/testing_images/"):
                print("I will read testing dataset from existing directory")
                self.xtest_patches = []
                for i in os.listdir(self.test_folder + "/testing_images/"):
                    im = rasterio.open(self.test_folder + "/testing_images/"+i).read()
                    self.xtest_patches.append(im)
                self.xtest_patches = np.asarray(self.xtest_patches)
                print("read; shape is:", self.xtest_patches.shape)
                # shutil.rmtree(self.test_folder + "/testing_images")  # Removes all the subdirectories!
                # os.makedirs(self.test_folder + "/testing_images")
            else:
                os.makedirs(self.test_folder + "/testing_images/")
                print("working on images...grab some food until then")
                self.xtest_patches = Patch.image_patching(self,self.Xtest_main)
                # self.Xtest = Augumentation.augumentation(self,self.xtest_patches)
                x_file_path = self.test_folder +"/testing_images"
                self.save_images(x_file_path,self.xtest_patches,"xtest")

            # testing labels creation
            if os.path.exists(self.test_folder + "/testing_labels/"):
                print("I will read test labels from existing directory")
                # shutil.rmtree(self.test_folder + "/testing_labels")  # Removes all the subdirectories!
                # os.makedirs(self.test_folder + "/testing_labels")
                self.ytest_patches = []
                for i in os.listdir(self.test_folder  + "/testing_labels/"):
                    im = rasterio.open(self.test_folder  + "/testing_labels/"+i).read()
                    self.ytest_patches.append(im)
                self.ytest_patches = np.asarray(self.ytest_patches)
                print("read; shape is:", self.ytest_patches.shape)
            else:
                os.makedirs(self.test_folder + "/testing_labels/")
                print("It's still gonna take a while...have some patience")
                self.ytest_patches =Patch.image_patching(self,self.Ytest_main)
                # self.Ytest = Augumentation.augumentation(self,self.ytest_patches)
                y_file_path = self.test_folder +"/testing_labels/"
                self.save_images(y_file_path,self.ytest_patches,"ytest")
        else:
            print("First, be sure what you want")
    
    def torch_nothot(self):
        self.create_directory()
        if self.for_what == "training_source":
        #for training dataset
            self.tensor_xs = torch.Tensor(self.Xs_train) 
            self.tensor_ys= torch.Tensor(self.Ys_train)
            self.tensor_xs_train = TensorDataset(self.tensor_xs,self.tensor_ys) 
            self.source_dataloader = DataLoader(self.tensor_xs_train) 
            print("Finally atleast train dataloader section works 😌 ")
        elif self.for_what == "training_target":
        #for training dataset
            self.tensor_xt = torch.Tensor(self.Xt_train.astype(np.int32)) 
            #sending fake labels
            fakearray = np.ones((self.Xt_train.shape))
            self.tensor_yt = torch.Tensor(fakearray)
            self.tensor_xt_train = TensorDataset(self.tensor_xt,self.tensor_yt) 
            self.target_dataloader = DataLoader(self.tensor_xt_train) 
            # print(f"send to dataloader datatype{type(self.tensor_xt)}")
            print("Finally atleast target train dataloader section works 😌 ")
        else:
            #for testing dataset
            self.tensor_xp = torch.Tensor(self.xtest_patches) 
            self.tensor_yp = torch.Tensor(self.ytest_patches)
            self.tensor_test = TensorDataset(self.tensor_xp,self.tensor_yp) 
            self.test_dataloader = DataLoader(self.tensor_test) 
            print("Finally atleast test dataloader section works 😌")
            
    def array_torch(self):
        OneHotEncoding.binary_hot_encoding(self)
        if self.for_what == "training_source":
        #for train source dataset
            self.tensor_xs = torch.Tensor(self.Xs_train) 
            self.tensor_ys = torch.Tensor(self.Ys_train)
            self.tensor_xs_train = TensorDataset(self.tensor_xs,self.tensor_ys) 
            self.source_dataloader = DataLoader(self.tensor_xs_train) 
            print("Finally atleast train source dataloader section works 😌 ")
        elif self.for_what == "training_target":
        #for train target dataset
            self.tensor_xt = torch.Tensor(self.Xt_train, dtype=torch.float32) 
            #self.tensor_yt = torch.Tensor(self.Yt_train)
            self.tensor_xt_train = TensorDataset(self.tensor_xt,self) 
            self.target_dataloader = DataLoader(self.tensor_xt_train) 
            print("Finally atleast train target dataloader section works 😌 ")
        else:
            #for testing dataset
            self.tensor_xp = torch.Tensor(self.xtest_patches) 
            self.tensor_yp = torch.Tensor(self.Ytest)
            self.tensor_test = TensorDataset(self.tensor_xp,self.tensor_yp) 
            self.test_dataloader = DataLoader(self.tensor_test) 
            print("Finally atleast test dataloader section works 😌")




# if __name__ == "__main__":
#     DATASET = Dataset()
#     if a == "avoid hot":
#         DATASET.torch_nothot()
#     else:
#         DATASET.array_torch()





