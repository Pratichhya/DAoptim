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

# files
#from test_size import Shape
from .patching import Patch
from .augmentation import Augumentation
from .one_hot import OneHotEncoding


class Dataset():
    def __init__(self, data_folder, patchsize, for_what):
        # connecting to the folder
        print("Buckle up, here with start the journey")
        self.data_folder = data_folder
        self.train_folder = self.data_folder + "train_data/"
        self.test_folder = self.data_folder + "test_data/"
        self.patchsize = patchsize
        self.for_what = for_what

    # fetching training and testing folder
    def get_data(self):
        self.train_img_files = []
        self.train_label_files = []
        self.test_img_files = []
        self.test_label_files = []
        # collecting training images files
        for image in os.listdir(self.train_folder):
            if image == "images":
                train_img_folder = os.path.join(self.train_folder, image)
                for img in os.listdir(train_img_folder):
                    train_img_path = os.path.join(train_img_folder, img)
                    self.train_img_files.append(train_img_path)
                    # print(f"training images:{self.train_img_files}")
            elif image == "labels":
                train_labels_folder = os.path.join(self.train_folder, image)
                for lab in os.listdir(train_labels_folder):
                    train_label_path = os.path.join(train_labels_folder, lab)
                    self.train_label_files.append(train_label_path)
                    # print(f"training labels:{self.train_label_files}")

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
            return read_img

    # reading labels
    def read_labels(self, label_list):
        # append all the labels as numpy after reading
        for val in label_list:
            read_labels = rasterio.open(val).read()
            return read_labels

    # loading training and testing dataset
    def load_data(self):
        print("Ok just wait here, I will go search for data")
        self.get_data()
        print("wow, you go girl, you found all the dataset required for the trip")
        self.Xtrain_main = self.read_images(self.train_img_files)
        self.Ytrain_main = self.read_labels(self.train_label_files)
        self.Xtest_main = self.read_images(self.test_img_files)
        self.Ytest_main = self.read_labels(self.test_label_files)

    def save_images(self, path, array, name):
        for i in range(array.shape[0]):
            # Create the output files with each unique file name
            tiff.imwrite(os.path.join(
                path, f"{name}_{i+1}"+".tif"), array[i, :, :, :])
        print("Congratulations!!! 🎉🎉🎉  for saving all your augmented patches")

    # save the created patches in a different folder
    def create_directory(self):
        self.load_data()
        # self.save_images()
        if self.for_what == "training":
            # training images creation
            if os.path.exists(self.train_folder + "/training_images"):
                # Removes all the subdirectories!
                shutil.rmtree(self.train_folder + "/training_images")
                os.makedirs(self.train_folder + "/training_images")
            else:
                os.makedirs(self.train_folder + "/training_images")
            print("working on images...grab some food until then")
            self.xtrain_patches = Patch.image_patching(self, self.Xtrain_main)
            self.Xtrain = Augumentation.augumentation(
                self, self.xtrain_patches)
            x_file_path = self.train_folder + "/training_images"
            self.save_images(x_file_path, self.Xtrain, "xtrain")

            # training labels creation
            if os.path.exists(self.train_folder + "/training_labels"):
                # Removes all the subdirectories!
                shutil.rmtree(self.train_folder + "/training_labels")
                os.makedirs(self.train_folder + "/training_labels")
            else:
                os.makedirs(self.train_folder + "/training_labels")
            print("It's still gonna take a while...have some patience")
            self.ytrain_patches = Patch.image_patching(self, self.Ytrain_main)
            self.Ytrain = Augumentation.augumentation(
                self, self.ytrain_patches)
            x_file_path = self.train_folder + "/training_labels"
            self.save_images(x_file_path, self.Ytrain, "xtrain")

        elif self.for_what == "testing":
            # testing image creation
            if os.path.exists(self.test_folder + "/testing_images"):
                # Removes all the subdirectories!
                shutil.rmtree(self.test_folder + "/testing_images")
                os.makedirs(self.test_folder + "/testing_images")
            else:
                os.makedirs(self.test_folder + "/testing_images")
            print("working on images...grab some food until then")
            self.xtest_patches = Patch.image_patching(self, self.Xtest_main)
            # self.Xtest = Augumentation.augumentation(self,self.xtest_patches)
            x_file_path = self.test_folder + "testing_images"
            self.save_images(x_file_path, self.xtest_patches, "xtest")

            # testing labels creation
            if os.path.exists(self.test_folder + "/testing_labels"):
                # Removes all the subdirectories!
                shutil.rmtree(self.test_folder + "/testing_labels")
                os.makedirs(self.test_folder + "/testing_labels")
            else:
                os.makedirs(self.test_folder + "/testing_labels")
            print("It's still gonna take a while...have some patience")
            self.ytest_patches = Patch.image_patching(self, self.Ytest_main)
            # self.Ytest = Augumentation.augumentation(self,self.ytest_patches)
            y_file_path = self.test_folder + "testing_labels"
            self.save_images(y_file_path, self.ytest_patches, "ytest")
        else:
            print("First, be sure what you want")

    def array_torch(self):
        OneHotEncoding.binary_hot_encoding(self)
        if self.for_what == "training":
            # for training dataset
            self.tensor_x = torch.Tensor(self.Xtrain)
            self.tensor_y = torch.Tensor(self.Ytrain)
            self.tensor_train = TensorDataset(self.tensor_x, self.tensor_y)
            self.train_dataloader = DataLoader(self.tensor_train)
            print("Finally atleast train dataloader section works 😌 ")
        else:
            # for testing dataset
            self.tensor_xp = torch.Tensor(self.xtest_patches)
            self.tensor_yp = torch.Tensor(self.Ytest)
            self.tensor_test = TensorDataset(self.tensor_xp, self.tensor_yp)
            self.test_dataloader = DataLoader(self.tensor_test)
            print("Finally atleast test dataloader section works 😌")


if __name__ == "__main__":
    DATASET = Dataset()
    DATASET.array_torch()
