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
        
    ## if metadata is available for dataset  
    # def read_labels2(self, label_list):
    #     # append all the labels as numpy after reading
    #     for val in label_list:
    #         with rasterio.open(val) as src:
    #             array = src.read()
    #             meta = src.profile
    #         return array, meta

    # loading training and testing dataset
    def load_data(self):
        print("Ok just wait here, I will go search for data")
        self.get_data()
        print("wow, you go girl, you found all the dataset required for the trip")
        self.Xtrain_main = self.read_images(self.train_img_files)
        self.Ytrain_main = self.read_labels(self.train_label_files)
        self.Xtest_main = self.read_images(self.test_img_files)
        self.Ytest_main = self.read_labels(self.test_label_files)
        
        
        # image patching in case of single images
    def image_patching(self,data):
        self.load_data()
        print("But did you check what did they give us? AVOID SHAPE ERROR")
        print(f"initial data shape is: {data.shape}")
        nbands, nrows, ncols = data.shape
        imgarray = data
        patchsamples = np.zeros(
            shape =(0, nbands, self.patchsize, self.patchsize), dtype=imgarray.dtype
        )
        for i in range(int(nrows / self.patchsize)):
            for j in range(int(ncols / self.patchsize)):
                tocat = imgarray[
                    :,
                    i * self.patchsize : (i + 1) * self.patchsize,
                    j * self.patchsize : (j + 1) * self.patchsize,
                ]
                tocat = np.expand_dims(tocat, axis=0)
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
        self.patchsamples = patchsamples
        return self.patchsamples

    # augumentation of data
    def augumentation(self, patches):
        # image flipping
        self.data_fliped = patches[:, :, ::-1, :]
        # image mirroring
        self.data_mirrored = patches[:, :, :, ::-1]
        # appending all together
        self.data_fliped = np.append(patches, self.data_fliped, axis=0)
        self.all_images = np.append(self.data_fliped, self.data_mirrored, axis=0)
        return self.all_images

    def save_images(self,path,array,name):
        for i in range(array.shape[0]):
            #Create the output files with each unique file name
            tiff.imwrite(os.path.join(path, f"{name}_{i}"+".tif"), array[i,:,:,:])
        print("Congratulations!!! 🎉🎉🎉  for saving all your augmented patches")
    
    # save the created patches in a different folder
    def create_directory(self):
        self.load_data()
        # self.save_images()
        if self.for_what == "training":
            # training images creation
            if os.path.exists(self.train_folder + "/training_images"):
                shutil.rmtree(self.train_folder + "/training_images")  # Removes all the subdirectories!
                os.makedirs(self.train_folder + "/training_images")
            else:
                os.makedirs(self.train_folder + "/training_images")
            print("working on images...grab some food until then")
            self.xtrain_patches = self.image_patching(self.Xtrain_main)
            self.Xtrain = self.augumentation(self.xtrain_patches)
            x_file_path = self.train_folder + "/training_images"
            self.save_images(x_file_path,self.Xtrain,"xtrain")

            # training labels creation
            if os.path.exists(self.train_folder + "/training_labels"):
                shutil.rmtree(self.train_folder + "/training_labels")  # Removes all the subdirectories!
                os.makedirs(self.train_folder + "/training_labels")
            else:
                os.makedirs(self.train_folder + "/training_labels")
            print("It's still gonna take a while...have some patience")
            self.ytrain_patches = self.image_patching(self.Ytrain_main)
            self.Ytrain = self.augumentation(self.ytrain_patches)
            x_file_path = self.train_folder + "/training_labels"
            self.save_images(x_file_path,self.Ytrain,"xtrain")

        elif self.for_what == "testing":
            # testing image creation
            if os.path.exists(self.test_folder + "/testing_images"):
                shutil.rmtree(self.test_folder + "/testing_images")  # Removes all the subdirectories!
                os.makedirs(self.test_folder + "/testing_images")
            else:
                os.makedirs(self.test_folder + "/testing_images")
            print("working on images...grab some food until then")
            self.xtest_patches = self.image_patching(self.Xtest_main)
            self.Xtest = self.augumentation(self.xtest_patches)
            x_file_path = self.test_folder +"testing_images"
            self.save_images(x_file_path,self.Xtest,"xtest")

            # testing labels creation
            if os.path.exists(self.test_folder + "/testing_labels"):
                shutil.rmtree(self.test_folder + "/testing_labels")  # Removes all the subdirectories!
                os.makedirs(self.test_folder + "/testing_labels")
            else:
                os.makedirs(self.test_folder + "/testing_labels")
            print("It's still gonna take a while...have some patience")
            self.ytest_patches = self.image_patching(self.Ytest_main)
            self.Ytest = self.augumentation(self.ytest_patches)
            y_file_path = self.test_folder +"testing_labels"
            self.save_images(y_file_path,self.Ytest,"ytest")
        else:
            print("First, be sure what you want")
        
    #applying one-hot encoding in labels
    def one_hot_encoding(self):
        self.create_directory()
        print("one-hot 🔥 encoding")
        if self.for_what == "training":
            ones = np.ones(self.Ytrain.shape)
            self.ytrain_inverted = ones-self.Ytrain # Because 0-1 is 1 and 1-0 is zero
            self.Ytrain = np.concatenate((self.ytrain_inverted,self.Ytrain),axis=1)
            print(self.Ytrain.shape)
        else:
            ones = np.ones(self.Ytest.shape)
            self.ytest_inverted = ones-self.Ytest # Because 0-1 is 1 and 1-0 is zero
            self.Ytest = np.concatenate((self.ytest_inverted,self.Ytest),axis=1)
            print(self.Ytest.shape)
        
    def array_torch(self):
        self.one_hot_encoding()
        if self.for_what == "training":
        #for training dataset
            self.tensor_x = torch.Tensor(self.Xtrain) 
            self.tensor_y = torch.Tensor(self.Ytrain)
            self.tensor_train = TensorDataset(self.tensor_x,self.tensor_y) 
            self.train_dataloader = DataLoader(self.tensor_train) 
            print("Finally atleast train dataloader section works 😌 ")
        else:
            #for testing dataset
            self.tensor_xp = torch.Tensor(self.Xtest) 
            self.tensor_yp = torch.Tensor(self.Ytest)
            self.tensor_test = TensorDataset(self.tensor_xp,self.tensor_yp) 
            self.test_dataloader = DataLoader(self.tensor_test) 
            print("Finally atleast test dataloader section works 😌")



if __name__ == "__main__":
    DATASET = Dataset()
    DATASET.array_torch()
# #creating class object
# DATASET = Dataset("/home/jovyan/private/ubs_prati/thesis/semantic_segmentationPS/data_loader/",128,"testing")
# DATASET.array_torch()




