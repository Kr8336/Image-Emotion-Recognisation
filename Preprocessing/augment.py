# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import time
import torch
import random
import shutil
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from random import randrange

# from torchvision.datasets import ImageFolder


if __name__ == '__main__':

    random.seed(10)
    min_class_size = 3000
    img_shape = (224, 224)
    folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Normalized_FI'
    save_folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Final_FI'
    if not os.path.isdir(save_folder_name):
        print('Creating Augmented images dataset...')
        os.mkdir(save_folder_name)
    else:
        print('Found Existing directory!')
    classes = os.listdir(folder_name)

    for clas in classes:

        if not os.path.isdir(os.path.join(save_folder_name, clas)):
            print('Creating folder of class', clas)
            os.mkdir(os.path.join(save_folder_name, clas))


        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)
        idx = len(files)
        random.shuffle(files)

        # Copy pasting images from the normalized dataset
        for file in files[:min_class_size]:
            shutil.copy(os.path.join(class_name, file), os.path.join(save_folder_name, clas, file))

        for file in files:

            if idx >= min_class_size:
                print('3000 images extracted for class', clas)
                break

            file_name = os.path.join(class_name, file)

            img = Image.open(file_name)
            augmented_img = T.GaussianBlur(kernel_size=(51, 91), sigma=randrange(1, 3))(img)
            augmented_img = T.RandomRotation(degrees=randrange(15, 90))(augmented_img)
            # print('augmenting image:', file)
            # the below line is prone to errors for idx less than 1000, but
            # it will work for us since all folders already have more than 1000 files
            save_aug_file = os.path.join(save_folder_name, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
            idx+=1
            if len(os.listdir(os.path.join(folder_name, clas)))<min_class_size//2:
                augmented_img2 = T.ColorJitter()(img)
                augmented_img2 = T.RandomErasing(p=0.7, scale=(0.025, 0.5), ratio=(0.3, 2))(augmented_img2)
                save_aug_file2 = os.path.join(save_folder_name, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
                idx+=1
                augmented_img.save(save_aug_file2)

            if len(os.listdir(os.path.join(folder_name, clas)))<min_class_size//3:
                augmented_img3 = T.ElasticTransform(alpha=50, sigma=5)(img)
                augmented_img3 = T.RandomHorizontalFlip()(augmented_img3)
                augmented_img3 = T.RandomVerticalFlip()(augmented_img3)

                save_aug_file3 = os.path.join(save_folder_name, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
                idx += 1
                augmented_img.save(save_aug_file3)
            # augmented_img.show()

            # if augmented_img.shape != (224, 224):
            #     print('Image corrupted!')
            #     sys.exit(0)
