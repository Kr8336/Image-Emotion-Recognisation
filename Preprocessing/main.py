

import os
import sys

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

if __name__ == '__main__':

    img_shape = (224, 224)
    folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Flickr\Flickr'
    save_folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Resized_FI'

    if not os.path.isdir(save_folder_name):
        print('Creating Reshaped images dataset...')
        os.mkdir(save_folder_name)
    classes = os.listdir(folder_name)

    for clas in classes:

        if not os.path.isdir(os.path.join(save_folder_name, clas)):
            print('Creating folder of class', clas)
            os.mkdir(os.path.join(save_folder_name, clas))

        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)

        for file in files:
            file_name = os.path.join(class_name, file)
            save_file_name = os.path.join(save_folder_name, clas, file)
            # print(save_file_name)
            # define the transforms
            img = Image.open(file_name)

            transform = T.Resize((224, 224))
            resized_img = transform(img)

            resized_img.save(save_file_name)
