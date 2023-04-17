import os
import sys
import torch
import imagehash
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


## Helper function for hashing
def alpha_remover(image):
    if image.mode != 'RGBA':
        return image
    canvas = Image.new('RGBA', image.size, (255, 255, 255, 255))
    canvas.paste(image, mask=image)
    return canvas.convert('RGB')


##Function for image hashing
def with_ztransform_preprocess(path, hashfunc, hash_size=8):
    image = alpha_remover(Image.open(path))
    image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
    data = image.getdata()
    quantiles = np.arange(100)
    quantiles_values = np.percentile(data, quantiles)
    zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
    image.putdata(zdata)

    return hashfunc(image)


def remove_images(files):
    for file in files :
        os.remove(file)

def main(path_to_dataset=r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of '
                         'Machine Learning\datasets_coursework2\Flickr\Flickr',
         base_img=r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine '
                  r'Learning\datasets_coursework2\Flickr\Flickr\amusement\amusement_0668.jpg',
         emp_img_len=500,
         emp_img_bre=374, ):
    classes = os.listdir(path_to_dataset)
    base_hash = with_ztransform_preprocess(path=base_img, hashfunc=imagehash.dhash, hash_size=8)
    counter = 0
    files_to_delete = []

    for clas in classes:
        class_path = os.path.join(path_to_dataset, clas)
        files = os.listdir(class_path)

        for file in files:
            image_filename = os.path.join(class_path, file)
            if image_filename == base_img:
                continue
            im = Image.open(image_filename)
            if im.size == (emp_img_len, emp_img_bre):

                if with_ztransform_preprocess(path=image_filename, hashfunc=imagehash.dhash, hash_size=8) == base_hash:
                    files_to_delete.append(image_filename)
                    # os.remove(image_filename)
                    print('Empty Image found!')
                    counter += 1
                else:
                    print('Image is not empty!')

    print('Extracted and Removed', counter, 'images from Flickr Dataset ')
    files_to_delete.append(base_img)
    remove_images(files_to_delete)


if __name__ == '__main__':
    main()
