import os
import sys
from random import randrange
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image

## Exploratory data analysis for image dimensions
def eda(folder_name):

    classes = os.listdir(folder_name)
    shapes = []
    for clas in classes:

        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)

        for file in files:
            file_name = os.path.join(class_name, file)
            img = Image.open(file_name)
            w, h = img.size
            shapes.append((w, h))

    print(len(set(shapes)))
    return shapes


## Image augmentation example
def sample_images(file_name):
    ## sample augmented images
    deskpath = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop'
    img = Image.open(file_name)
    img.save(os.path.join(deskpath, 'orig_img.jpg'))

    augmented_img = T.GaussianBlur(kernel_size=(51, 91), sigma=randrange(1, 3))(img)
    augmented_img = T.RandomRotation(degrees=randrange(15, 90))(augmented_img)

    augmented_img.save(os.path.join(deskpath, 'aug1.jpg'))
    idx += 1
    augmented_img2 = T.ColorJitter()(img)
    augmented_img2 = T.RandomErasing(p=0.7, scale=(0.025, 0.5), ratio=(0.3, 2))(augmented_img2)
    augmented_img2.save(os.path.join(deskpath, 'aug2.jpg'))

    # if len(os.listdir(os.path.join(folder_name, clas)))<min_class_size//3:
    augmented_img3 = T.ElasticTransform(alpha=50.0, sigma=5.0)(img)
    augmented_img3 = T.RandomHorizontalFlip()(augmented_img3)
    augmented_img3 = T.RandomVerticalFlip()(augmented_img3)
    augmented_img3.save(os.path.join(deskpath, 'aug3.jpg'))


if __name__ == '__main__':
    shapes = eda(r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine ' \
                 r'Learning\datasets_coursework2\Flickr\Flickr')

    sample_images(r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine '
                  r'Learning\datasets_coursework2\Flickr\Flickr\fear\fear_0000.jpg')
