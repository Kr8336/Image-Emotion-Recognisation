import os
import random
import shutil
import argparse
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from random import randrange


def augment(source_folder, destination_folder, min_class_size, img_shape):
    # Check if destinationfolder exists
    if not os.path.isdir(destination_folder):
        print('Creating Augmented images dataset...')
        os.mkdir(destination_folder)
    else:
        print('Found Existing directory!')
    classes = os.listdir(source_folder)

    # list classes in dataset
    for clas in classes:
        if clas != 'fear':
            continue
        if not os.path.isdir(os.path.join(destination_folder, clas)):
            print('Creating folder of class', clas)
            os.mkdir(os.path.join(destination_folder, clas))

        # class path
        class_name = os.path.join(source_folder, clas)
        files = os.listdir(class_name)
        idx = len(files)
        random.shuffle(files)
        print('Augmenting for Class', clas)
        # Copy pasting images from the normalized dataset
        for file in tqdm(files[:min_class_size]):
            shutil.copy(os.path.join(class_name, file), os.path.join(destination_folder, clas, file))

        # open images in class
        for file in tqdm(files):

            if idx >= min_class_size:
                print('3000 images extracted for class', clas)
                break

            file_name = os.path.join(class_name, file)

            img = Image.open(file_name)
            aug_img1 = T.GaussianBlur(kernel_size=(51, 91), sigma=randrange(1, 3))(img)
            aug_img1 = T.RandomRotation(degrees=randrange(15, 90))(aug_img1)
            aug1_file = os.path.join(destination_folder, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
            aug_img1.save(aug1_file)
            idx += 1

            if len(os.listdir(os.path.join(destination_folder, clas))) < min_class_size // 2:
                aug_img2 = T.ColorJitter()(img)
                aug_img2 = T.RandomErasing(p=0.7, scale=(0.025, 0.5), ratio=(0.3, 2))(aug_img2)
                aug2_file = os.path.join(destination_folder, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
                idx += 1
                aug_img2.save(aug2_file)

            if len(os.listdir(os.path.join(destination_folder, clas))) < min_class_size // 3:
                aug_img3 = T.ElasticTransform(alpha=50, sigma=5)(img)
                aug_img3 = T.RandomHorizontalFlip()(aug_img3)
                aug_img3 = T.RandomVerticalFlip()(aug_img3)

                aug3_file = os.path.join(destination_folder, clas, file_name.split('.')[0][:-4] + str(idx) + '.jpg')
                idx += 1
                aug_img3.save(aug3_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='augment.py',
            description='Augments dataset')

    parser.add_argument('-i', '--input_folder', default=r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Normalized_FI')
    parser.add_argument('-s', '--seed', default=10, type=int)
    parser.add_argument('-o', '--output_folder', default=r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Final_FI')
    parser.add_argument('-m', '--min_class_size', default=3000, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    img_shape = (224, 224)

    augment(args.input_folder, args.output_folder, args.min_class_size, img_shape)