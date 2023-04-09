import os
import sys
import torch
import torchvision
import pycocotools-windows

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

if __name__ == '__main__':

    folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine Learning\datasets_coursework2\Flickr\Flickr'

    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
    model.eval()

    if not os.path.isdir(folder_name):
        print('Folder does not exist, please enter correct directory... ')
        sys.exit(0)

    classes = os.listdir(folder_name)
    string_object = []
    for clas in classes:
        class_name = os.path.join(folder_name, clas)
        files = os.listdir(class_name)

        class_strings = []
        for file in files:
            file_name = os.path.join(class_name, file)
            img = Image.open(file_name)

            img_tensor = pil_to_tensor(img)
            img_tensor_normalized = img_tensor/255.0
            predictions = model(img_tensor_normalized)
            import pdb; pdb.set_trace()
        string_object.append(class_strings)
