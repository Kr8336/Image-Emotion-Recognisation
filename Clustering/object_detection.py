import os
import sys
import torch
# import pycocotools-windows

from PIL import Image
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

if __name__ == '__main__':

    folder_name = r'C:\Users\c22056054\OneDrive - Cardiff University\Desktop\SM\Semester-II\Applications of Machine ' \
                  r'Learning\datasets_coursework2\Final_FI'

    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn_v2(pretrained=weights, box_score_threshold=0.95)
    # print(model.eval())

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # preprocess = weights.transforms()


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
            img = read_image(file_name)
            # img.show()
            # batch = [preprocess(img)]
            # prediction = model(batch)[0]
            #
            # labels = [weights.meta["categories"][i] for i in prediction["labels"]]
            # print(labels)

            # box = draw_bounding_boxes(img, boxes=prediction["boxes"],
            #                           labels=labels,
            #                           colors="red",
            #                           width=4, font_size=30)
            # im = to_pil_image(box.detach())
            # im.show()

            results = yolo_model([Image.open(file_name)])
            results.show()
            break
        string_object.append(class_strings)
