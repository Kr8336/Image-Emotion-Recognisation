import numpy as np
from skimage import io, color

# Path to dataset
img_category_dir = "D:/Uni UK/Msc Artificial Intelligence/Semester 2/CM316 Applications of Machine Learning/Coursework 2/Flickr"
category = {"amusement":4923, "anger":1255, "awe":3133, "contentment":5356, "disgust":1657, "excitement":2914, "fear":1046, "sadness":2901}

# Load the dataset into img_collection and seperate them by classification
img_collection = {}
for c in category.keys():
        img_collection[f"{c}"] = io.imread_collection(f"{img_category_dir}/{c}/*.jpg")

# Calculate the brightness of each image
for c in category.keys():
    brightness_list = []
    for img in img_collection[f"{c}"]:
        grayscale_img = color.rgb2gray(img)
        brightness = grayscale_img.mean()
        brightness_list.append(brightness)
    # Calculate the mean brightness
    mean_brightness = np.mean(brightness_list)

    print(f"Average image brightness of {c}: {mean_brightness:0.2f}")
