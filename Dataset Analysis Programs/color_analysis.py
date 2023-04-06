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
    color_list = []
    for img in img_collection[f"{c}"]:
        img_color = img.mean(axis=(0, 1))
        color_list.append(img_color)
    color_list = np.array(color_list)
    # Separate each channel into a list for all images
    red = color_list[:, 0]
    green = color_list[:, 1]
    blue = color_list[:, 2]

    # Calculate mean of each channel
    mean_r = np.mean(red)
    mean_g = np.mean(green)
    mean_b = np.mean(blue)

    print(f"Average image color of {c}:")
    print(f"red: {mean_r:0.0f}, green:{mean_g:0.0f}, blue:{mean_b:0.0f}")
