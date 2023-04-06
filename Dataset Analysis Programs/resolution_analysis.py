import numpy as np
from skimage import io

# Path to dataset
img_category_dir = "D:/Uni UK/Msc Artificial Intelligence/Semester 2/CM316 Applications of Machine Learning/Coursework 2/Flickr"
category = {"amusement":4923, "anger":1255, "awe":3133, "contentment":5356, "disgust":1657, "excitement":2914, "fear":1046, "sadness":2901}

# Load the dataset into img_collection and seperate them by classification
img_collection = {}
for c in category.keys():
        img_collection[f"{c}"] = io.imread_collection(img_category_dir + "/" + c + "/*.jpg")

# Calculate the height and width of each image
for c in category.keys():
    heights = []
    widths = []
    for img in img_collection[f"{c}"]:
        height, width, _ = img.shape
        heights.append(height)
        widths.append(width)
    # Calculate the mean height and width
    mean_height = np.mean(heights)
    mean_width = np.mean(widths)

    print(f"Average image size of {c}: {mean_width:0.0f}x{mean_height:0.0f}")
