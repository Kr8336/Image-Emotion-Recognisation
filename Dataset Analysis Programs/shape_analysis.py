from skimage import io, color, metrics, transform
import numpy as np

# Path to dataset
img_category_dir = "D:/Uni UK/Msc Artificial Intelligence/Semester 2/CM316 Applications of Machine Learning/Coursework 2/Flickr"
category = {"amusement":4923, "anger":1255, "awe":3133, "contentment":5356, "disgust":1657, "excitement":2914, "fear":1046, "sadness":2901}

# Load the dataset into img_collection and seperate them by classification
img_collection = {}
for c in category.keys():
        img_collection[f"{c}"] = io.imread_collection(f"{img_category_dir}/{c}/*.jpg")

for c in category.keys():
    symmetries = []
    for img in img_collection[c]:
        # Create a grayscale image and then resize the image to 256x256 to avoid an image having an odd number of pixels
        gray_img = color.rgb2gray(img)
        gray_img = transform.resize(gray_img, (256, 256))

        # Create two sides from one image split down the middle
        height, width = gray_img.shape
        left = gray_img[:, :width//2]
        right = gray_img[:, width//2:]

        # Mirror the left side
        f_left = left[:, ::-1]

        # Calculate the mse by overlapping the flipped left side and the right side
        mse = metrics.mean_squared_error(f_left, left)
        symmetries.append(mse)
    print(f"{c}:")
    print(f"There are {len([i for i in symmetries if i < 0.1])} symmetric images in {c}")
    print(f"There are {len([i for i in symmetries if i > 0.1])} asymmetric images in {c}")
