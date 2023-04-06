from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, util
import numpy as np

# Path to dataset
img_category_dir = "D:/Uni UK/Msc Artificial Intelligence/Semester 2/CM316 Applications of Machine Learning/Coursework 2/Flickr"
category = {"amusement":4923, "anger":1255, "awe":3133, "contentment":5356, "disgust":1657, "excitement":2914, "fear":1046, "sadness":2901}

# Load the dataset into img_collection and seperate them by classification
img_collection = {}
for c in category.keys():
        img_collection[f"{c}"] = io.imread_collection(f"{img_category_dir}/{c}/*.jpg")

for c in category.keys():
    textures = []
    for img in img_collection[c]:
        # Convert image to grayscale then as unsigned integer
        gray_img = color.rgb2gray(img)
        gray_img = util.img_as_ubyte(gray_img)

        # Calculate the GLCM Matrix
        gray_level_matrix = graycomatrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)

        # Calculate the properties of the textures
        energy = graycoprops(gray_level_matrix, 'energy')[0, 0]
        textures.append(energy)

    print(f"{c}:")
    print(f"The number of smooth images are {len([en for en in textures if en > 0.5])}.")
    print(f"The number of rough images are {len([en for en in textures if en <= 0.5])}.")
