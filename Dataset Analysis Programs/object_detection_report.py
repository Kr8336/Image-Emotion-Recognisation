from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

category = {"amusement:":4923, "anger:":1255, "awe:":3133, "contentment:":5356, "disgust:":1657, "excitement:":2914, "fear:":1046, "sadness:":2901}

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
           'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def calculate_combined_mean(category, classes):
    '''
    Returns the data outputted by the object_detection_analysis.py tool and also
    returns the combined_mean of the average amount of objects per image in each category.
    '''
    with open("object_detection_data.txt", "r") as f: # Read lines from file
        lines = [line.strip() for line in f.readlines()]

    counter_data = {}

    # Convert the strings from the .txt file back into Counter objects
    for i,l in enumerate(lines):
        if l not in category.keys():
            counter_data[lines[i-1]] = eval(l)

    # Calculate the average number of every object per category
    for c in category.keys():
        for k in counter_data[c]:
            counter_data[c][k]/=category[c]


    combined_mean = Counter()
    denominator = sum(category.values())

    # Calculate the combined mean of each category for each object to find the most common objects across all categories
    for obj in classes:
        numerator = 0
        for c in category.keys():
            if obj not in counter_data[c]:
                counter_data[c][obj] = 0
            numerator += counter_data[c][obj] * category[c]
        combined_mean[obj] = numerator/denominator

    return counter_data, combined_mean

def create_heatmap(category, counter_data, combined_mean):
    '''
    This function takes data from the combined mean and returns a heatmap
    showing the 10 most common objects for each of the categories.
    '''
    # Create a dict with the most common words for each category
    most_common_objects = {}
    filtered_combined_mean = np.delete((np.array(combined_mean.most_common(11))), 1, axis=0) # Remove clock as it was classified incorrectly (a missing image is classified as clock)
    print(filtered_combined_mean)

    for c in category:
        most_common_objects[c] = {key: value for key, value in counter_data[c].items() if key in filtered_combined_mean[:,0]}

    # Convert Counter objects to DataFrames
    dataframes = []
    for c in category:
        dataframes.append(pd.DataFrame.from_dict(most_common_objects[c], orient="index", columns=[c[:-1]]))

    # Create a new DataFrame for the combined data
    df = pd.concat(dataframes, axis=1, join="inner")
    df_norm = df.div(df.max(axis=1), axis=0)

    print(df)
    print(df_norm.keys())

    # Create the heatmap
    heatmap = sns.heatmap(df_norm, cmap='YlGnBu', annot=True, fmt='g') #Sequential colormap
    # sns.heatmap(df_norm, cmap='RdBu_r', center=0.5, annot=True, fmt='g', vmin=0, vmax=1) # Divergent color palette
    plt.title('Normalized Average Object per Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Object')
    plt.show()


counter_data, combined_mean = calculate_combined_mean(category, classes)
create_heatmap(category, counter_data, combined_mean)
