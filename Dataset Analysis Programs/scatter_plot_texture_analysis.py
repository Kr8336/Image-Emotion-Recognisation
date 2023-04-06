import matplotlib.pyplot as plt

# Define the data
label_percentages = {'Disgust':4.8, 'Contentment':5.0, 'Amusement':5.0, 'Excitement':7.5, 'Awe':8.3, 'Fear':9.9, 'Anger':10.3, 'Sadness':12.5}

# Create a scatter plot using the data
plt.scatter([i for i in range(len(label_percentages))], label_percentages.values())

# Create the axis-titles for the scatter plot
plt.xlabel('Labels')
plt.xticks([i for i in range(len(label_percentages))], label_percentages.keys())
plt.ylabel('Percentage of Smooth Images')
plt.title('Percentage of Smooth Images for each Label')

plt.show()
