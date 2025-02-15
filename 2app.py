# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Step 1: Load the RGB image
# image_path = "data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_050.jpg"
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Step 2: Convert to HSV color space and extract features
# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# saturation = image_hsv[:, :, 1]
# value = image_hsv[:, :, 2]

# # Flatten the features for clustering
# features = np.stack([saturation.flatten(), value.flatten()], axis=1)

# # Step 3: Apply K-Means clustering
# kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
# cluster_labels = kmeans.labels_.reshape(saturation.shape)

# # Step 4: Assign categories based on cluster means
# cluster_means = [features[cluster_labels.flatten() == i].mean(axis=0) for i in range(3)]
# sorted_indices = np.argsort([mean[1] for mean in cluster_means])  # Sort by brightness (value)
# category_mapping = {sorted_indices[0]: 0,  # Dry
#                     sorted_indices[1]: 1,  # Wet
#                     sorted_indices[2]: 2}  # Water
# classified_image = np.vectorize(category_mapping.get)(cluster_labels)

# # Step 5: Calculate percentages
# total_pixels = classified_image.size
# dry_pixels = np.sum(classified_image == 0)
# wet_pixels = np.sum(classified_image == 1)
# water_pixels = np.sum(classified_image == 2)

# dry_percentage = (dry_pixels / total_pixels) * 100
# wet_percentage = (wet_pixels / total_pixels) * 100
# water_percentage = (water_pixels / total_pixels) * 100

# # Print results
# print(f"Dry Soil: {dry_percentage:.2f}%")
# print(f"Wet Soil: {wet_percentage:.2f}%")
# print(f"Water: {water_percentage:.2f}%")

# # Step 6: Visualization
# # Display classified image
# color_map = {0: [139, 69, 19],  # Dry (Brown)
#              1: [173, 216, 230],  # Wet (Light Blue)
#              2: [0, 0, 255]}  # Water (Blue)

# classified_visual = np.zeros((*classified_image.shape, 3), dtype=np.uint8)
# for label, color in color_map.items():
#     classified_visual[classified_image == label] = color

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_rgb)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Classified Image")
# plt.imshow(classified_visual)
# plt.axis('off')
# plt.show()

# # Display distribution as a pie chart
# labels = ['Dry Soil', 'Wet Soil', 'Water']
# sizes = [dry_percentage, wet_percentage, water_percentage]
# colors = ['sandybrown', 'lightblue', 'blue']

# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
# plt.title("Soil Moisture Distribution")
# plt.show()


# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Step 1: Load the satellite image
# image_path = "data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_050.jpg"
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Step 2: Convert to LAB color space for better color segmentation
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# lab_flattened = image_lab.reshape((-1, 3))  # Flatten for clustering

# # Step 3: Apply K-Means clustering
# n_clusters = 4  # Four categories: Water, Dry Soil, Wet Soil, Salty-Muddy
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(lab_flattened)
# cluster_labels = kmeans.labels_.reshape(image_lab.shape[:2])

# # Step 4: Assign categories based on cluster means
# cluster_means = kmeans.cluster_centers_
# sorted_indices = np.argsort([mean[1] for mean in cluster_means])  # Sort by LAB brightness

# # Map clusters to categories based on your clarified observations
# category_mapping = {
#     sorted_indices[0]: 0,  # Water (whitish)
#     sorted_indices[1]: 1,  # Dry Soil (greenish)
#     sorted_indices[2]: 2,  # Wet Soil (yellow)
#     sorted_indices[3]: 3,  # Salty-Muddy (brown)
# }
# classified_image = np.vectorize(category_mapping.get)(cluster_labels)

# # Step 5: Calculate percentages
# total_pixels = classified_image.size
# water_pixels = np.sum(classified_image == 0)
# dry_pixels = np.sum(classified_image == 1)
# wet_pixels = np.sum(classified_image == 2)
# salty_muddy_pixels = np.sum(classified_image == 3)

# water_percentage = (water_pixels / total_pixels) * 100
# dry_percentage = (dry_pixels / total_pixels) * 100
# wet_percentage = (wet_pixels / total_pixels) * 100
# salty_muddy_percentage = (salty_muddy_pixels / total_pixels) * 100

# # Print results
# print(f"Water: {water_percentage:.2f}%")
# print(f"Dry Soil: {dry_percentage:.2f}%")
# print(f"Wet Soil: {wet_percentage:.2f}%")
# print(f"Salty-Muddy: {salty_muddy_percentage:.2f}%")

# # Step 6: Visualization
# # Assign colors for visualization
# color_map = {
#     0: [0, 128, 0], # White for Water
#     1: [165, 42, 42],      # Green for Dry Soil
#     2: [255, 255, 0],    # Yellow for Wet Soil
#     3: [255, 255, 255],   # Brown for Salty-Muddy
# }

# classified_visual = np.zeros((*classified_image.shape, 3), dtype=np.uint8)
# for label, color in color_map.items():
#     classified_visual[classified_image == label] = color

# # Combine Pie Chart and Map in one frame
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# # Display Classified Map
# ax[0].imshow(classified_visual)
# ax[0].set_title("Classified Map")
# ax[0].axis('off')

# # Display Pie Chart
# labels = ['Water', 'Dry Soil', 'Wet Soil', 'Salty-Muddy']
# sizes = [water_percentage, dry_percentage, wet_percentage, salty_muddy_percentage]
# colors = ['green', 'white', 'yellow', 'brown']

# ax[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
# ax[1].set_title("Soil and Water Distribution")

# plt.tight_layout()
# plt.show()



import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the satellite image
image_path = "data/2km_Jan21-Dec24_Sentinel-2_L2A-855842435060540-timelapse_050.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to LAB color space and apply median blur
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_lab = cv2.medianBlur(image_lab, 3)  # Reduce noise
lab_flattened = image_lab.reshape((-1, 3))

# K-Means Clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(lab_flattened)
cluster_labels = kmeans.labels_.reshape(image_lab.shape[:2])

# Analyze cluster LAB means
cluster_means = kmeans.cluster_centers_
for i, mean in enumerate(cluster_means):
    print(f"Cluster {i}: LAB mean = {mean}")

# Dynamic Mapping of Clusters
sorted_indices = np.argsort([mean[1] for mean in cluster_means])  # Sort by LAB brightness
category_mapping = {
    sorted_indices[0]: 0,  # Water
    sorted_indices[1]: 1,  # Dry Soil
    sorted_indices[2]: 2,  # Wet Soil
    sorted_indices[3]: 3,  # Salty-Muddy
}
classified_image = np.vectorize(category_mapping.get)(cluster_labels)

# Calculate Percentages
total_pixels = classified_image.size
percentages = {
    'Water': np.sum(classified_image == 0) / total_pixels * 100,
    'Wet Soil': np.sum(classified_image == 1) / total_pixels * 100,
    'Salty-Muddy': np.sum(classified_image == 2) / total_pixels * 100,
    'Dry Soil': np.sum(classified_image == 3) / total_pixels * 100,
}
for category, perc in percentages.items():
    print(f"{category}: {perc:.2f}%")

# Visualization
color_map = {
    0: [0, 0, 255],  # Blue for Water
    1: [139, 69, 19],  # Brown for Dry Soil
    2: [124, 252, 0],  # Light Green for Wet Soil
    3: [255, 255, 255],  # White for Salty-Muddy
}

classified_visual = np.zeros((*classified_image.shape, 3), dtype=np.uint8)
for label, color in color_map.items():
    classified_visual[classified_image == label] = color

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(classified_visual)
ax[0].set_title("Classified Map")
ax[0].axis('off')

labels = list(percentages.keys())
sizes = list(percentages.values())
colors = ['blue', 'brown', 'green', 'white']
ax[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax[1].set_title("Soil and Water Distribution")

plt.tight_layout()
plt.show()
