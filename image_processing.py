import cv2
import numpy as np
from sklearn.cluster import KMeans

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_flattened = image_lab.reshape((-1, 3))

    kmeans = KMeans(n_clusters=4, random_state=0).fit(lab_flattened)
    cluster_labels = kmeans.labels_.reshape(image_lab.shape[:2])
    cluster_means = kmeans.cluster_centers_
    sorted_indices = np.argsort([mean[1] for mean in cluster_means])

    category_mapping = {
        sorted_indices[0]: "water",
        sorted_indices[1]: "dry",
        sorted_indices[2]: "wet",
        sorted_indices[3]: "salty_muddy",
    }
    
    total_pixels = cluster_labels.size
    result = {label: np.sum(cluster_labels == idx) / total_pixels * 100 for idx, label in category_mapping.items()}
    return result
