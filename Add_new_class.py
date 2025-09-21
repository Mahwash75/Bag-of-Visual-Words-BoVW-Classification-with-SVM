# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:18:08 2025

@author: HP
"""

import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC  # Using SVM instead of KNN
import matplotlib.pyplot as plt

# Load images
def load_images(path, category):
    images = []
    for file in glob.glob(f'{path}/{category}*.jpg') + glob.glob(f'{path}/{category}*.jpeg'):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f'Warning: Could not read {file}')
    return images

path = 'D:/Computer Vision/bovwData/bovwData'
cat_images = load_images(path, 'cat')
chair_images = load_images(path, 'chair')
cycle_images = load_images(path, 'cycle')  # New Class

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Extract SIFT features
def extract_sift_features(images):
    descriptors = []
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors.extend(des)
    return np.array(descriptors) if descriptors else np.empty((0, 128))

cat_descriptors = extract_sift_features(cat_images[:10])
chair_descriptors = extract_sift_features(chair_images[:10])
cycle_descriptors = extract_sift_features(cycle_images[:10])  # New Class

# Combine descriptors for clustering
all_descriptors = np.vstack((cat_descriptors, chair_descriptors, cycle_descriptors))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=200, n_init=10, random_state=42)
kmeans.fit(all_descriptors)
codewords = kmeans.cluster_centers_

# Compute histograms for images
def compute_histograms(images, codewords):
    histograms = []
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            hist = np.zeros(len(codewords))
            labels = kmeans.predict(des)
            for label in labels:
                hist[label] += 1
            histograms.append(hist)
    return np.array(histograms) if histograms else np.empty((0, len(codewords)))

# Compute histograms for training images
cat_hist_train = compute_histograms(cat_images[:10], codewords)
chair_hist_train = compute_histograms(chair_images[:10], codewords)
cycle_hist_train = compute_histograms(cycle_images[:10], codewords)  # New Class

# Create training labels
cat_labels = np.zeros(len(cat_hist_train))
chair_labels = np.ones(len(chair_hist_train))
cycle_labels = np.full(len(cycle_hist_train), 2)  # Label for new class

# Combine training data and labels
X_train = np.vstack((cat_hist_train, chair_hist_train, cycle_hist_train))
y_train = np.hstack((cat_labels, chair_labels, cycle_labels))

# Train SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)  # Using Linear SVM
svm.fit(X_train, y_train)

# Compute histograms for test images
cat_hist_test = compute_histograms(cat_images[10:], codewords)
chair_hist_test = compute_histograms(chair_images[10:], codewords)
cycle_hist_test = compute_histograms(cycle_images[10:], codewords)  # New Class

# Create test labels
cat_labels_test = np.zeros(len(cat_hist_test))
chair_labels_test = np.ones(len(chair_hist_test))
cycle_labels_test = np.full(len(cycle_hist_test), 2)

# Combine test data and labels
X_test = np.vstack((cat_hist_test, chair_hist_test, cycle_hist_test))
y_test = np.hstack((cat_labels_test, chair_labels_test, cycle_labels_test))

# Predict using SVM classifier
y_pred = svm.predict(X_test)

# Display classification results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def show_prediction(ax, image, label):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Predicted: {"Cat" if label == 0 else "Chair" if label == 1 else "Cycle"}')
    ax.axis('off')

# Ensure test images exist before displaying results
if len(cat_images) > 10 and len(chair_images) > 10 and len(cycle_images) > 10:
    show_prediction(axs[0], cat_images[10], y_pred[0])
    show_prediction(axs[1], chair_images[10], y_pred[1])
    show_prediction(axs[2], cycle_images[10], y_pred[2])
else:
    print("Not enough test images available to display predictions.")

plt.show()