import numpy as np
import pandas as pd
import matplotlib.pyplot as plt







# read csv file for task1
train_meta = pd.read_csv("task1_data/train_metadata.csv")
test_meta = pd.read_csv("task1_data/test_metadata.csv")
color = pd.read_csv("task1_data/color_histogram.csv")
hog = pd.read_csv("task1_data/hog_pca.csv")
add = pd.read_csv("task1_data/additional_features.csv")

features = color.merge(hog, on="image_id")
features = features.merge(add, on="image_id")