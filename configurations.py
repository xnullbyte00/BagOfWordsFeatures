
from os import listdir, getcwd, name, system, path
import csv

CLEAR_SCREEN = "cls"
if (name == "posix"):
    CLEAR_SCREEN  = "clear"

try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from scipy.cluster.vq import kmeans, vq
    from sklearn.preprocessing import StandardScaler


    HOG = 0*96
    MBHx = 1*96
    MBHy = 2*96
    BATCH_SIZE = 20
    FEATURES = 1000
    txt_files_path="files"
    csv_files_path="csv"
    CLEAR_SCREEN = "cls"
    if (name == 'posix'):
        CLEAR_SCREEN = "clear"
    files = listdir(txt_files_path)
    csv_file_names = ["HOG_features.csv", "MBHx_features.csv", "MBHy_features.csv"] 
except:
    system(CLEAR_SCREEN)
    raise Exception("\n\n\n\n\nCannot find libraries. Please install them by giving command 'pip install -r requirements.txt' ")