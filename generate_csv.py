import csv
import os
from tqdm import tqdm
files = os.listdir()

def GenerateCSVFile(dataset):
    with open("files_list.csv", 'w', newline="") as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerows(dataset)

new_files = []
for i in tqdm(range(0, len(files))):
    new_files.append([files[i]])
GenerateCSVFile(new_files)