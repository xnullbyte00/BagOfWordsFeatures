from preprocess import DatasetCreation
from features import ExtractFeatures
import model

from configurations import *

system(CLEAR_SCREEN)
print("=====================================================================")
print("         PREPROCESSING OF DATA TO MAKE BAG OF VISION WORDS           ")
print("=====================================================================")

CODEBOOK_SELECTION = 3
do_preprocess = False
while (CODEBOOK_SELECTION > 2):
    CODEBOOK_SELECTION = int(input('\n\nWhich process you would like to do?\n 1. Complete process (Preprocess and Train)\n 2. Train the dataset only\n Your Choice: '))
    if (CODEBOOK_SELECTION == 1):
            do_preprocess = True
            break
    elif (CODEBOOK_SELECTION == 2):
            do_preprocess = False
            break
    else:
            print("\n\nNo appropriate selection has been made\n\n")
            
if (do_preprocess):
    data_features_list = []
    
    
    print("Data matrix has been made out of text dataset files....\n\n\n")
    for i in tqdm(range(0, len(files))):
        data_features_list.append(DatasetCreation(files[i], include_label = False).DataBeforeKmeans())
    
    splitted_features_list = ExtractFeatures.getspecificFeatureMatrix(data_features_list)
    
    print("\n\n\nCodebooks of HOG, MBHx and MBHy are in progress.")
    print("It would be longer process. Please be patent....\n\n\n")
    codebook_list = []
    for i in tqdm(range(0, 3)):
        codebook_list.append(ExtractFeatures(splitted_features_list[i], standardize = False).getFinalCodeBook())
    
    print("\n\n\n .csv files of data matrix are being generated ....")
    
    DatasetCreation.generateCSVFileFromCodeBook(codebook_list, files, "dataset.csv")
    
print("\n\n\n Training the model...")
model.main()

print("\n\n\n The process is complete.....")


'''
/* Will be used later---


print("Files are being preprocessed....\n\n\n")
for i in tqdm(range(0, len(files))):
    DatasetCreation(files[i]).start()
print("\n\n\nAll files have been converted into .csv format. Please check at: ", path.join(getcwd(),csv_files_path))

*/-------
'''