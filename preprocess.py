# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:10:18 2021

@author: Ahmad4185
"""

from configurations import *
class DatasetCreation:
    def __init__(self,filename,  hog_starting_point=40,\
                mbhx_starting_point=244, mbhy_starting_point=340, \
                mode = 'r', offset = 96, txt_files_path="files", csv_files_path = "csv", include_label = True):
        
        self.__WorkingDirectory = path.join(getcwd(), txt_files_path)
        try:
            f = open((path.join(self.__WorkingDirectory,filename)), mode)
            self.__data = f.read()
        except:
            raise Exception("Cannot process. Given File has some problems.")
        self.__Path = path.join(getcwd(), csv_files_path, filename.split(".")[0]+'.csv')
        self.__ClassName = filename.split("-")[1]+filename.split("-")[2]
        self.__HOGInitial = hog_starting_point
        self.__MBHxInitial = mbhx_starting_point
        self.__MBHyInitial = mbhy_starting_point
        self.__DataMatrix = []
        self.__HeaderMatrix = []
        self.__offset = offset
        self.__features = ['HOG', 'MBHx', 'MBHy']
        self.__include_label = include_label

    def __GenerateDataMatrix(self):
        rows = self.__data.split('\n')
        for row in rows:
            columns = row.split('\t')
            columns = columns[self.__HOGInitial:self.__HOGInitial+self.__offset]+\
                    columns[self.__MBHxInitial:self.__MBHxInitial+self.__offset]+\
                    columns[self.__MBHyInitial:self.__MBHyInitial+self.__offset]
            if (self.__include_label):
                columns = columns+[self.__ClassName]
            self.__DataMatrix.append(columns)
        self.__DataMatrix= self.__DataMatrix[:-1]

    def __GenerateHeaderMatrix(self):
            self.__HeaderMatrix = []
            for feature in self.__features:
                for i in range(self.__offset):
                    self.__HeaderMatrix.append((feature)+"-"+str(i+1))
            if (self.__include_label):
                self.__HeaderMatrix.append('Label')

    def __GenerateCSVFile(self, dataset):
            with open(self.__Path, 'w', newline="") as csvfile:
                filewriter = csv.writer(csvfile)
                filewriter.writerows(dataset)

    def __PrepareDataMatrix(self):
            self.__GenerateDataMatrix()
            self.__GenerateHeaderMatrix()
            self.__DataMatrix.insert(0, self.__HeaderMatrix)
            return np.array(self.__DataMatrix, dtype=object)

    def DataBeforeKmeans(self):
        self.__GenerateDataMatrix()
        df = pd.DataFrame(self.__DataMatrix)
        df  = df.sample(frac = FEATURES/len(self.__DataMatrix))
        self.__DataMatrix = df.values.tolist()
        return np.array(self.__DataMatrix, dtype=float)
    
    @staticmethod
    def generateCSVFileFromCodeBook(codebook, files_list, csv_file_name, address_path = "datasets"):
        import os
        csv_file_path = os.path.join(getcwd(), address_path, csv_file_name)
        grand_codebook = []
        hog_list = codebook[0].tolist()
        mbhx_list = codebook[1].tolist()
        mbhy_list = codebook[2].tolist()
        for i in range(len(files_list)):
            label = files_list[i].split("-")[1]+files_list[i].split("-")[2]
            grand_codebook.append(hog_list[i]+mbhx_list[i]+mbhy_list[i])
            grand_codebook[i].append(label)
        header_matrix = []
        for feature_name in csv_file_names:
            feature_name = feature_name.split(".csv")[0]
            for i in range(int((len(grand_codebook[0])-1)/3)):
                header_matrix.append(feature_name+"_"+str(i))
        header_matrix.append("Label")
        grand_codebook.insert(0, header_matrix)

        grand_codebook = np.array(grand_codebook, dtype=object)
        with open(csv_file_path, 'w', newline="") as csvfile:
                filewriter = csv.writer(csvfile)
                filewriter.writerows(grand_codebook)
        print("Dataset has been saved as",csv_file_path)
            
            
    def start(self):
            self.__GenerateCSVFile(self.__PrepareDataMatrix())
