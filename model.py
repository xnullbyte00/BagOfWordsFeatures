import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
from time import sleep



class machine_learning:

    def __init__(self, data_dir, model_type, trained_model_name, result_file_name):
        self.data_dir = data_dir
        self.model_type = model_type
        self.trained_model_name = trained_model_name
        self.result_file_name = result_file_name
        
    def train_model(self):
        model = None
        #Splitting into Data Matrix and Label Classes Vectors
        X = pd.read_csv(self.data_dir)
        y = X['Labels']
        X = X.drop('Labels',axis=1)
        X = X.drop('Label',axis=1)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = pd.DataFrame(X)

        #Spliting the Dataset into train and validation
        X_train, X_test, y_train, y_test =(train_test_split(X, y, train_size=0.70, test_size=0.30))
        print("Dataset has been splitted into 70% Training and 30% Validation")

        if (self.model_type == 'svm'):
            model = SVC(kernel='linear', probability = True)
            print("SVM Model is being trained")
            
        elif (self.model_type == 'knn'):
            model = KNeighborsClassifier(n_neighbors=3)
            print("KNN Model is being trained")

        elif (self.model_type == 'ann'):
            model = MLPClassifier()
            print("ANN Model is being trained")

        else:
            Exception("No appropriate model name is selected")

        print("model type", self.model_type)
        print("model object", model)
    

        cls =model.fit(X_train,y_train)
        val_predictions=cls.predict(X_test)

        print("The confusion matrix of validation dataset:\n\n")
        print(confusion_matrix(y_test,val_predictions))

        print("The classification report of validation dataset:\n\n")
        print(classification_report(y_test,val_predictions))


        with open(self.trained_model_name,'wb') as f:
            pickle.dump(model,f)

        print('The model has been saved with with following path')
        print(self.trained_model_name)

        input("Press enter key to continue")

    def test_model(self):
        try:
            with open(self.trained_model_name,'rb') as f:
                mp=pickle.load(f)
        
                test_data = pd.read_csv(self.data_dir)
                y = test_data['Label']
                class_names = np.array(list(set(y)))
                test_data = test_data.drop('Label',axis=1)

                predictions = mp.predict(test_data)
                class_probabilities = mp.predict_proba(test_data)

            print("The confusion matrix of test dataset:\n\n")
            print(confusion_matrix(y,predictions))
            
            
                # Plot non-normalized confusion matrix
            titles_options = "Confusion matrix, without normalization"
            
            if (self.model_type == 'svm'):
                color = plt.cm.Reds
            elif (self.model_type == 'ann'):
                color = plt.cm.Purples
            elif (self.model_type == 'knn'):
                color = plt.cm.Greens
            else:
                raise Exception("Model Name is not right")
            
            #for title, normalize in titles_options:
            disp = plot_confusion_matrix(mp, test_data, y,
                                        display_labels=class_names,
                                        cmap=color
                                        )
            disp.ax_.set_title(titles_options)
            
            print(titles_options)
            print(disp.confusion_matrix)
            plt.xticks(rotation=90)
            #plt.tight_layout()
            plt.savefig(self.result_file_name)
            


            print("The classification report of test dataset:\n\n")
            print(classification_report(y,predictions))
        
            #  print(min(class_probabilities))
            #  print(max(class_probabilities))
            min_max = []
            print('\n\n\n The class probabilities are given below')
            print('Class Label','Predicted Label','Probabilities')
            for i in range(len(class_probabilities)):
                #print(y[i], '  ', predictions[i],'    ----',np.round(max(class_probabilities[i])*100,2),'%')
                min_max.append(np.round(max(class_probabilities[i])*100,2))
            print("maximum confidence score is: ", max(min_max))
            print("minimum confidence score is:", min(min_max))
            print("Average confidence score is:", sum(min_max)/len(min_max))

            plt.show()
            input("Press enter key to continue")
        except:
            print("\n\n Sorry! Request model does not exist. You have to train it first.\n\n")
            input("Press enter key to continue")

def main():

    
    os.system('cls')
    print("=====================================================================")
    print("         MODEL TRAINING AND TESTING ON PREPROCESSED DATA             ")
    print("=====================================================================")

    PROCESS_TYPE = 'train' #Just to remove unbound warning
    MODEL_TYPE = 'SVM' #Just to remove unbound warning

    PROCESS_TYPE_SELECTION = 3
    MODEL_TYPE_SELECTION = 4

    while (PROCESS_TYPE_SELECTION > 2):
        PROCESS_TYPE_SELECTION = int(input('\n\nWhich process you would like to do?\n 1. Train a new model\n 2. Test the existing model\n Your Choice: '))
        if (PROCESS_TYPE_SELECTION == 1):
            PROCESS_TYPE = 'train'
            break
        elif (PROCESS_TYPE_SELECTION == 2):
            PROCESS_TYPE = 'test'
            break
        else:
            print("\n\nNo appropriate selection has been made\n\n")



    while (MODEL_TYPE_SELECTION > 3):
            MODEL_TYPE_SELECTION = int(input('\n\nWhich model you would like to train or test?\n 1. SVM\n 2. KNN\n 3. ANN\n Your Choice: '))

            if (MODEL_TYPE_SELECTION == 1):
                MODEL_TYPE = 'svm'
                break
            elif (MODEL_TYPE_SELECTION == 2):
                MODEL_TYPE = 'knn'
                break
            elif (MODEL_TYPE_SELECTION == 3):
                MODEL_TYPE = 'ann'
                break
            else:
                print("\n\nNo appropriate selection has been made\n\n")

  
    CSV_FILE_NAME = os.path.join(os.getcwd(), "datasets", "dataset.csv")
    MODEL_FILE_NAME =os.path.join(os.getcwd(), "models", MODEL_TYPE)
    RESULT_FILE_NAME = os.path.join(os.getcwd(), "results", MODEL_TYPE+'.jpg')



    if (PROCESS_TYPE == 'train'):
        machine_learning(CSV_FILE_NAME, MODEL_TYPE, MODEL_FILE_NAME, RESULT_FILE_NAME ).train_model()
    else:
        machine_learning(CSV_FILE_NAME, MODEL_TYPE, MODEL_FILE_NAME, RESULT_FILE_NAME).test_model()


