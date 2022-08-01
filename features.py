from configurations import *

class ExtractFeatures:
    def __init__(self, data_features_list, kmean_clusters = 256, standardize = True):
        self.__dataFeaturesList = data_features_list
        self.__descriptors = None
        self.__KmeanClusters = kmean_clusters
        self.__NumberOfSamples = len(data_features_list)
        self.__computedCodeBook = None
        self.__standardize = standardize


    def __getStackedMatrix(self):
        self.__descriptors = self.__dataFeaturesList[0]
        for descriptor in self.__dataFeaturesList:
            self.__descriptors = np.vstack((self.__descriptors, descriptor))
        self.__descriptors = self.__descriptors.astype(float)
    
    def __generateCodeBook(self):
        voc, variance = kmeans(self.__descriptors, self.__KmeanClusters, 1)
        self.__computedCodeBook = np.zeros((self.__NumberOfSamples, self.__KmeanClusters), "float32")
        for i in range(self.__NumberOfSamples):
            words, distance = vq(self.__dataFeaturesList[i],voc)
            for w in words:
                self.__computedCodeBook[i][w] += 1
    
    def __standardizeCodeBook(self):
        stdSlr = StandardScaler().fit(self.__computedCodeBook)
        self.__computedCodeBook = stdSlr.transform(self.__computedCodeBook)

    @staticmethod
    def getspecificFeatureMatrix(data_matrix):
        hog_features_list  = []
        mbhx_features_list  = []
        mbhy_features_list  = []
        for data_array in data_matrix:
            data_array = np.transpose(data_array)
            hog_features_list.append(np.transpose(data_array[0*96:1*96]))
            mbhx_features_list.append(np.transpose(data_array[1*96:2*96]))
            mbhy_features_list.append(np.transpose(data_array[2*96:3*96]))
        return [hog_features_list, mbhx_features_list, mbhy_features_list]
    
    def getFinalCodeBook(self):
        self.__getStackedMatrix()
        self.__generateCodeBook()
        if (self.__standardize):
            self.__standardizeCodeBook()
        return self.__computedCodeBook


