# Author@ Liu
# Time：2022/5/5 22:21
import pandas as pd
import numpy as np
import random
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""
    train_set = 1337
    test_set = 410
    unseen_set = 300
"""

def ReadData():
    """
        1. Read data
    """
    filePath = "D:/pythonproject/ThesisModel/Data/dataSample.xlsx"
    dataSet = pd.read_excel(filePath)
    dataSet.drop(dataSet[dataSet['counts'] < 1].index,inplace=True)
    return dataSet

def SplitFeatureAndTarget(dataSet):
    """
        2. Split Feature And Target
    """
    rows = dataSet.shape[0]
    columns = dataSet.shape[1]
    dataSetArray = np.array(dataSet)
    features = dataSetArray[:,0:columns - 1]
    target = dataSetArray[:,columns - 1].reshape((dataSetArray.shape[0],1))
    return features,target

def DataDistribution(features):
    """
        3. Judge whether the data is normal distribution
    """
    transform = MinMaxScaler(feature_range=(1, 3), copy=True)
    features_DataDistribution = transform.fit_transform(features)
    result1 = []
    result2 = []
    for i in range(features_DataDistribution.shape[1]):
        if stats.kstest(features_DataDistribution[:,i],'norm',(features_DataDistribution[:,i].mean(),features_DataDistribution[:, i].std()))[1] > 0.05:
            print("Characteristic {0} satisfies normal distribution".format(i+1))
            result1.insert(i,i+1)
        else:
            print("Characteristic {0} does not satisfy normal distribution".format(i+1))
            result2.insert(i,i+1)
    print("Satisfy normal distribution:",result1)
    print("Not satisfying normal distribution:",result2)
    for i in range(len(result2)):
        features_DataDistribution[:, result2[i] - 1] = stats.boxcox(features_DataDistribution[:, result2[i] - 1]
                                                                    ,stats.boxcox_normmax(features_DataDistribution[:, result2[i] - 1] + 1))
    return features_DataDistribution

def Drop_noisy(features_DataDistribution,target):
    """
        4. Noise removal
    """
    features_Drop_noisy = pd.DataFrame(features_DataDistribution,)
    new_features = features_Drop_noisy.copy()
    features_describe = new_features.describe()
    for column in features_Drop_noisy.columns:
        mean = features_describe.loc['mean', column]
        std = features_describe.loc['std', column]
        minvalue = mean - 3 * std
        maxvalue = mean + 3 * std
        features_Drop_noisy = features_Drop_noisy[features_Drop_noisy[column] >= minvalue]
        features_Drop_noisy = features_Drop_noisy[features_Drop_noisy[column] <= maxvalue]

    user_index = list(features_Drop_noisy.index.values)
    number_copy = np.zeros(len(user_index))
    for i in range(len(user_index)):
        number_copy[i] = user_index[i]

    new_target = np.zeros(number_copy.shape[0])
    for i in range(new_target.shape[0]):
        new_target[i] = target[int(number_copy[i])]
    return features_Drop_noisy,new_target

def SplitTestAndTrain(features_Drop_noisy,new_target):
    """
        5. Divide the data set to complete cross validation
    """
    transform = MinMaxScaler(feature_range=(1, 10), copy=True)
    features_STT = transform.fit_transform(features_Drop_noisy)
    for i in range(features_STT.shape[0]):
        for j in range(features_STT.shape[1]):
            features_STT[i,j] = math.log10(features_STT[i,j])
    x_train,x_test,y_train,y_test = train_test_split(features_STT,new_target,shuffle=True,test_size=0.2,random_state=32)
    return x_train,x_test,y_train,y_test

def Normalization(x_train,x_test,y_train,y_test):
    """
        6. Data normalization
    """
    transform = MinMaxScaler()
    x_train_MM = transform.fit_transform(x_train)
    x_test_MM = transform.fit_transform(x_test)
    y_train_MM = transform.fit_transform(y_train.reshape((y_train.shape[0],1)))
    y_test_MM = transform.fit_transform(y_test.reshape((y_test.shape[0],1)))
    return x_train_MM,x_test_MM,y_train_MM,y_test_MM

def CreateUnseenData(x_train,y_train):
    """
        7. Create UnseenData
    """
    sampleNumber = 300
    unseen_features = np.zeros((sampleNumber,x_train.shape[1]))
    unseen_target = np.zeros(sampleNumber)
    indexList = []
    nums = list(range(0, len(x_train)))
    for i in range(sampleNumber):
        num = random.choice(nums)
        nums.remove(num)
        indexList.append(num)
        unseen_features[i,:] = x_train[indexList[i],:]
        unseen_target[i] = y_train[indexList[i]]
    x_train = np.delete(x_train, indexList,axis=0)
    y_train = np.delete(y_train, indexList)
    return unseen_features,unseen_target,x_train,y_train


trainLength = 1337
testLength = 410
unseenLength = 300


















