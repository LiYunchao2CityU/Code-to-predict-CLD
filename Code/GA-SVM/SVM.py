# Author@ Liu
# Time：2022/5/6 13:55
import numpy as np
import pandas as pd
from ThesisModel import dataprocessing as dp
from sklearn.svm import SVR
from ThesisModel import subsequent_treatment as st

if __name__ == '__main__':
    number = 20
    train_ev = 0;train_r2 = 0;train_mse = 0;train_mae = 0
    test_ev = 0;test_r2 = 0;test_mse = 0;test_mae = 0
    unseen_ev = 0;unseen_r2 = 0;unseen_mse = 0;unseen_mae = 0
    lengthTrain = dp.trainLength;lengthTest = dp.testLength;lengthUnseen = dp.unseenLength
    trainTrue = np.zeros(lengthTrain);trainPrediction= np.zeros(lengthTrain)
    testTrue = np.zeros(lengthTest);testPrediction= np.zeros(lengthTest)
    unseenTrue = np.zeros(lengthUnseen);unseenPrediction = np.zeros(lengthUnseen)
    for i in range(number):
        dataSet = dp.ReadData()
        features, target = dp.SplitFeatureAndTarget(dataSet)
        features_DataDistribution = dp.DataDistribution(features)
        new_features, new_target = dp.Drop_noisy(features_DataDistribution, target)
        x_train, x_test, y_train, y_test = dp.SplitTestAndTrain(new_features, new_target)
        x_train_MinMax, x_test_MinMax, y_train_MinMax, y_test_MinMax = dp.Normalization(x_train, x_test, y_train, y_test)
        unseen_features,unseen_target,x_train_new,y_train_new = dp.CreateUnseenData(x_train_MinMax, y_train_MinMax)
        regression = SVR(C=0.1,kernel='rbf',gamma=2).fit(x_train_new,y_train_new)
        y_train_prediction = regression.predict(x_train_new)
        train_evScore,train_r2Score,train_mseScore,train_maeScore = st.Metrics(y_train_new,y_train_prediction)
        train_ev = train_evScore + train_ev; train_r2 = train_r2Score + train_r2; train_mse = train_mseScore + train_mse
        train_mae = train_maeScore + train_mae
        trainTrue = y_train_new + trainTrue;trainPrediction = y_train_prediction + trainPrediction


        y_test_prediction = regression.predict(x_test_MinMax)
        test_evScore,test_r2Score,test_mseScore,test_maeScore= st.Metrics(y_test_MinMax,y_test_prediction)
        test_ev = test_evScore + test_ev; test_r2 = test_r2Score + test_r2; test_mse = test_mseScore + test_mse
        test_mae = test_maeScore + test_mae
        testTrue = y_test_MinMax.flatten() + testTrue;testPrediction = y_test_prediction + testPrediction

        y_unseen_prediction = regression.predict(unseen_features)
        unseen_evScore,unseen_r2Score,unseen_mseScore,unseen_maeScore= st.Metrics(unseen_target,y_unseen_prediction)
        unseen_ev = unseen_evScore + unseen_ev; unseen_r2 = unseen_r2Score + unseen_r2; unseen_mse = unseen_mseScore + unseen_mse
        unseen_mae = unseen_maeScore + unseen_mae
        unseenTrue = unseen_target + unseenTrue;unseenPrediction = y_unseen_prediction + unseenPrediction
    print("--------------------------------------Train Set--------------------------------------")
    print("Expected variance of training set：", '%.2f%%' % (train_ev*5))
    print("R2 Score of training set：", '%.2f%%' % (train_r2 * 5))
    print("MSE Score of training set：", '%.2f%%' % (train_mse * 5))
    print("MAE Score of training set：", '%.2f%%' % (train_mae * 5))
    print("--------------------------------------Test Set--------------------------------------")
    print("Expected variance of testing set：", '%.2f%%' % (test_ev*5))
    print("R2 Score of testing set：", '%.2f%%' % (test_r2 * 5))
    print("MSE Score of testing set：", '%.2f%%' % (test_mse * 5))
    print("MAE Score of testing set：", '%.2f%%' % (test_mae * 5))
    print("--------------------------------------Unseen Set--------------------------------------")
    print("Expected variance of unseen set：", '%.2f%%' % (unseen_ev*5))
    print("R2 Score of unseen set：", '%.2f%%' % (unseen_r2 * 5))
    print("MSE Score of unseen set：", '%.2f%%' % (unseen_mse * 5))
    print("MAE Score of unseen set：", '%.2f%%' % (unseen_mae * 5))

    trainTrue,trainPrediction = st.Transpose(trainTrue,trainPrediction)
    trainResult = pd.DataFrame(data=[trainTrue.tolist(),trainPrediction.tolist()]).transpose()
    trainResult.columns = ["True","Prediction"]
    testTrue,testPrediction = st.Transpose(testTrue,testPrediction)
    testResult = pd.DataFrame(data=[testTrue.tolist(),testPrediction.tolist()]).transpose()
    testResult.columns = ["True","Prediction"]
    unseenTrue,unseenPrediction = st.Transpose(unseenTrue,unseenPrediction)
    unseenResult = pd.DataFrame(data=[unseenTrue.tolist(),unseenPrediction.tolist()]).transpose()
    unseenResult.columns = ["True","Prediction"]
    sheetNameList = ["TrainData","TestData","UnseenData"]
    dataSetList = [trainResult,testResult,unseenResult]
    excelFilePath = "D:/pythonproject/ThesisModel/GA-SVM/result/SVMResultData.xlsx"
    st.GenerateExcel(excelFilePath)
    for i in range(len(sheetNameList)):
        st.WriteExcel(dataSetList[i],sheetNameList[i],excelFilePath)
    print("--------------------------------------Finish!!!--------------------------------------")












