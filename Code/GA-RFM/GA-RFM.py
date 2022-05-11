# Author@ Liu
# Time：2022/5/7 10:18
import numpy as np
import geatpy as ga
import pandas as pd
from ThesisModel import dataprocessing as dp
from ThesisModel import subsequent_treatment as st
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.ensemble import RandomForestRegressor
from joblib import dump,load
from sklearn.model_selection import cross_val_score

class MyProblem(ga.Problem):
    def __init__(self,PoolType):
        name = 'MyProblem'
        M = 1
        maxormins = [-1]
        Dim = 2
        varTypes = [1, 1]
        lb = [1, 1]
        ub = [100, 3]
        lbin = [0] * Dim
        ubin = [1] * Dim
        ga.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        dataSet = dp.ReadData()
        features, target = dp.SplitFeatureAndTarget(dataSet)
        features_DataDistribution = dp.DataDistribution(features)
        new_features, new_target = dp.Drop_noisy(features_DataDistribution, target)
        x_train, x_test, y_train, y_test = dp.SplitTestAndTrain(new_features, new_target)
        x_train, x_test, y_train, y_test = dp.Normalization(x_train, x_test, y_train, y_test)
        unseen_features, unseen_target, x_train_new, y_train_new = dp.CreateUnseenData(x_train, y_train)
        self.x_train = x_train_new
        self.y_train = y_train_new
        self.x_test = x_test
        self.y_test = y_test
        self.unseen_features = unseen_features
        self.unseen_target = unseen_target
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())
            self.pool = ProcessPool(num_cores)

    def evalVars(self, Vars):
        N = Vars.shape[0]
        args = list(zip(list(range(N)), [Vars] * N, [self.x_train] * N, [self.y_train] * N))
        if self.PoolType == 'Thread':
            f = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            f = np.array(result.get())
        return f
    def SaveModel(self,n_estimators,min_samples_split):
        regression = RandomForestRegressor(n_estimators=n_estimators, criterion='squared_error',
                                           max_features='auto',
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=1, max_leaf_nodes=None, min_weight_fraction_leaf=0.0,
                                           random_state=32).fit(self.x_train, self.y_train)
        dump(regression,"D:/pythonproject/ThesisModel/GA-RFM/result/GA-RFMModel.joblib")
    def UnseenTest(self):
        model = load("D:/pythonproject/ThesisModel/GA-RFM/result/GA-RFMModel.joblib")
        prediction = model.predict(self.unseen_features)
        unseen_evScore, unseen_r2Score, unseen_mseScore, unseen_maeScore = st.Metrics(self.unseen_target,prediction)
        print("--------------------------------------Unseen Set--------------------------------------")
        print("Expected variance of unseen set：", '%.2f%%' % (unseen_evScore * 100))
        print("R2 Score of unseen set：", '%.2f%%' % (unseen_r2Score * 100))
        print("MSE Score of unseen set：", '%.2f%%' % (unseen_mseScore * 100))
        print("MAE Score of unseen set：", '%.2f%%' % (unseen_maeScore * 100))
        unseen_result = pd.DataFrame(data=self.unseen_features)
        unseen_result.insert(3, column="Ture", value=self.unseen_target)
        unseen_result.insert(4, column="Prediction", value=prediction)
        unseen_result.columns = ["cut","rpm","CL","Ture","Prediction"]
        unseen_result.to_excel("D:/pythonproject/ThesisModel/GA-RFM/result/UnseenResultData.xlsx",index=False)

    def test(self, n_estimators, min_samples_split):
        number = 20
        train_ev = 0;train_r2 = 0;train_mse = 0;train_mae = 0
        test_ev = 0;test_r2 = 0;test_mse = 0;test_mae = 0
        unseen_ev = 0;unseen_r2 = 0;unseen_mse = 0;unseen_mae = 0
        lengthTrain = dp.trainLength;lengthTest = dp.testLength;lengthUnseen = dp.unseenLength
        trainTrue = np.zeros(lengthTrain);trainPrediction = np.zeros(lengthTrain)
        testTrue = np.zeros(lengthTest);testPrediction = np.zeros(lengthTest)
        unseenTrue = np.zeros(lengthUnseen);unseenPrediction = np.zeros(lengthUnseen)
        for i in range(number):
            regression = RandomForestRegressor(n_estimators=n_estimators, criterion='squared_error',
                                               max_features='auto',
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=1, max_leaf_nodes=None, min_weight_fraction_leaf=0.0,
                                               random_state=32).fit(self.x_train, self.y_train)
            y_train_prediction = regression.predict(self.x_train)
            train_evScore, train_r2Score, train_mseScore, train_maeScore = st.Metrics(self.y_train, y_train_prediction)
            train_ev = train_evScore + train_ev;train_r2 = train_r2Score + train_r2;train_mse = train_mseScore + train_mse
            train_mae = train_maeScore + train_mae
            trainTrue = self.y_train + trainTrue;trainPrediction = y_train_prediction + trainPrediction

            y_test_prediction = regression.predict(self.x_test)
            test_evScore,test_r2Score,test_mseScore,test_maeScore = st.Metrics(self.y_test,y_test_prediction)
            test_ev = test_evScore + test_ev;test_r2 = test_r2Score + test_r2;test_mse = test_mseScore + test_mse
            test_mae = test_maeScore + test_mae
            testTrue = self.y_test.flatten() + testTrue;testPrediction = y_test_prediction + testPrediction

            unseen_prediction = regression.predict(self.unseen_features)
            unseen_evScore, unseen_r2Score, unseen_mseScore, unseen_maeScore = st.Metrics(self.unseen_target,
                                                                                          unseen_prediction)
            unseen_ev = unseen_evScore + unseen_ev;unseen_r2 = unseen_r2Score + unseen_r2;
            unseen_mse = unseen_mseScore + unseen_mse;unseen_mae = unseen_maeScore + unseen_mae
            unseenTrue = self.unseen_target + unseenTrue;unseenPrediction = unseen_prediction + unseenPrediction
        print("--------------------------------------Train Set--------------------------------------")
        print("Expected variance of training set：", '%.2f%%' % (train_ev * 5))
        print("R2 Score of training set：", '%.2f%%' % (train_r2 * 5))
        print("MSE Score of training set：", '%.2f%%' % (train_mse * 5))
        print("MAE Score of training set：", '%.2f%%' % (train_mae * 5))
        print("--------------------------------------Test Set--------------------------------------")
        print("Expected variance of testing set：", '%.2f%%' % (test_ev * 5))
        print("R2 Score of testing set：", '%.2f%%' % (test_r2 * 5))
        print("MSE Score of testing set：", '%.2f%%' % (test_mse * 5))
        print("MAE Score of testing set：", '%.2f%%' % (test_mae * 5))
        print("--------------------------------------Unseen Set--------------------------------------")
        print("Expected variance of unseen set：", '%.2f%%' % (unseen_ev * 5))
        print("R2 Score of unseen set：", '%.2f%%' % (unseen_r2 * 5))
        print("MSE Score of unseen set：", '%.2f%%' % (unseen_mse * 5))
        print("MAE Score of unseen set：", '%.2f%%' % (unseen_mae * 5))

        trainTrue, trainPrediction = st.Transpose(trainTrue, trainPrediction)
        trainResult = pd.DataFrame(data=[trainTrue.tolist(), trainPrediction.tolist()]).transpose()
        trainResult.columns = ["True", "Prediction"]
        testTrue, testPrediction = st.Transpose(testTrue, testPrediction)
        testResult = pd.DataFrame(data=[testTrue.tolist(), testPrediction.tolist()]).transpose()
        testResult.columns = ["True", "Prediction"]
        unseenTrue, unseenPrediction = st.Transpose(unseenTrue, unseenPrediction)
        unseenResult = pd.DataFrame(data=[unseenTrue.tolist(), unseenPrediction.tolist()]).transpose()
        unseenResult.columns = ["True", "Prediction"]
        sheetNameList = ["TrainData","TestData", "UnseenData"]
        dataSetList = [trainResult,testResult,unseenResult]
        excelFilePath = "D:/pythonproject/ThesisModel/GA-RFM/result/GA-RFMResultData.xlsx"
        st.GenerateExcel(excelFilePath)
        for i in range(len(sheetNameList)):
            st.WriteExcel(dataSetList[i], sheetNameList[i], excelFilePath)
        print("--------------------------------------Finish!!!--------------------------------------")

def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    x_train = args[2]
    y_train = args[3]
    n_estimators = Vars[i, 0]
    min_samples_split = Vars[i, 1]
    regression = RandomForestRegressor(n_estimators=n_estimators, criterion='squared_error', max_features='auto',
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=1, max_leaf_nodes=None, min_weight_fraction_leaf=0.0,
                                       random_state=32).fit(x_train, y_train)
    scores = cross_val_score(regression, x_train, y_train, cv=20)
    ObjV_i = [scores.mean()]
    return ObjV_i


if __name__ == '__main__':
    problem = MyProblem(PoolType='Thread')
    algorithm = ga.soea_DE_rand_1_bin_templet(problem,
                                              ga.Population(Encoding='RI', NIND=50),
                                              MAXGEN=30,
                                              logTras=1,
                                              trappedValue=1e-6,
                                              maxTrappedCount=10)
    res = ga.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    if res['success']:
        problem.SaveModel(n_estimators=res['Vars'][0, 0], min_samples_split=res['Vars'][0, 1])
        problem.UnseenTest()
        problem.test(n_estimators=res['Vars'][0, 0], min_samples_split=res['Vars'][0, 1])