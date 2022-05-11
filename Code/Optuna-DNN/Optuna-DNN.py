# Author@ Liu
# Time：2022/5/8 16:47
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from ThesisModel import dataprocessing as dp
from ThesisModel import subsequent_treatment as st
from tensorflow.keras.regularizers import L1,L2
from tensorflow.keras.initializers import RandomNormal,Zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,GaussianDropout
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState



def CreateModel(trial):
    unit1 = trial.suggest_categorical("unit1", [1,2,3,4,5,6,7,8])
    w1_mean = trial.suggest_float("w1_mean", 0.0001, 1, log=True)
    w1_stddev = trial.suggest_float("w1_stddev", 0.0001, 1, log=True)
    L1_1 = trial.suggest_float("L1_1", 0.0001, 1, log=True)
    L2_1 = trial.suggest_float("L2_1", 0.0001, 1, log=True)
    activation1 = trial.suggest_categorical("activation1",['relu','selu','tanh','linear'])
    dropout = trial.suggest_float('dropout',0.0001, 1,log=True)
    unit2 = trial.suggest_categorical("unit2", [1,2,3,4,5,6,7,8])
    w2_mean = trial.suggest_float("w2_mean", 0.0001, 1, log=True)
    w2_stddev = trial.suggest_float("w2_stddev", 0.0001, 1, log=True)
    activation2 = trial.suggest_categorical("activation2", ['relu', 'selu', 'tanh', 'linear'])
    gaussian = trial.suggest_float('gaussian', 0.0001, 1, log=True)
    unit3 = trial.suggest_categorical("unit3", [1,2,3,4])
    w3_mean = trial.suggest_float("w3_mean", 0.0001, 1, log=True)
    w3_stddev = trial.suggest_float("w3_stddev", 0.0001, 1, log=True)
    activation3 = trial.suggest_categorical("activation3", ['relu', 'selu', 'tanh', 'linear'])
    w4_mean = trial.suggest_float("w4_mean", 0.0001, 1, log=True)
    w4_stddev = trial.suggest_float("w4_stddev", 0.0001, 1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)

    b = Zeros()
    model = Sequential()
    model.add(Dense(units=unit1, input_dim=3, kernel_initializer=RandomNormal(mean=w1_mean,stddev=w1_stddev,seed=None)
                    ,bias_initializer=b,kernel_regularizer=L1(L1_1),activity_regularizer=L2(L2_1),activation=activation1))
    model.add(Dropout(dropout))
    model.add(Dense(units=unit2, kernel_initializer=RandomNormal(mean=w2_mean,stddev=w2_stddev,seed=None),
                    bias_initializer=b,activation=activation2))
    model.add(GaussianDropout(gaussian))
    model.add(Dense(units=unit3, kernel_initializer=RandomNormal(mean=w3_mean,stddev=w3_stddev,seed=None),
                    bias_initializer=b,activation=activation3))
    model.add(Dense(units=1, kernel_initializer=RandomNormal(mean=w4_mean,stddev=w4_stddev,seed=None),
                    bias_initializer=b,activation=None))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=["mae"])
    return model

def Objective(trial):
    tf.keras.backend.clear_session()
    monitor = "mae"
    model = CreateModel(trial)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        TFKerasPruningCallback(trial, monitor)
    ]
    model.fit(x_train_new, y_train_new, epochs=3000, batch_size=64,shuffle=True, verbose=True
              , validation_split=0.2,validation_freq=3,callbacks=callbacks)
    y_train_prediction = model.predict(x_train_new)
    evScore,r2Score,mseScore,maeScore = st.Metrics(y_train_new,y_train_prediction)
    print("Expected variance of training set：", '%.2f%%' % (evScore * 100))
    print("R2 Score of training set：", '%.2f%%' % (r2Score * 100))
    print("MSE Score of training set：", '%.2f%%' % (mseScore * 100))
    print("MAE Score of training set：", '%.2f%%' % (maeScore * 100))
    objection = evScore
    return objection

def show_result(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value:", trial.value)
    print("Params:")
    for key, value in trial.params.items():
        print("{}:{}".format(key, value))

def DNN(study,x_train_new, y_train_new,x_test_MinMax,unseen_features):
    param = study.best_params
    b = Zeros()
    model = Sequential()
    model.add(Dense(units=param["unit1"], input_dim=3,
                    kernel_initializer=RandomNormal(mean=param["w1_mean"],stddev=param["w1_stddev"],seed=None)
                    ,bias_initializer=b,kernel_regularizer=L1(param["L1_1"]),activity_regularizer=L2(param["L2_1"]),
                    activation=param["activation1"]))
    model.add(Dropout(param["dropout"]))
    model.add(Dense(units=param["unit2"],
                    kernel_initializer=RandomNormal(mean=param["w2_stddev"],stddev=param["w2_stddev"],seed=None),
                    bias_initializer=b,activation=param["activation2"]))
    model.add(GaussianDropout(param["gaussian"]))
    model.add(Dense(units=param["unit3"],
                    kernel_initializer=RandomNormal(mean=param["w3_stddev"],stddev=param["w3_stddev"],seed=None),
                    bias_initializer=b,activation=param["activation2"]))
    model.add(Dense(units=1, kernel_initializer=RandomNormal(mean=param["w4_stddev"],stddev=param["w4_stddev"],seed=None),
                    bias_initializer=b,activation=None))
    model.compile(loss='mse', optimizer=Adam(learning_rate=param["learning_rate"]), metrics=["mae"])
    model.fit(x_train_new, y_train_new, epochs=3000, batch_size=64,shuffle=True, verbose=True
              , validation_split=0.2,validation_freq=3)
    y_test_prediction = model.predict(x_test_MinMax)
    unseen_target_prediction = model.predict(unseen_features)
    return y_test_prediction,unseen_target_prediction

def SaveModel(study,x_train_new, y_train_new):
    param = study.best_params
    b = Zeros()
    model = Sequential()
    model.add(Dense(units=param["unit1"], input_dim=3,
                    kernel_initializer=RandomNormal(mean=param["w1_mean"],stddev=param["w1_stddev"],seed=None)
                    ,bias_initializer=b,kernel_regularizer=L1(param["L1_1"]),activity_regularizer=L2(param["L2_1"]),
                    activation=param["activation1"]))
    model.add(Dropout(param["dropout"]))
    model.add(Dense(units=param["unit2"],
                    kernel_initializer=RandomNormal(mean=param["w2_stddev"],stddev=param["w2_stddev"],seed=None),
                    bias_initializer=b,activation=param["activation2"]))
    model.add(GaussianDropout(param["gaussian"]))
    model.add(Dense(units=param["unit3"],
                    kernel_initializer=RandomNormal(mean=param["w3_stddev"],stddev=param["w3_stddev"],seed=None),
                    bias_initializer=b,activation=param["activation2"]))
    model.add(Dense(units=1, kernel_initializer=RandomNormal(mean=param["w4_stddev"],stddev=param["w4_stddev"],seed=None),
                    bias_initializer=b,activation=None))
    model.compile(loss='mse', optimizer=Adam(learning_rate=param["learning_rate"]), metrics=["mae"])
    model.fit(x_train_new, y_train_new, epochs=3000, batch_size=64,shuffle=True, verbose=True
              , validation_split=0.2,validation_freq=3)
    prediction = model.predict(unseen_features)
    unseen_evScore, unseen_r2Score, unseen_mseScore, unseen_maeScore = st.Metrics(unseen_target, prediction)
    unseen_result = pd.DataFrame(data=unseen_features)
    unseen_result.insert(3, column="Ture", value=unseen_target)
    unseen_result.insert(4, column="Prediction", value=prediction)
    unseen_result.columns = ["cut", "rpm", "CL", "Ture", "Prediction"]
    unseen_result.to_excel("D:/pythonproject/ThesisModel/Optuna-DNN/result/UnseenResultData.xlsx", index=False)
    model.save("D:/pythonproject/ThesisModel/Optuna-DNN/result/Optuna-DNNModel.h5")
    print("--------------------------------------Unseen Set--------------------------------------")
    print("Expected variance of unseen set：", '%.2f%%' % (unseen_evScore * 100))
    print("R2 Score of unseen set：", '%.2f%%' % (unseen_r2Score * 100))
    print("MSE Score of unseen set：", '%.2f%%' % (unseen_mseScore * 100))
    print("MAE Score of unseen set：", '%.2f%%' % (unseen_maeScore * 100))
    return model




if __name__ == '__main__':
    dataSet = dp.ReadData()
    features, target = dp.SplitFeatureAndTarget(dataSet)
    features_DataDistribution = dp.DataDistribution(features)
    new_features, new_target = dp.Drop_noisy(features_DataDistribution, target)
    x_train, x_test, y_train, y_test = dp.SplitTestAndTrain(new_features, new_target)
    x_train_MinMax, x_test_MinMax, y_train_MinMax, y_test_MinMax = dp.Normalization(x_train, x_test, y_train, y_test)
    unseen_features, unseen_target, x_train_new, y_train_new = dp.CreateUnseenData(x_train_MinMax, y_train_MinMax)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
    study.optimize(Objective, n_trials=25, timeout=600)
    show_result(study)

    number = 20
    test_ev = 0;test_r2 = 0;test_mse = 0;test_mae = 0
    unseen_ev = 0;unseen_r2 = 0;unseen_mse = 0;unseen_mae = 0
    lengthTest = dp.testLength;lengthUnseen = dp.unseenLength
    testTrue = np.zeros(lengthTest);testPrediction = np.zeros(lengthTest)
    unseenTrue = np.zeros(lengthUnseen);unseenPrediction = np.zeros(lengthUnseen)

    for i in range(number):
        y_test_prediction,unseen_target_prediction = DNN(study,x_train_new, y_train_new,x_test_MinMax,unseen_features)
        test_evScore, test_r2Score, test_mseScore, test_maeScore = st.Metrics(y_test_MinMax, y_test_prediction)
        test_ev = test_evScore + test_ev;test_r2 = test_r2Score + test_r2;test_mse = test_mseScore + test_mse
        test_mae = test_maeScore + test_mae
        testTrue = y_test_MinMax.flatten() + testTrue;testPrediction = y_test_prediction.flatten() + testPrediction

        unseen_evScore,unseen_r2Score,unseen_mseScore,unseen_maeScore= st.Metrics(unseen_target,unseen_target_prediction)
        unseen_ev = unseen_evScore + unseen_ev; unseen_r2 = unseen_r2Score + unseen_r2; unseen_mse = unseen_mseScore + unseen_mse
        unseen_mae = unseen_maeScore + unseen_mae
        unseenTrue = unseen_target.flatten() + unseenTrue;unseenPrediction = unseen_target_prediction.flatten() + unseenPrediction

    testTrue, testPrediction = st.Transpose(testTrue, testPrediction)
    testResult = pd.DataFrame(data=[testTrue.tolist(), testPrediction.tolist()]).transpose()
    testResult.columns = ["True", "Prediction"]
    unseenTrue, unseenPrediction = st.Transpose(unseenTrue, unseenPrediction)
    unseenResult = pd.DataFrame(data=[unseenTrue.tolist(), unseenPrediction.tolist()]).transpose()
    unseenResult.columns = ["True", "Prediction"]
    sheetNameList = ["TestData", "UnseenData"]
    dataSetList = [testResult, unseenResult]
    excelFilePath = "D:/pythonproject/ThesisModel/Optuna-DNN/result/Optuna-DNNResultData.xlsx"
    st.GenerateExcel(excelFilePath)
    for i in range(len(sheetNameList)):
        st.WriteExcel(dataSetList[i], sheetNameList[i], excelFilePath)
    model = SaveModel(study,x_train_new, y_train_new)
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
    print(study.best_params)
    print("--------------------------------------Finish!!!--------------------------------------")

