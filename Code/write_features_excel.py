# Author@ Liu
# Time：2022/5/10 15:30
import pandas as pd
from ThesisModel import dataprocessing as dp

dataSet = dp.ReadData()
features, target = dp.SplitFeatureAndTarget(dataSet)
features_DataDistribution = dp.DataDistribution(features)
new_features, new_target = dp.Drop_noisy(features_DataDistribution, target)
x_train, x_test, y_train, y_test = dp.SplitTestAndTrain(new_features, new_target)

nameList = ["cut","rpm","CL","counts"]
test_features = pd.DataFrame(data=[x_test,y_test])
test_features.columns = nameList
test_features.to_excel("D:/pythonproject/ThesisModel/result/test_features.xlsx",index=False)

x_train_MinMax, x_test_MinMax, y_train_MinMax, y_test_MinMax = dp.Normalization(x_train, x_test, y_train, y_test)
unseen_features, unseen_target, x_train_new, y_train_new = dp.CreateUnseenData(x_train_MinMax, y_train_MinMax)

train_features = pd.DataFrame(data=[x_train_new,y_train_new])
train_features.columns = nameList
train_features.to_excel("D:/pythonproject/ThesisModel/result/train_features.xlsx",index=False)
unseen_features = pd.DataFrame(data=unseen_features)
unseen_features.columns = nameList
unseen_features.to_excel("D:/pythonproject/ThesisModel/result/unseen_features.xlsx",index=False)

