# Author@ Liu
# Time：2022/5/6 14:39
import numpy as np
from sklearn.metrics import explained_variance_score,r2_score,mean_squared_error\
    ,mean_absolute_error
import openpyxl
import pandas as pd


def Metrics(true,prediction):
    """
        1. Metrics Model
    """
    evScore = explained_variance_score(true,prediction)
    r2Score = r2_score(true,prediction)
    mseScore = mean_squared_error(true,prediction)
    maeScore = mean_absolute_error(true,prediction)
    return evScore,r2Score,mseScore,maeScore

def GenerateExcel(excelFilePath):
    """
        2. GenerateExcel
    """
    book = openpyxl.Workbook()
    book.save(excelFilePath)

def Transpose(true,prediction):
    """
        3. Data Transpose
    """
    true = np.array(true).flatten()/20
    prediction = np.array(prediction).flatten()/20
    return true,prediction

def WriteExcel(dataSet,SheetName,excelFilePath):
    """
        4. Write Result Into Excel
    """
    with pd.ExcelWriter(excelFilePath, mode='a') as writer:
        dataSet.to_excel(writer, sheet_name=SheetName, index=False)






