
import Functions as f
import datetime as dt
import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn import neural_network as skl_nn
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt

DataV1 = f.InputData()
DataV1.ImportRaw(fname="data.csv")

#holding period
DataV1.AddReturns(12)

#Low pass filter, Comparison of long term and short term trends basis for MCAD 
DataV1.AddFeature("price EWMA 26",DataV1.FeatureData["last"].ewm(span=12).mean())
DataV1.AddFeature("price EWMA 12",DataV1.FeatureData["last"].ewm(span=26).mean())

DataV1.AddFeature("volume EWMA 12",DataV1.FeatureData["volume"].ewm(span=12).mean())
DataV1.AddFeature("volume EWMA 26",DataV1.FeatureData["volume"].ewm(span=26).mean())

#High pass filter, for the estimation of stock variance
HighPass = DataV1.FeatureData['last']-2*DataV1.FeatureData['last'].shift(1)+DataV1.FeatureData['last'].shift(2)
VarEst = pd.DataFrame(HighPass).ewm(span=12).var()
DataV1.AddFeature("price var", VarEst)

splits = {"Training":0.6,"Cross-Validation":0.2,"Test":0.2}

Out = DataV1.FormatData(export= True,exportpath="Processed Data/V1",applynormalisation=True,splits=splits)
TmboolOut = DataV1.GenTmbool(splits=splits)

StratModel = f.LearnModel_BayesianRidge(data = Out, splits=splits, tmbool=TmboolOut)
StratModel.Train_GridSearchFit("Training")
StratModel.PredictModel("Cross-Validation")

StratEval = f.StratEval_TopNPerforming(StratModel)
StratEval.GenStrat(n=5,set="Cross-Validation")

StratEval.EvalRes("Cross-Validation",5)
StratEval.EvalPlot("Cross-Validation",True,"V1","ResultPlots")


