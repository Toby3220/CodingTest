import datetime as dt
import os

import pandas as pd 
import numpy as np
from pathlib import Path
import sklearn.neural_network as skl_nn
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms


# V3.0

class InputData:
    def __init__(self) -> None:
        pass
    
    def ImportRaw(self,fname: str)-> None:
        #import Data
        self.RawData = pd.read_csv(fname)
        self.RawData.set_index("date",inplace=True)
    
        #Unique set of Dates, Tickers and Features
        self.TickerSet = set(self.RawData.loc[:,"ticker"])
        self.DateSet = list(set(self.RawData.index))
        self.DateSet.sort()

        self.FeatureSet = set(self.RawData.columns)
        self.FeatureSet.remove("ticker")
        
        #Compile DataBase
        self.FeatureData = dict()
        
        self.tmbool = pd.DataFrame(data=1,columns=self.TickerSet,index=self.DateSet)

        for feature in self.FeatureSet:
            tempdf = pd.DataFrame(data=np.nan,columns=self.TickerSet,index=self.DateSet)
            for ticker in self.TickerSet:
                templist = self.RawData[self.RawData.loc[:,"ticker"]==ticker].loc[:,feature]
                tempdf.loc[:,ticker]=templist
            self.tmbool*= tempdf!=np.nan
            self.FeatureData[feature]=tempdf
            
    def FormatData(self, export: bool = False,exportpath:str="Processed Data", applymask: bool = False, applynormalisation: bool = False, splits: dict={"All":1}) -> dict:
    ##DESCRIPTION
        out = dict()
        
        for datasplits in splits.keys():
            out[datasplits] = pd.DataFrame(columns=self.FeatureSet)
            
        for feature in self.FeatureSet:
            cdf = pd.DataFrame(self.FeatureData[feature])#cdf = current dataframe
            cdf = cdf.reindex_like(self.tmbool)
            
            #sorting rows and columns
            cdf.sort_index(axis=0,inplace=True) 
            cdf.sort_index(axis=1,inplace=True)
    
            if (applymask == True):  
                cdf[self.tmbool==False]=np.NaN

            startval = 0
            cumfrac = 0
            totrows = len(self.DateSet)
            for datasplit in splits.keys():
                cumfrac += splits[datasplit]
                endval = min(int(np.round(totrows*cumfrac)),totrows-1)
                MasterData = out[datasplit] 
                DataAdd = cdf.iloc[startval:endval,:].stack(dropna=False)
                startval=endval
                
                if (applynormalisation == True) and (feature!= "Returns"):
                    #Min-Max Scaling
                    maxval = DataAdd[DataAdd.isin([0,np.NaN,np.inf,-np.inf,True,False])==False].max().max()
                    minval = DataAdd[DataAdd.isin([0,np.NaN,np.inf,-np.inf,True,False])==False].min().min()
                    range = maxval-minval
                    if np.isnan(maxval)==False:
                        DataAdd-=minval
                        DataAdd = DataAdd.divide(range)

                MasterData[feature]=DataAdd
                out[datasplit]=MasterData
           
        if export == True:
            for datasplit in splits.keys():
                fp = Path('{}/{}.csv'.format(exportpath,datasplit))  
                fp.parent.mkdir(parents=True, exist_ok=True)  
                out[datasplit].to_csv(fp)
        return out
    
    def GenTmbool(self,splits:dict)->dict:
        tmboolout = dict()
        startval = 0
        cumfrac = 0
        endval = 0
        totrows = len(self.DateSet)
        for datasplit in splits.keys():
            cumfrac += splits[datasplit]
            endval = min(int(np.round(totrows*cumfrac)),totrows-1)
            DataAdd = self.tmbool.iloc[startval:endval,:]
            startval=endval
            tmboolout[datasplit]=DataAdd
        return tmboolout

    
    
    def GenEmptyDF(self)-> pd.DataFrame:
        df = pd.DataFrame(data=np.nan,index=self.DateSet,columns=self.TickerSet)

    def AddFeature(self,Feature:str,FeatureData:pd.DataFrame)->None:
        self.FeatureSet.add(Feature)
        self.FeatureData[Feature]=FeatureData
    
    def AddReturns(self,period:int)->None:
        curprice=self.FeatureData["last"]
        futprice=curprice.shift(-period)
        Returns =(futprice-curprice).divide(curprice)
        self.AddFeature("Returns",Returns)
    
        
class LearnModel:
    
    def __init__(self) -> None:
        self.EstRet = dict()
        self.AccRet = dict()

        pass

    def __init__(self, splits:dict, pathdirectory: str = "Processed Data/V1", tmbooldircetory: str = "Processed Data/V1/Tmbool") -> None:
        self.EstRet = dict()
        self.Splits = splits
        self.Data = dict()
        self.Tmbool=dict()
        self.AccRet = dict()
        for datasplits in splits.keys():
            pname = "{}/{}.csv".format(pathdirectory,datasplits)
            tname = "{}/{}.csv".format(tmbooldircetory,datasplits)
            cdf = pd.read_csv(pname,parse_dates=True,infer_datetime_format=True)
            tmboolcdf = pd.read_csv(pname,parse_dates=True,infer_datetime_format=True)

            cdf.rename(columns={"Unnamed: 0":"date","Unnamed: 1":"ticker"},inplace=True)
            cdf.set_index([pd.Index(range(0,len(cdf))),"date", "ticker"],inplace=True)

            tmboolcdf.rename(columns={"Unnamed: 0":"date"})
            tmboolcdf.set_index(["date"],inplace=True)


            self.Data[datasplits] = cdf
            self.Tmbool[datasplits] = tmboolcdf
    
    def __init__(self,data: dict,splits: dict,tmbool: dict) -> None:
        self.EstRet = dict()
        self.Data = data
        self.Splits = splits
        self.Tmbool = tmbool
        self.AccRet = dict()




    def PrepForSkl(self,set: str,prepX: bool= True, prepY: bool = True)-> None:
        TrainingData = self.Data[set]
        TrainingData.dropna(inplace=True)
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        if prepY:
            self.Y = TrainingData.loc[:,"Returns"]
        if prepX:
            self.X = TrainingData.drop("Returns",axis=1)
        
    def TrainModel(self,set: str):
        pass
    def PredictModel(self,set:str)-> None:
        pass
    


class LearnModel_BayesianRidge (LearnModel): 
    
    def TrainModel(self,set: str)-> None:
        self.PrepForSkl(set)
        self.Model = skl_lm.BayesianRidge()
        self.Model.fit(self.X.to_numpy(),self.Y.to_numpy())

    def Train_GridSearchFit(self,set:str,random_state=2):
        self.PrepForSkl(set)
        self.Model = skl_lm.BayesianRidge()
        param_grid = {
            'alpha_1': [0.00001,0.0001, 0.0005],
            'alpha_2': [0.00001,0.0001, 0.0005],
            'lambda_1': [0.00001,0.0001, 0.0005],
            'lambda_2': [0.00001,0.0001, 0.0005],

        }
        CrossValidationSet = skl_ms.TimeSeriesSplit()
        self.Model = skl_ms.GridSearchCV(self.Model,param_grid,verbose=4,scoring='neg_mean_squared_error',cv=CrossValidationSet).fit(self.X,self.Y)
        self.BestParam = self.Model.best_params_
        print(self.BestParam)
    

    def PredictModel(self,set: str,)-> None:
        self.PrepForSkl(set)
        self.EstRet[set] = pd.DataFrame(self.Model.predict(self.X.to_numpy()),index=self.X.index,columns=["Retruns"]).unstack()
        self.AccRet[set] = self.Y.unstack()
        
        
class StratEval:

    def __init__(self,TrainedModel: LearnModel):
        self.Model = TrainedModel
        self.WghMat = dict()
        self.EstRet = dict()
        self.AccRet = dict()
        self.Results = dict()
        self.PerformanceStrat = dict()
        self.PerformanceBase = dict()
        self.PerformanceMetrics = ["Mean Annual Ret", "Sharpe", "Max Drawdown"]
        pass

    def GenStrat(self):
        pass

    def Eval(self,bps,set):
        pass

    def EvalRes(self, set:str, bps=0,Outpath= "Results/V1/Output.csv" , Verbose:bool = True, discardhead:int= -1):
        
        self.Results[set] =self.CreateOutput(set,bps, Outpath, ReturnDF = True)

        Results_ActualEarnings = self.Results[set].head(discardhead)["Actual Earnings"]
        Results_AverageEarnings =self.Results[set].head(discardhead)["Evenly Weighted Avg"]

        if Verbose: print("Stratergy Performance")
        [StratMeanAnnualRet, StratSharpe, StratMaxDrawDown, StratMDDPosition] = self.CalcMeasurements(Results_ActualEarnings, Verbose=Verbose)
        if Verbose: print("Average Performance")
        [AvgMeanAnnualRet, AvgSharpe, AvgMaxDrawDown, AvgMDDPosition] = self.CalcMeasurements(Results_AverageEarnings, Verbose=Verbose)
        
        self.PerformanceStrat[set] = pd.Series([StratMeanAnnualRet, StratSharpe, StratMaxDrawDown],index=self.PerformanceMetrics)
        self.PerformanceBase[set] = pd.Series([AvgMeanAnnualRet, AvgSharpe, AvgMaxDrawDown],index=self.PerformanceMetrics)

    def EvalPlot(self,set:str,Export_Figures=False,Export_Figure_Names:str="NA",Export_Location_Name:str="NA"):
        self.Plotting(self.Results[set],Export_Figures,Export_Figure_Names,Export_Location_Name,self.WghMat[set])

    def CreateOutput(self, set: str, bps: float = 0, Outpath: str = "Results/V1/Output.csv", ReturnDF: bool = False, Export: bool = True):
        WghMat = self.WghMat[set]
        AccRet = self.Model.AccRet[set]
        EstRet = self.Model.EstRet[set]

        AvgMat = pd.DataFrame(1,index=WghMat.index,columns=WghMat.columns)
        AvgMat = AvgMat.divide(AvgMat.sum(axis=1,skipna=True).sum())
                
        [Earnings,Transactions] = self.CalcReturns(WghMat,AccRet,bps)
        [EstEarnings,temp] = self.CalcReturns(WghMat,EstRet,bps)
        [AvgControl,temp] = self.CalcReturns(AvgMat,AccRet,0)

        StratPortGrowth = (Earnings+1).cumprod()
        SPXPortGrowth = (AvgControl+1).cumprod()
        GrowthPDiff = (StratPortGrowth-SPXPortGrowth)/SPXPortGrowth

        StratAnnRet = (StratPortGrowth.shift(-12)-StratPortGrowth)/StratPortGrowth
        SPXAnnRet = (SPXPortGrowth.shift(-12)-SPXPortGrowth)/SPXPortGrowth
        PrefDiff = StratAnnRet-SPXAnnRet
        

        out = pd.concat([EstEarnings,Earnings,AvgControl,Transactions,StratPortGrowth,SPXPortGrowth,GrowthPDiff,StratAnnRet,SPXAnnRet,PrefDiff],axis=1)
        out = pd.DataFrame(out.values,out.index,columns=["Estimated Earnings","Actual Earnings","Evenly Weighted Avg","Transaction Costs","Strategy Portfolio Growth", "Avg Mean Portfolio Growth","Percentage Growth Difference","Strategy Annual Return","Avg Mean Annual Return","Annual Return Diference"])
        
        
        if Export:
            fp = Path('{}'.format(Outpath))  
            fp.parent.mkdir(parents=True, exist_ok=True)  
            out.to_csv(fp)

            fp = Path('{}/WghMat.csv'.format("Debug-Log"))  
            fp.parent.mkdir(parents=True, exist_ok=True)  
            WghMat.to_csv(fp)

            

        if ReturnDF == True:
            return out

    def Plotting(self,df:pd.DataFrame,savefigbool:bool=False,figname:str="NA",figlocname:str="NA",WghMat=None):

        #1st Figure: Portfolio Growth Over Time
        Skey = "Strategy Portfolio Growth"
        Mkey = "Avg Mean Portfolio Growth"
        plt.figure(0)
        # h1, = plt.plot(df[Skey],"b",linewidth=2)
        h1, = plt.plot(df[Skey],linewidth=2)
        h2, = plt.plot(df[Mkey],"r",linewidth=2)
        plt.title("Portfolio Growth Over Time")
        plt.xlabel("Months")
        plt.ylabel("Growth")
        plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
        plt.axvline(x=df.index[0],color="k",linestyle="--")
        
        #2nd Figure: Percentage Growth Difference
        Skey ="Percentage Growth Difference"
        plt.figure(1)
        plt.plot(df[Skey]*100)
        plt.axline((0,0),slope=0,color="k",linestyle="--")
        #plt.fill_between(np.arange(0,df.shape[0],1,int),df[Skey],0,interpolate=True,color="c")
        plt.title(Skey)
        plt.xlabel("Months")
        plt.ylabel("% Difference")
        plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
        plt.axvline(x=df.index[0],color="k",linestyle="--")

        #3rd Figure: Annual Returns
        Skey = "Strategy Annual Return"
        Mkey = "Avg Mean Annual Return"
        plt.figure(2)
        # h1, = plt.plot(df[Skey],"b",linewidth=2)
        h1, = plt.plot(df[Skey]*100,linewidth=2)
        h2, = plt.plot(df[Mkey]*100,"r",linewidth=2)
        plt.title("Annual Returns")
        plt.xlabel("Months")
        plt.ylabel("% Return")
        plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
        plt.axvline(x=df.index[0],color="k",linestyle="--")

        #4th Figure: Annual Return
        Skey = "Avg Mean Annual Return"
        Mkey = "Strategy Annual Return"

        plt.figure(3)
        plt.scatter(x=df[Skey]*100,y=df[Mkey]*100)
        plt.axvline(color="k",linestyle="--")
        plt.axline((0,0),slope=0,color="k",linestyle="--")
        plt.axline((0,0),slope=1,color="b",linestyle="-")

        plt.title("Stratergy Returns Vs Even Avg Returns")
        plt.ylabel("Strategy % Returns")
        plt.xlabel("Even Avg % Returns")

        #5th Figure: Annula Return Difference
        Skey ="Annual Return Diference"
        plt.figure(4)
        plt.plot(df[Skey]*100)
        plt.axline((0,0),slope=0,color="k",linestyle="--")
        #plt.fill_between(np.arange(0,df.shape[0],1,int),df[Skey],0,interpolate=True,color="c")
        plt.title("% Points Difference on Returns")
        plt.xlabel("Months")
        plt.ylabel("% Points Difference")
        plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
        plt.axvline(x=df.index[0],color="k",linestyle="--")

        #6th Figure: Portfolio Growth Over Time
        Skey = "Strategy Portfolio Growth"
        Mkey = "Avg Mean Portfolio Growth"
        plt.figure(5)
        # h1, = plt.plot(df[Skey],"b",linewidth=2)
        h1, = plt.plot(np.log(df[Skey]),linewidth=2)
        h2, = plt.plot(np.log(df[Mkey]),"r",linewidth=2)
        plt.title("Log Portfolio Growth Over Time")
        plt.xlabel("Months")
        plt.ylabel("Log Growth")
        plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
        plt.axvline(x=df.index[0],color="k",linestyle="--")

        #7th Figure: Scatter between Estimated and Actual Earnings
        Mkey = "Actual Earnings"
        Skey = "Estimated Earnings"
        df1=df[df[Skey]<=10]
        df1=df1[df1[Skey]>=0]
        
        plt.figure(6)
        plt.scatter(x=df1[Skey]*100,y=df1[Mkey]*100)
        plt.axvline(color="k",linestyle="--")
        plt.axline((0,0),slope=0,color="k",linestyle="--")
        plt.axline((0,0),slope=1,color="b",linestyle="-")

        plt.title("Actual Vs Estimated Returns")
        plt.xlabel("Estimated % Returns")
        plt.ylabel("Actual % Returns")

        # 8th Figure Maximum Weights 
        if WghMat is not None:
            ans = WghMat.max(1,skipna=True,numeric_only=True)

            plt.figure(8)
            plt.plot(100*ans)
            plt.xticks(np.arange(10,105,12,int),np.arange(2014,2022,1,int))
            plt.title("Max Stock Weight in Strategy")
            plt.xlabel("Months")
            plt.ylabel("% Weighting")

        plt.show()
        
        #save pictures
        if savefigbool == True:
            for k in np.arange(0,7,1,int):
                plt.figure(k)
                accfigname = "{}figure{}".format(figname,k)
                figpath = Path("{}/{}.png".format(figlocname,accfigname))
                plt.savefig(fname=figpath)


    def CalcMeasurements(self,rets, periods_per_year = 12, Verbose = True):
        #rets = 1D array or pandas series
        n = len(rets)/periods_per_year   #no. of years
        cumrets = (1+rets).cumprod()   #cumulative returns

        #Mean Annual Return
        MeanAnnualRet = (cumrets[-1]**(1/n) - 1)

        #scale to average annual return and volatility
        Sharpe = (MeanAnnualRet)/(np.std(rets) * np.sqrt(periods_per_year))
    
        max_cumrets = cumrets.cummax()  #max previous cumret
        dd = 1 - cumrets/max_cumrets   #all drawdowns
        MaxDrawDown = np.max(dd)
        MDDPosition = np.argmax(dd)
        if Verbose:
            #Add This Back
            os.system("pause")
            print("Results:\n - Mean Annual Returns: \t{} \n - Sharpe Ratio: \t\t{} \n - Maximum Drawdown: \t\t{}".format(MeanAnnualRet,Sharpe,MaxDrawDown))

        return MeanAnnualRet, Sharpe, MaxDrawDown, MDDPosition

    def CalcReturns(self,weightmat:pd.DataFrame,returnmat:pd.DataFrame,k:float=0) ->pd.Series:
    ##DESCRIPTION
    # returning a Series of Retruns, given a weighting and returns matrix

    #Takes:
    #   - weightmat     - the weighting matrix, index = time series data, column = tickers, values = the fractional weight of the portfolio at said time
    #   - returnmat     - the return matrix, same index AND columns a the weightmat; corrisponds to the actual (%) Return of the ticker at said time period
    #Returns: 
    #   - ActualRet     - the commission ajusted (%) Returns
    #   - TransCost     - the total transactional cost, for said time period

    ##CODE
        twm = weightmat.sort_index(axis=0).sort_index(axis=1).values
        ret = returnmat.sort_index(axis=0).sort_index(axis=1).values
        prod = pd.DataFrame(twm*ret,index=weightmat.index,columns =weightmat.columns)
        PortfolioRet = prod.sum(1,skipna=True)
        TransCost = (k/10000)*(weightmat-weightmat.shift(-1,axis=0,fill_value=0)).abs().sum(1,skipna=True)
        ActualRet = PortfolioRet-TransCost
        return ActualRet, TransCost

    

        
        #save pictures
    


    

class StratEval_TopNPerforming (StratEval):
    def GenStrat(self,n:int,set:str): 
        
        CurEstRet = self.Model.EstRet[set]
        CurWghMat = pd.DataFrame(CurEstRet.sort_index(axis=1))
        CurWghMat[CurWghMat.isna()]=0

        #Record Directoinality 
        Direction = CurWghMat
        Direction[CurWghMat>=0]= 1
        Direction[CurWghMat<0]= -1

        # find largest values
        AbsCurWghMat= CurWghMat.abs()
        AbsCurWghMat = self.NRowlargest(AbsCurWghMat,n)        

        AbsCurWghMat.sort_index(axis=0,inplace=True)
        AbsCurWghMat.sort_index(axis=1,inplace=True)

        #normalise
        AbsCurWghMat = AbsCurWghMat.divide(AbsCurWghMat.sum(axis=1,skipna=True,numeric_only=True),axis=0)

        # Restore Directionality 
        CurWghMat = AbsCurWghMat*Direction

        self.WghMat[set] = CurWghMat
    
    def NRowlargest(self,df:pd.DataFrame, n: int):
            for k in np.arange(0,df.shape[0],1,int):
                df.sort_values(by=[df.index[k]],axis=1,inplace=True,ascending=False)
                df.iloc[k,n:]=0
            return df
    
    

        



        

        
    
    
    
