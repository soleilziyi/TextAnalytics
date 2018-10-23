
# coding: utf-8

# In[16]:

from __future__ import division

import pandas as pd
from pandas import ExcelWriter
import numpy as np
#import html2text
from datetime import date
from scipy.stats import norm
from scipy.optimize import minimize
import random
import os

#Packages for pulling text data 
from urllib.request import urlopen  # the lib that handles the url stuff
from bs4 import BeautifulSoup
import pandas.io.data as web
from pandas.tseries.offsets import BDay

#Packages for text data processing
import nltk, re, pprint
#nltk.download()
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer

#Packages for statistical learning 
'''
from sklearn.preprocessing import normalize as Normal
from sklearn import svm
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn import linear_model
'''
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import matplotlib
matplotlib.style.use('ggplot')

# In[17]:

#Function to get event listed in the raw 8-K text 
#Input:string (text) & string (events_list)
#Output: pd.DataFrame
def get_Events(text,events_list):
    
    events_stats=pd.DataFrame({"count":[0]*len(events_list)},index=events_list)
    
    for event in events_list:
        try:
            re.search(event,text)
            if re.search(event,text):
                events_stats.loc[event,"count"]=1 
        except:
            pass
        
    return(events_stats)

#Function to give the events list in a document  
def Doc_Events(df,events_list):
    ndocs=len(df)
    nevents=len(events_list)
    
    df2=pd.DataFrame(np.random.randn(ndocs,nevents),index=df.index,columns=events_list)

    for i in range(0,ndocs):
        text=df.Text.iloc[i]
        events_stats=get_Events(text,events_list)
        df2.iloc[i]=events_stats["count"]
    
    df3=pd.concat((df[['CIK','Company Name','Date Filed']],df2),axis=1)
    
    return(df3)
    
#Get aggregated events time series for the index 
def Index_Events(Events_df,Index,Time_range):
    
    Event_df=Events_df.sort_values("Company Name")
    Events_df_new=Events_df.merge(Index[["Name","Weight"]],right_on="Name",left_on="Company Name",how="left")

    Events_df_new=Events_df_new.drop("Name",axis=1)
    Events_df_new2=Events_df_new.copy()
    Events_df_new2=Events_df_new2.drop("Weight",axis=1)
     
    for item in Events_df_new2.columns[3:]:
        Events_df_new2[item]=Events_df_new[item]*Events_df_new["Weight"]
      
    Temp=Events_df_new2.sort_values("Date Filed")
    
    Temp=Temp.groupby("Date Filed").sum()
    Temp=Temp.drop("CIK",axis=1)
    
    Time_temp=pd.DataFrame(index=Time_range)
    Temp=Temp.merge(Time_temp,how="outer",right_index=True,left_index=True)
    Temp=Temp.fillna(0)
    
    return(Temp)

#Get index of docs containing one event 
def get_event_doc(Events_df,item):
    docs_index=Events_df[Events_df[item]==1].index
    return(docs_index)

#Get text containig one event 
def get_event_text(text,item):
    beg_match=re.search(item,text)
    beg=beg_match.start()
    item_pattern=re.compile('ITEM\s\d.\d\d\s\s\s+')
    m=item_pattern.search(text,beg+1)
    if m:
        end=min(m.start(),len(text))
    else: 
        end=len(text)
    event_text=text[beg:end]
    return(event_text)

#Clean text Data 
def clean_text(raw_text,comp_name):

    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    comp_name=re.sub("[^a-zA-Z]", " ", comp_name) 
    
    # 2. split into individual words
    words = letters_only.lower().split()
    comp_words=comp_name.lower().split()
    
    # 3. Convert stopwords to a list
    stops = set(stopwords.words("english"))                  
    
    # 4. Remove stop words
    meaningful_words = [w for w in words if not (w in stops or w in comp_words)]   
    
    # 5. Stemming to get root words 
    ps = PorterStemmer()
    stemmed_words=[ps.stem(w) for w in meaningful_words]
    
    # return the result
    return(" ".join( stemmed_words ))  

#Get noun of cleaned text data 
def get_noun(text):
        
    sentences = nltk.sent_tokenize(text) #tokenize sentences
    nouns = [] #empty to array to hold all nouns

    for sentence in sentences:
         for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
             #if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                 nouns.append(word)
    return(" ".join( nouns ))

#Get top n keywords of a text 
def get_Keywords(s, n):
    #s is a row of tfidf table
    x=min(n, sum(s!=0)) #if there is less than 10 words in a line
    temp = np.argpartition(-s, x) #get the top x keywords
    top_n=s.iloc[temp[:x]].index.tolist()
    top_n.extend([np.NaN]*(n-x))
    return top_n

#Get top n keywords of a text 
def Doc_Keywords(df, tfidf_df, n ):
    df2=pd.DataFrame(index=df.index, columns=range(0,n))
    
    for i, row in tfidf_df.iterrows():
        #print(i)
        df2.loc[i,:]=pd.Series(get_Keywords(row,n), index=range(0,n))
    
    df3=pd.concat((df[['CIK','Company Name','Date Filed']],df2),axis=1)
    return df3

#Define train data 
def get_train(X,Y,train_ratio):
    train_sample=int(X.shape[0]*train_ratio)
    X_train=X[:train_sample,:]
    Y_train=Y[:train_sample]
    return(X_train, Y_train )

#Define test data 
def get_test(X,Y,train_ratio):
    train_sample=int(X.shape[0]*train_ratio)
    X_test=X[train_sample:,:]
    Y_test=Y[train_sample:]
    return(X_test, Y_test )

#Get optimal lasso regression hyper-parameters 
def Lasso_param(X_train,Y_train):
    
    #rough grid
    kf = KFold(n_splits=10)
    parameters = {'alpha':np.arange(0,10,0.5)}
    lasso = linear_model.Lasso()
    clf = GridSearchCV(lasso, parameters,cv=kf)
    clf.fit(X_train, Y_train)
    best_param=clf.best_params_['alpha']
    print(best_param)
    
    #fine grid
    parameters = {'alpha':np.arange(best_param-0.5,best_param+0.6,0.1)}
    clf = GridSearchCV(lasso, parameters,cv=kf)
    clf.fit(X_train, Y_train)
    best_param=clf.best_params_['alpha']

    return best_param

#Get optimal SVR regression hyper-parameters 
def SVR_param(X_train,Y_train):
    
    #rough grid
    kf = KFold(n_splits=10)
    parameters = {'C':np.exp(np.arange(-5,10,1))}
    svr = SVR(kernel='linear',epsilon=0.01)
    clf = GridSearchCV(svr, parameters,cv=kf)
    clf.fit(X_train, Y_train)
    best_power=np.log(clf.best_params_['C'])
    
    #fine grid
    parameters = {'C':np.exp(np.arange(best_power-0.5,best_power+0.6,0.1))}
    clf = GridSearchCV(svr, parameters,cv=kf)
    clf.fit(X_train, Y_train)
    best_power=np.log(clf.best_params_['C'])

    return best_power

#Get optimal SVR regression hyper-parameters  - to debug
def SVM_param(X_train,Y_train):

    #rough grid
    skf = StratifiedKFold(n_splits=10)
    parameters = {'C':np.exp(np.arange(-5,10,1))}
    svc = svm.SVC(kernel='linear')
    clf = GridSearchCV(svc, parameters,cv=skf)
    clf.fit(X_train, Y_train)
    best_power=np.log(clf.best_params_['C'])
    
    #fine grid
    parameters = {'C':np.exp(np.arange(best_power-0.5,best_power+0.6,0.1))}
    clf = GridSearchCV(svc, parameters,cv=skf)
    clf.fit(X_train, Y_train)
    best_power=np.log(clf.best_params_['C'])
    
    return(best_power)


# In[18]:

class KeywordAnalyzer:
    
    # constructor: pass in the keyword data frame, and the list of tickers associated with company names
    def __init__(self, keywords_df, ticker_df ):
        self.__Keywords=keywords_df.copy()
        self.__Keywords['Ticker']=self.__Keywords['Company Name'].apply(lambda x:ticker_df[ticker_df.Name==x]['Ticker'].tolist()[0])
        self.__Keywords['Ticker']=self.__Keywords['Ticker'].apply(lambda x:x.split(' ')[0])
        try:
            self.__Keywords['Date Filed']=self.__Keywords['Date Filed'].apply(lambda x: pd.to_datetime(x, format='%Y-%M-%d'))
        except:
            pass
        
        #initialize response dataframe
        self.__Y=pd.DataFrame(index=keywords_df.index)
        #initialize return time series
        self.__TS=pd.DataFrame()
        
        
    # display       
    def get_X(self):
        return self.__Keywords
    
    
    # get Y data as response
    def get_Y(self, kind='Return'):
        if(kind in self.__Y.columns.tolist()): #already calculate
            pass
        else:
            self.__Y[kind]=self.__Keywords.apply(lambda x: get_Ticker_Y(x.loc['Ticker'], x.loc['Date Filed'], kind),axis=1)
            
        return self.__Y
            
    # get stock return time series
    def get_TS(self, window=10):
        if(str(window-1) in self.__TS.columns.tolist()): #already called
            pass
        else:
            #tmp_df=pd.DataFrame(index=self.__TS.index, columns=np.arange(window,-1,-1))
            for i,row in self.__Keywords.iterrows():
                tmp=get_Ticker_TS(row.Ticker, row['Date Filed'], window)
                tmp.name=i
                self.__TS=self.__TS.append(tmp)
             
        return (self.__TS)
    
    def prep_Classifier(self, kind='Return', normalize=True):
        tmp=self.__Keywords.drop(['Ticker','CIK','Company Name','Date Filed'],axis=1).copy()
        if(normalize):
            TS_mat=Normal(self.__TS.as_matrix(),axis=0)
            KW_mat=Normal(tmp.as_matrix(),axis=0)
        else:
            TS_mat=self.__TS.as_matrix()
            KW_mat=tmp.as_matrix()
                
        label=(self.__Y[kind].as_matrix().T>0)*2-1
        
        return TS_mat, KW_mat, label
      


# In[19]:

#helper function
def get_Ticker_Y(ticker, date, kind, window=22, window2=5):
    #get trading day
    if(date==date+BDay(0)): # if date is a trading day
        T0=date
        T1=date+BDay(1)
    else:
        T0=date-BDay(1)
        T1=date+BDay(1)
    
    
    if(kind=='Return'):# get T+1 return
        try:
            price = web.DataReader(ticker, 'yahoo', T0, T1)
        except:
            try:
                price = web.DataReader(ticker, 'google', T0, T1)
            except:
                return 0
        
        try:
            return price.Close[1]/price.Close[0]-1
        except:
            return 0
        
    elif(kind=='Return_Z'): # get T+1 Z score of return
        try:
            price = web.DataReader(ticker, 'yahoo', date-BDay(window), T1)
        except:
            try:
                price = web.DataReader(ticker, 'google', date-BDay(window), T1)
            except:
                return 0
        
        daily_return=price.Close.pct_change(1)
        return float((daily_return.tail(1)-daily_return.mean())/daily_return.std())
        
    elif(kind=='Vol_Ratio'): # get T+1 
        try:
            price = web.DataReader(ticker, 'yahoo', date-BDay(window), T1+BDay(window2-1))
        except:
            try:
                price = web.DataReader(ticker, 'google', date-BDay(window), T1+BDay(window2-1))
            except:
                return 0
            
        daily_return=price.Close.pct_change(1)    
        return float(daily_return.tail(5).std(ddof=0)/daily_return.head(22).std(ddof=0))

        


# In[20]:

# helper function
def get_Ticker_TS(ticker, date, window):
    if(date==date+BDay(0)): # if date is a trading day
        T0=date
    else:
        T0=date-BDay(1)
        
        
    try:
        price = web.DataReader(ticker, 'yahoo', T0-BDay(window+3), T0)
    except:
        try:
            price = web.DataReader(ticker, 'google', T0-BDay(window+3), T0)
        except:
            return 0
        
    r=price.Close.pct_change(1).tail(window)
    r.index=['Lag '+ str(i) for i in np.arange(len(r)-1,-1,-1)]
    return r


# In[ ]:
class Text_Garch:
    def __init__(self, TS, Text):
        self.TS=TS # Time series of RETURN not Price,
        self.Text=Text # range of dates will be handled after
        self.tmp_LL=0
        print('Ready')
    
    def Log_likelihood(self,text,start,end, AR_lags,intercept, decision):
        # parse decision
        local_TS=self.TS.loc[start:end]
        # initial var and noise
        var=abs(decision[0])
        noise=decision[1]**2
        # auto-regressive parameters
        decision1=decision[2:2+intercept[0]+len(AR_lags)]
        # var parameters
        decision2=decision[2+intercept[0]+len(AR_lags):]                                        
                                                
        if intercept[0]==True:
            alpha=decision1[0]
            beta=decision1[1:]
        else:
            alpha=0
            beta=decision1
            
        if intercept[1]==True:
            mu=decision2[0]
            gamma=decision2[1:]
        else:
            mu=0
            gamma=decision2[:]
            
        if text:
            phi=decision2[-2] 
            decay=decision2[-1] 
        else:
            phi=0
            decay=0
        
        self.params={"alpha":alpha,"beta":beta,"gamma":gamma,"mu":mu,"phi":phi,"decay":decay}
        
        # get decay_Text
        self.decay_Text=pd.Series(index=local_TS.index)
        for i,item in self.decay_Text.iteritems():
            if i>self.Text.index[0]: # after first event
                weights= np.array(0.5**((i-self.Text.loc[:i].index).days/decay))
                self.decay_Text.loc[i]=self.Text.loc[:i].as_matrix().dot(weights)
            else:
                self.decay_Text.loc[i]=0
        self.decay_Text.fillna(0)

        # shift autoregressive lags
        log_likelihood=0
        Y=local_TS.iloc[max(AR_lags):]
        X=pd.DataFrame(columns=AR_lags)
        for i in AR_lags:
            X[i]=local_TS.shift(i)
        # calculate likelihood 
        if text:
            for i, item in Y.iteritems():
                expectation=float(alpha+beta.dot(X.loc[i,:].as_matrix().reshape(-1,1)))
                log_likelihood+=norm.logpdf(item, expectation, np.sqrt(var))
                #update noise and var
                noise=(item-expectation)**2
                var=abs(mu+gamma[0]*noise+gamma[1]*var+phi*self.decay_Text.loc[i])
        else:
            for i, item in Y.iteritems():
                expectation=float(alpha+beta.dot(X.loc[i,:].as_matrix().reshape(-1,1)))
                log_likelihood+=norm.logpdf(item, expectation, np.sqrt(var))
                #update noise and var
                noise=(item-expectation)**2
                var=abs(mu+gamma[0]*noise+gamma[1]*var)
            
        
        self.var=var
        self.expectation=expectation
            
        self.decay=decay
        self.text_coef=phi
        
        if(log_likelihood>self.tmp_LL+0.1):
            print(log_likelihood)
            self.tmp_LL=log_likelihood
        return -log_likelihood
    
    def Estimate(self,text=True,start=None,end=None ,AR_lags=[1],intercept=[True, True],decay_range=(0.01,5),initial=None):
        if start==None:
            start=self.TS.index[0]
        if end==None:
            end=self.TS.index[-1]
            
        #objective function wrapper
        obj = lambda x: self.Log_likelihood(text,start,end, AR_lags,intercept, x)
        nb=sum(intercept)+len(AR_lags)+text*2+2+2
        print(nb)
        
        if initial==None: 
            initial=np.random.rand(nb)*0.1
            
        if text:
            bnds=[(None,None) for i in range(0,nb-2)]
            bnds=bnds+[(0,None)]+[decay_range]
        else:
            bnds=[(None,None) for i in range(0,nb)]
            
        res=minimize(obj,initial,bounds=bnds)
        self.var0=abs(res.x[0])
        self.noise0=(res.x[1])
        self.MLL=-res.fun
        print(res.success)
        return self.MLL, self.text_coef, self.decay


def save_xls(list_dfs, xls_path,df_names):
    writer = ExcelWriter(xls_path)
    for n, df in enumerate(list_dfs):
        df.to_excel(writer,df_names[n])
    writer.save()  

