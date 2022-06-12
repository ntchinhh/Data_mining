import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC     
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
import random
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def comparative(df):
    Y = df[['class']]
    X = df.iloc[:,df.columns !='class']
    r,c = df.shape
    
    acc_DT =list()
    acc_Navie = list()
    acc_SVM =list()
    acc_KNN = list()
    acc_logistic = list()
    logistic_MSE =list()
    logistic_RMSE= list()

    time_DT =list()
    time_Naive =list()
    time_SVM =list()
    time_KNN = list()
    time_logistic = list()

    for i in range(100):
        print ("Chay Lan thu: ", i)

# split data
        #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, stratify=Y, train_size=0.7)
        #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.7)
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.7)
# decision tree        
        start = datetime.now()
        model = DecisionTreeClassifier()                  
        model.fit(X_Train, Y_Train)               
        y_pred = model.predict(X_Test)
        end = datetime.now() -start
        time_DT.append(end)

        print("Accuracy Cay Quyet Dinh:",metrics.accuracy_score(Y_Test, y_pred))
        acc_DT.append(metrics.accuracy_score(Y_Test, y_pred))
        
# naive classifier
        start = datetime.now()
        model_navie = GaussianNB()
        #model_navie = MultinomialNB()        
        model_navie.fit(X_Train, Y_Train.values.ravel()) 
        y_pred = model_navie.predict(X_Test) 
        end = datetime.now() -start
        time_Naive.append(end)
        
        print("Accuracy Naive Bayes: ", metrics.accuracy_score(Y_Test, y_pred))
        acc_Navie.append(metrics.accuracy_score(Y_Test, y_pred))

# SVM
        start = datetime.now()
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(X_Train, Y_Train.values.ravel())
        y_pred = svclassifier.predict(X_Test)
        end = datetime.now() -start
        time_SVM.append(end)

        print("Accuracy SVM:",metrics.accuracy_score(Y_Test, y_pred))
        acc_SVM.append(metrics.accuracy_score(Y_Test, y_pred))
    
# KNN
        start = datetime.now()
        model_KNN = KNeighborsClassifier()
        model_KNN.fit(X_Train, Y_Train.values.ravel())
        y_pred = model_KNN.predict(X_Test)
        end = datetime.now() -start
        time_KNN.append(end)

        print("Accuracy KNN:",metrics.accuracy_score(Y_Test, y_pred))
        acc_KNN.append(metrics.accuracy_score(Y_Test, y_pred))
    
    results =[]
    results.append(acc_DT)
    results.append(acc_Navie)
    results.append(acc_SVM)
    results.append(acc_KNN)
    
    names =('Decision tree', 'Navie bayes', 'SVM', 'KNN')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy')    
    plt.show()            
    
    print ("Results")
    print ("Accuracy")
    
    print ("Accuracy Decision tree: ", np.mean(acc_DT))    
    print ("Accuracy Naive Bayes: ", np.mean(acc_Navie))    
    print ("Accuracy SVM: ", np.mean(acc_SVM))    
    print ("Accuracy KNN: ", np.mean(acc_KNN))    
       
    print ("Time")
    print ("Time Decision tree: ", np.mean(time_DT))
    print ("Time Naive Bayes: ", np.mean(time_Naive))
    print ("Time SVM: ", np.mean(time_SVM))
    print ("Time KNN: ", np.mean(time_KNN))
    
def run_compare(df):
    r,c = df.shape
    X = df.iloc[:,df.columns !='class']
    features = X.columns.tolist()
    y = df[['class']]

    acc_logistic = list()
   
    logistic_MSE =list()
    logistic_RMSE= list()
    time_logistic = list()
     
    for i in range (100):
        
        Train_x, Test_x, Train_y, Test_y = train_test_split(X, y, train_size=0.7)
               
        model_logistic  = LogisticRegression()
        model_logistic.fit(Train_x, Train_y)
        score = model_logistic.score(Train_x, Train_y.values.ravel())        
        #print("R-squared:", score)
        predictions = model_logistic.predict(Test_x)                
        mse = mean_squared_error(Test_y.values.ravel(), predictions)
        
        print("Logistic")
        print("MSE: ", mse)
        print("RMSE: ", np.sqrt(mse))
        print(model_logistic.coef_)              # cac he so beta 
        print(model_logistic.intercept_)         # he so chan tren 
        logistic_MSE.append(mse)
        logistic_RMSE.append(np.sqrt(mse))
        acc = metrics.accuracy_score(Test_y, predictions)    
        acc_logistic.append(acc)         
        
        start = datetime.now()
        model_logistic  = LogisticRegression()
        model_logistic.fit(Train_x, Train_y)
        score = model_logistic.score(Train_x, Train_y.values.ravel())
        predictions = model_logistic.predict(Test_x)
        end = datetime.now() -start
        time_logistic.append(end)
        
    print ("Accuracy Logistic: ", np.mean(acc_logistic))
    print ("Time Logistic_Rg: ", np.mean(time_logistic))
    print ("Logistic: MSE va RMSE: ", np.mean(logistic_MSE), ' : ',np.mean(logistic_RMSE))

    names =('Logistic') 
    results.append(logistic_MSE)
    results.append(logistic_MSE)
    results.append(logistic_MSE)
    results.append(logistic_MSE)
  
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()    
    
    return 
    
def main():   
    df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Data\\CNS1.csv' ) 
    df.columns.values[0] = "class"   
   # readData(df)    
    comparative(df)
   # run_compare(df)
    return

if __name__ == "__main__":
    main()
