import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from tkinter import *
import tkinter as tk
root = tk.Tk()
root.geometry('600x600')
tk.Label(root, text="Kidney Disease Prediction", fg= "black", font = "Times 30 bold").pack()
root.title("Prediction")

def test1():
    dataset = pd.read_csv('kidney_prediction.csv')
    dataset.dropna()
    y = dataset['Class'].values
    x = dataset.drop(['Class'], axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)
    standardScaler = StandardScaler()
    columns_to_scale = ['Bp', 'Bu', 'Sod', 'Wbcc', 'Rbcc']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
    regressorsvm = SVC(kernel = 'linear')
    regressorsvm.fit(X_train,y_train)
    ys_pred = regressorsvm.predict(X_test)
    r2_score(y_test,ys_pred)
    regressorsvm.score(X_test,y_test)
    s=[int(c1.get()),float(c2.get()),int(c3.get()),int(c4.get()),int(c5.get()),int(c6.get()),float(c7.get()),float(c8.get()),float(c9.get()),float(c10.get()),int(c11.get()),float(c12.get()),int(c13.get())]
    s=np.array(s)
    predsvm=regressorsvm.predict(s.reshape(1,-1))
    str1 = "Using SVM Algorithm! You don't have kidney disease! You need not worry!"
    str2 = "Using SVM Algorithm! You have kidney disease! Consult Nephrologist Immediately!"
    if(predsvm[0] == 0):
        l14['text'] = str1;
    else:
        l14['text'] = str2;

def test2():
    dataset = pd.read_csv('kidney_prediction.csv')
    dataset.dropna()
    y = dataset['Class'].values
    x = dataset.drop(['Class'], axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)
    standardScaler = StandardScaler()
    columns_to_scale = ['Bp', 'Bu', 'Sod', 'Wbcc', 'Rbcc']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
    
    knn = KNeighborsClassifier(n_neighbors=7) 
    knn.fit(X_train, y_train) 
    ys_pred = knn.predict(X_test)
    r2_score(y_test,ys_pred)
    knn.score(X_test,y_test)
    s=[int(c1.get()),float(c2.get()),int(c3.get()),int(c4.get()),int(c5.get()),int(c6.get()),float(c7.get()),float(c8.get()),float(c9.get()),float(c10.get()),int(c11.get()),float(c12.get()),int(c13.get())]
    s=np.array(s)
    predknn=knn.predict(s.reshape(1,-1))
    str1 = "Using KNN Algorithm! You don't have kidney disease! You need not worry!"
    str2 = "Using KNN Algorithm! You have kidney disease! Consult Nephrologist Immediately!"
    if(predknn[0] == 0):
        l14['text'] = str1;
    else:
        l14['text'] = str2;
    
def test3():
    print("Welcome")

l1=Label(root,text='BP', font = "Times 13 bold")
l1.place(x=10,y=100)
l2=Label(root,text='SG', font = "Times 13 bold")
l2.place(x=10,y=130)
l3=Label(root,text='AL', font = "Times 13 bold")
l3.place(x=10,y=160)
l4=Label(root,text='SU', font = "Times 13 bold")
l4.place(x=10,y=190)
l5=Label(root,text='RBC', font = "Times 13 bold")
l5.place(x=10,y=220)
l6=Label(root,text='BU', font = "Times 13 bold")
l6.place(x=10,y=250)
l7=Label(root,text='SC', font = "Times 13 bold")
l7.place(x=10,y=280)
l8=Label(root,text='SOD', font = "Times 13 bold")
l8.place(x=10,y=310)
l9=Label(root,text='POT', font = "Times 13 bold")
l9.place(x=10,y=340)
l10=Label(root,text='HEMO', font = "Times 13 bold")
l10.place(x=10,y=370)
l11=Label(root,text='WBCC', font = "Times 13 bold")
l11.place(x=10,y=400)
l12=Label(root,text='RBCC', font = "Times 13 bold")
l12.place(x=10,y=430)
l13=Label(root,text='HTN', font = "Times 13 bold")
l13.place(x=10,y=460)
l14=Label(root,text=" ", fg= "black", font = "Times 25 bold")
l14.place(x=200,y=490)

c1s=IntVar()
c1=Entry(root,textvariable=c1s)
c1.place(x=150,y=100)
c2s=IntVar()
c2=Entry(root,textvariable=c2s)
c2.place(x=150,y=130)
c3s=IntVar()
c3=Entry(root,textvariable=c3s)
c3.place(x=150,y=160)
c4s=IntVar()
c4=Entry(root,textvariable=c4s)
c4.place(x=150,y=190)
c5s=IntVar()
c5=Entry(root,textvariable=c5s)
c5.place(x=150,y=220)
c6s=IntVar()
c6=Entry(root,textvariable=c6s)
c6.place(x=150,y=250)
c7s=IntVar()
c7=Entry(root,textvariable=c7s)
c7.place(x=150,y=280)
c8s=IntVar()
c8=Entry(root,textvariable=c8s)
c8.place(x=150,y=310)
c9s=IntVar()
c9=Entry(root,textvariable=c9s)
c9.place(x=150,y=340)
c10s=IntVar()
c10=Entry(root,textvariable=c10s)
c10.place(x=150,y=370)
c11s=IntVar()
c11=Entry(root,textvariable=c11s)
c11.place(x=150,y=400)
c12s=IntVar()
c12=Entry(root,textvariable=c12s)
c12.place(x=150,y=430)
c13s=IntVar()
c13=Entry(root,textvariable=c13s)
c13.place(x=150,y=460)


b1 = Button(root,text="Support Vector Regression", font = "Times 18 bold", height = 2, width = 30, command=test1)
b1.place(x=100, y=600)
b2 = Button(root,text="KNN Regression", font = "Times 18 bold", height = 2, width = 30, command=test2)
b2.place(x=600, y=600)
b3 = Button(root,text="Logistic Regression", font = "Times 18 bold", height = 2, width = 30, command=test3)
b3.place(x=1100, y=600)


root.mainloop()
