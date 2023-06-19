import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def interpolation(X,Y):
    f_=np.zeros((len(Y),len(Y)))
    f_[0]=Y

    for i in range(len(Y)):
        for j in range(len(Y)-i-1):
            f_[i+1][j]=(f_[i][j+1]-f_[i][j])/(X[j+1+i]-X[j])

    return f_

def interpolation_(X,Y):
    f_=np.zeros((len(Y),len(Y)))
    f_[0]=Y

    for i in range(len(Y)):
        for j in range(len(Y)-i-1):
            f_[i+1][j]=(f_[i][j+1]-f_[i][j])/(X[j+1+i]-X[j])


    result=""
    for i in range(len(Y)):
        r=str(f_[len(Y)-i-1][0])
        for j in range(len(Y)-i-1):
            if(X[j]<0):
                 r=r+"(x" +str(abs(X[j]))+")"
            else:
                r=r+"(x-" +str(X[j])+")"
        result=result+"+"+r
        
    return result

def interpolation_t(X,Y, x):
    f_=np.zeros((len(Y),len(Y)))
    f_[0]=Y

    for i in range(len(Y)):
        for j in range(len(Y)-i-1):
            f_[i+1][j]=(f_[i][j+1]-f_[i][j])/(X[j+1+i]-X[j])


    result=0
    for i in range(len(Y)):
        r=f_[len(Y)-i-1][0]
        for j in range(len(Y)-i-1):
            r=r*(x- X[j])
        result=result+r
        
    return result

    



X=[-5000,-4000,-3000,-2000,-1000,0,1000,2000,3000,4000,5000,6000, 7000]
Y=[-135,-(17*110)/27,-40,-270/11,-10,0,400/60,90/11,10,110/9,15,130/7,70/3]
Z=[-48.46153846,-17.14285714,-17.14285714,-10,-10,0 , 4.285714286,4.285714286,4.285714286, 4.285714286,5.384615385, 5.384615385,6.666666667
]
print(interpolation(X,Z)[:,0])


x=np.arange(-5000,70000,500)
y=interpolation_t(X,Z,x)
#print(y)
plt.plot(X, Y)
plt.show()
