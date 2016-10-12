import numpy as np;
import scipy as sp;
import pandas as pd;
import copy;
from sklearn import linear_model;

def calcRegressionMatrix(humidityTrain, temperatureTrain, tempTest):

    length=len(humidityTrain);
    width=len(humidityTrain.iloc[0, :]);

    #width with only the first day data.
    widthFirstDay=(width-1)/3+1;

    returnRegHumiMatrix=copy.deepcopy(tempTest);

    #using a linear model
    simpleLinear=linear_model.LinearRegression();

    for i in range(0, length):
        #j is from 1 to 48, in terms of column position number.
        for j in range(1, widthFirstDay):

            hVector=np.array([humidityTrain.iloc[i, j],humidityTrain.iloc[i, j+48],humidityTrain.iloc[i, j+96]]);

            #converted into column vector
            hVector=hVector[None].T;

            #print 'input  values are ', hVector;
            tVector=np.array([temperatureTrain.iloc[i, j],temperatureTrain.iloc[i, j+48],temperatureTrain.iloc[i, j+96]]);

            #converted into column vector
            tVector=tVector[None].T;

            #fit the model
            simpleLinear.fit(tVector, hVector);

            #convert the predicted array formatted value into a scalar
            returnRegHumiMatrix.iloc[i, j]=simpleLinear.predict(tempTest.iloc[i, j])[0];
            returnRegHumiMatrix.iloc[i, j+48]=simpleLinear.predict(tempTest.iloc[i, j+48])[0];

    return returnRegHumiMatrix;


#calculate the absolute value between mean and an input value.
def calMAE(mean, inputValue):

    return np.abs(inputValue-mean);

def calMAEMatrix(regMatrix, testData, sizeOfWindow):

    length=len(regMatrix);
    width=len(regMatrix.iloc[0, :]);
    widthForReturn=len(testData.iloc[0, :]);

    returnMAEMatrix=copy.deepcopy(testData);

    for i in range(0, length):
        #j is from 1 to 48, in terms of column position number.
        for j in range(1, width):

            returnMAEMatrix.iloc[i, j]=calMAE(regMatrix.iloc[i, j], testData.iloc[i, j]);


    #now, make all positions in a window to be zero, bacause zero means no error at that variable
    #the idea is use two variables to control the loop, which are position in a column and the count that
    #maintain the total number of one round to be 15.
    position=0;
    for index in range(1, widthForReturn):
        count=0;

        #the varaible count controls the window size, it can be changed as go.
        while (count<sizeOfWindow):

            #set the positions that have readings to be zero
            returnMAEMatrix.iloc[position, index]=0;

            #set the posiitons that have readings to be the real readings, this matrix is to be printed
            #returnPrintTable.iloc[position, index]=testData.iloc[position, index];

            #the variable position controls the position of a variable in a column of the datatable
            #the maximum value of position is 49, and minimum is 0, it increments as loop runs.
            position=position+1;

            #once the position reaches the last position of a certain column, it is refreshed to be 0.
            if (position==50):
                position=0;
            count=count+1;

    #calclulate the overall error in the case of 'window method'.
    sum=returnMAEMatrix.iloc[0:50, 1:97].sum(axis=1).sum(axis=0);

    #overallError=sum/counter;
    overallError=sum/(length*(widthForReturn-1));

    return returnMAEMatrix, overallError;

humidity_train_data=pd.read_csv('humidity_train.csv');
humidity_test_data=pd.read_csv('humidity_test.csv');
temp_train_data=pd.read_csv('temp_train.csv');
temp_test_data=pd.read_csv('temp_test.csv');


humiRegMatrix=calcRegressionMatrix(humidity_train_data, temp_train_data, temp_test_data);

print 'This is humidity regression matrix', humiRegMatrix;

sizeVector=np.array([0, 5, 10, 20, 25]);
errorVector=np.arange(5)*0.0;


for index in range(0,5):
    sizeOfWindow=sizeVector[index];
    returnMAEMatrix, overallError=calMAEMatrix(humiRegMatrix, humidity_test_data, sizeOfWindow);

    #print 'This is mean for each variable \n', meanMatrix,'\n';
    #print 'This is variance for each variable \n', varianceMatrix, '\n'
    #print 'This is the Mean Absolute Error for each variable \n', maeMatrix;
    errorVector[index]=overallError;

print 'The MAE for budget in 0, 5, 10, 20, 25 is :',errorVector;

