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

    tempMatrix=tempTest.iloc[:, 0:49];

    returnRegHumiMatrix=copy.deepcopy(tempTest);
    returnMSEeMatrix=copy.deepcopy(tempMatrix);

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

            simpleLinear.fit(tVector, hVector);

            #convert the predicted array formatted value into a scalar
            returnRegHumiMatrix.iloc[i, j]=simpleLinear.predict(tempTest.iloc[i, j])[0];
            returnRegHumiMatrix.iloc[i, j+48]=simpleLinear.predict(tempTest.iloc[i, j+48])[0];

            #calculate MSE
            predictedHumiVector=simpleLinear.predict(tVector);

            MSE=calcMSE(hVector,predictedHumiVector);

            returnMSEeMatrix[i, j]=MSE;


    return returnRegHumiMatrix, returnMSEeMatrix;

def calcMSE(hVector, predicedHumiVector):

    length=len(hVector);

    sumOfError=0;

    for index in range(0, length):
        sumOfError=sumOfError+pow(hVector[index, 0]-predicedHumiVector[index, 0], 2);

    MSE=sumOfError/length;

    return MSE;

#calculate the absolute value between mean and an input value.
def calMAE(mean, inputValue):

    return np.abs(inputValue-mean);


#return a MAE matrix which has the same dimension with test data
def calMAEMatrix(humiRegMatrix, testData, returnMSEMatrix, numberOfVariance):

    length=len(humiRegMatrix);
    width=len(humiRegMatrix.iloc[0, :]);
    widthForReturn=len(testData.iloc[0, :]);

    returnMAEMatrix=copy.deepcopy(testData);

    print 'The shape of MSEMatrix is ', humiRegMatrix.shape;
    print 'The shape of testData is ', testData.shape;

    for i in range(0, length):
        #j is from 1 to 48, in terms of column position number.
        for j in range(1, width):

            returnMAEMatrix.iloc[i, j]=calMAE(humiRegMatrix.iloc[i, j], testData.iloc[i, j]);


    #now, make all positions that have highest variance to be zero, bacause zero means no error at that variable
    position=0;
    for index in range(1, widthForReturn):

        MSEMatrixIndex=index;

        #when outer index reach 50, inner index has to be set to 1
        if(MSEMatrixIndex>=49):
            MSEMatrixIndex=MSEMatrixIndex-48;

        #find all the positions at which the variables are of largest variance
        positionVector=markHighVariance(returnMSEMatrix.iloc[:, MSEMatrixIndex], numberOfVariance);

        for j in range(0, numberOfVariance):
            #set all the values of those postions that returned by positionVectors to be 0.
            returnMAEMatrix.iloc[positionVector[j], index]=0;


    #calclulate the overall error in the case of 'window method'.
    sum=returnMAEMatrix.iloc[0:50, 1:97].sum(axis=1).sum(axis=0);

    #overallError=sum/counter;
    overallError=sum/(length*(widthForReturn-1));

    return returnMAEMatrix, overallError;

def markHighVariance(inputVector, numberOfVariance):

    newVector=copy.deepcopy(inputVector);

    positionVector=np.arange(numberOfVariance)*0;

    length=len(newVector);

    for index in range(0, numberOfVariance):

        max=0;
        position=0;

        for j in range(0, length):
            if (newVector[j]>max):
                max=newVector[j];
                position=j;

        #store the max guy in the psitionVector, which will be returned in the end of this function.
        positionVector[index]=position;

        #after find a max one, delete it in the original array.
        newVector[position]=0;

    return positionVector;

humidity_train_data=pd.read_csv('humidity_train.csv');
humidity_test_data=pd.read_csv('humidity_test.csv');
temp_train_data=pd.read_csv('temp_train.csv');
temp_test_data=pd.read_csv('temp_test.csv');


humiRegMatrix, returnMSEeMatrix=calcRegressionMatrix(humidity_train_data, temp_train_data, temp_test_data);

print 'This is humidity regression matrix', humiRegMatrix;

sizeVector=np.array([0, 5, 10, 20, 25]);
errorVector=np.arange(5)*0.0;


for index in range(0,5):
    sizeOfWindow=sizeVector[index];
    returnMAEMatrix, overallError=calMAEMatrix(humiRegMatrix, humidity_test_data, returnMSEeMatrix, sizeOfWindow);

    #print 'This is mean for each variable \n', meanMatrix,'\n';
    #print 'This is variance for each variable \n', varianceMatrix, '\n'
    #print 'This is the Mean Absolute Error for each variable \n', maeMatrix;
    errorVector[index]=overallError;

print 'The MAE for budget in 0, 5, 10, 20, 25 is :',errorVector;

