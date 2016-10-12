import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;
from sklearn.svm import SVC;
from sklearn import linear_model;
import matplotlib.pyplot as pl;
import pandas as pd;
import csv;





def write_file(dataSet, index_number):


    header=np.zeros([1, 96]);

    for index in range(0,48):

        header[0,index]=0.5+index*0.5;
        header[0,index+48]=0.5+index*0.5;

    dataSet=np.concatenate((header, dataSet), axis=0);

    extra_column=np.zeros([51,1]);

    for index_extra_column in range(0, 51):

        extra_column[index_extra_column,0]=index_extra_column;

    dataSet=np.concatenate((extra_column, dataSet), axis=1);

    fileName=index_number;

    with open(fileName, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(dataSet);

    return;


index_vector=np.array([0, 5, 10, 20, 25]);

for index in range(0, 5):

    index_input=index_vector[index];


    file_name1='v'+str(index_input)+'.csv';

    training_h_array=np.loadtxt(file_name1, delimiter=',');

    print training_h_array.shape;

    write_file(training_h_array, file_name1);


    file_name2='w'+str(index_input)+'.csv';

    training_h_array=np.loadtxt(file_name2, delimiter=',');

    write_file(training_h_array, file_name2);