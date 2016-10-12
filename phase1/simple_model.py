import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;
from sklearn.svm import SVC;
from sklearn import linear_model;
import matplotlib.pyplot as pl;
import pandas as pd;
import csv;

class Mean_vector:

    #raw data
    trainning_humidity=1;
    trainning_temp=1;
    testing_humidity=1;
    testing_temp=1;

    #special matrices
    indicator_matrix_for_window=1;
    indicator_matrix_for_variance_temp=1;
    indicator_matrix_for_variance_humidity=1;

    #dimension of some matrices
    trainning_set_width=1;
    testing_set_width=1;
    data_height=1;

    #structure of the network: betas
    #each row represents the structure of a sensor to other sensors
    structure_beta0_temp=1;
    structure_beta1_temp=1;
    structure_beta0_humidity=1;
    structure_beta1_humidity=1;

    #variance of each sensor:
    global_variance_for_sensors_temp=1;
    global_variance_for_sensors_humidity=1;

    #prediction matrices
    prediction_temp=1;
    prediction_humidity=1;

    #local variance matrices
    local_variance_for_sensors_temp=1;
    local_variance_for_sensors_humidity=1;


    #initialize all the static variables, except for special matrices and structure of the network and
    #variance of eahc sensor and predictions.
    def __init__(self):

        #initialize trainning_humidity
        training_h_df=pd.read_csv('humidity_train.csv');

        training_h_array=np.array(training_h_df);

        self.trainning_humidity=training_h_array[:, 1:];


        #initialize trainning_temp
        training_t_df=pd.read_csv('temp_train.csv');

        training_t_array=np.array(training_t_df);

        self.trainning_temp=training_t_array[:, 1:];


        #initialize testing_humidity
        testing_h_df=pd.read_csv('humidity_test.csv');

        testing_h_array=np.array(testing_h_df);

        self.testing_humidity=testing_h_array[:, 1:];


        #initialize testing_temp
        testing_t_df=pd.read_csv('temp_test.csv');

        testing_t_array=np.array(testing_t_df);

        self.testing_temp=testing_t_array[:, 1:];

        #initialize other parameters
        self.trainning_set_width=len(self.trainning_temp[0, :]);
        self.testing_set_width=len(self.testing_temp[0, :]);
        self.data_height=len(self.trainning_temp);

        self.local_variance_for_sensors_temp=np.zeros([self.data_height, self.testing_set_width]);
        self.local_variance_for_sensors_humidity=np.zeros([self.data_height, self.testing_set_width]);

        #for the convinience of computation, set all the dimension match the testing dataset
        self.structure_beta0_temp=np.zeros([self.data_height, self.testing_set_width]);
        self.structure_beta1_temp=np.zeros([self.data_height, self.testing_set_width]);
        self.structure_beta0_humidity=np.zeros([self.data_height, self.testing_set_width]);
        self.structure_beta1_humidity=np.zeros([self.data_height, self.testing_set_width]);

        self.global_variance_for_sensors_temp=np.zeros([self.data_height, self.testing_set_width]);
        self.global_variance_for_sensors_humidity=np.zeros([self.data_height, self.testing_set_width]);


        #initialize the prediction temp matrix
        self.prediction_temp=np.zeros([self.data_height, self.testing_set_width]);

        #initialize the prediction temp matrix
        self.prediction_humidity=np.zeros([self.data_height, self.testing_set_width]);

        return;



    def init_indicator_matrix_for_window(self, window_size):

        self.indicator_matrix_for_window=np.zeros([self.data_height, self.testing_set_width]);

        row_number=0;

        for index in range(0, self.testing_set_width):

            count=0;

            while (count<window_size):

                #make the variance of that column to be zero in the required window size.
                self.indicator_matrix_for_window[row_number, index]=1;

                row_number+=1;

                if (row_number==50):
                    row_number=0;

                count=count+1;

        return;

    # #find the top # elements and mark them as one, input is a matrix, and the size which is the top#.
    def init_indicator_matrix_for_variance_temp(self, size):

        self.indicator_matrix_for_variance_temp=np.zeros([self.data_height, self.testing_set_width]);

        #find all the top # elements in the matrix
        for column_index in range(0, self.testing_set_width):

            #find all the top # elements for each column
            count=0;

            #find 'size' of times.
            while(count<size):

                max_element=self.local_variance_for_sensors_temp[0, column_index];
                max_position=0;

                for max_index in range(0, self.data_height):

                    if self.local_variance_for_sensors_temp[max_index, column_index]>max_element:

                        max_element=self.local_variance_for_sensors_temp[max_index, column_index];
                        max_position=max_index;

                self.indicator_matrix_for_variance_temp[max_position, column_index]=1;

                #after find a top element, make it zero
                self.local_variance_for_sensors_temp[max_position, column_index]=0;

                count+=1;

        return;


    def init_indicator_matrix_for_variance_humidity(self, size):

        self.indicator_matrix_for_variance_humidity=np.zeros([self.data_height, self.testing_set_width]);

        #find all the top # elements in the matrix
        for column_index in range(0, self.testing_set_width):

            #find all the top # elements for each column
            count=0;

            #find 'size' of times.
            while(count<size):

                max_element=self.local_variance_for_sensors_humidity[0, column_index];
                max_position=0;

                for max_index in range(0, self.data_height):

                    if self.local_variance_for_sensors_humidity[max_index, column_index]>max_element:

                        max_element=self.local_variance_for_sensors_humidity[max_index, column_index];
                        max_position=max_index;

                self.indicator_matrix_for_variance_humidity[max_position, column_index]=1;

                #after find a top element, make it zero
                self.local_variance_for_sensors_humidity[max_position, column_index]=0;

                count+=1;


        return;

    def learning_structure_temp(self):

        for sensor_index in range(0, self.data_height):

            for column_index in range(0, 48):

                input_x=np.array([self.trainning_temp[sensor_index, column_index],
                                  self.trainning_temp[sensor_index, column_index+48],
                                  self.trainning_temp[sensor_index, column_index+96]])

                prediction_value=np.mean(input_x);

                self.prediction_temp[sensor_index, column_index]=prediction_value;
                self.prediction_temp[sensor_index, column_index+48]=prediction_value;

                prediction_variance=np.var(input_x);

                self.local_variance_for_sensors_temp[sensor_index, column_index]=prediction_variance;
                self.local_variance_for_sensors_temp[sensor_index, column_index+48]=prediction_variance;

        return;


    def learning_structure_humidity(self):

        for sensor_index in range(0, self.data_height):

            for column_index in range(0, 48):

                input_x=np.array([self.trainning_humidity[sensor_index, column_index],
                                  self.trainning_humidity[sensor_index, column_index+48],
                                  self.trainning_humidity[sensor_index, column_index+96]])

                prediction_value=np.mean(input_x);

                self.prediction_humidity[sensor_index, column_index]=prediction_value;
                self.prediction_humidity[sensor_index, column_index+48]=prediction_value;

                prediction_variance=np.var(input_x);

                self.local_variance_for_sensors_humidity[sensor_index, column_index]=prediction_variance;
                self.local_variance_for_sensors_humidity[sensor_index, column_index+48]=prediction_variance;

        return;



    def inference_temp_window(self):

        #print 'initialized prediction_temp', self.prediction_temp;

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(0, self.testing_set_width):

            #infrerence on row number
            for sensor_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_window[sensor_index, column_index]==1:

                    self.prediction_temp[sensor_index, column_index]=self.testing_temp[sensor_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;

    def inference_humidity_window(self):

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(0, self.testing_set_width):

            #infrerence on row number
            for sensor_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_window[sensor_index, column_index]==1:

                    self.prediction_humidity[sensor_index, column_index]=self.testing_humidity[sensor_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;


    def inference_temp_variance(self):


        #inference the whole test table, predicting starts from the second column.
        for column_index in range(0, self.testing_set_width):

            #infrerence on row number
            for sensor_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_variance_temp[sensor_index, column_index]==1:

                    self.prediction_temp[sensor_index, column_index]=self.testing_temp[sensor_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;



    def inference_humidity_variance(self):


        #inference the whole test table, predicting starts from the second column.
        for column_index in range(0, self.testing_set_width):

            #infrerence on row number
            for sensor_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_variance_humidity[sensor_index, column_index]==1:

                    self.prediction_humidity[sensor_index, column_index]=self.testing_humidity[sensor_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;



    def calculate_MAE_temp(self):

        total_number_of_element=self.data_height*self.testing_set_width;

        error_matrix=abs(self.testing_temp-self.prediction_temp);

        total_error=np.sum(error_matrix);

        MAE=total_error/total_number_of_element;

        return MAE;


    def calculate_MAE_humidity(self):

        total_number_of_element=self.data_height*self.testing_set_width;

        error_matrix=abs(self.testing_humidity-self.prediction_humidity);

        total_error=np.sum(error_matrix);

        MAE=total_error/total_number_of_element;

        return MAE;



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

    cls=Mean_vector();

    #change the window size
    cls.init_indicator_matrix_for_window(index_input);

    cls.learning_structure_temp();
    cls.learning_structure_humidity();

    cls.inference_temp_window();
    cls.inference_humidity_window();

    print 'size is ', index_input;

    print 'MAE for window, temp, is ',cls.calculate_MAE_temp();
    print 'MAE for window, humidity, is ',cls.calculate_MAE_humidity();

    # id=str(index_input);
    #
    # fileName1='tw'+id+'.csv';
    # fileName2='w'+id+'.csv';
    #
    # #write_file(cls.prediction_temp, fileName1);
    # write_file(cls.prediction_humidity, fileName2);


    #learn again, and re-inference

    cls.learning_structure_temp();
    cls.learning_structure_humidity();

    cls.init_indicator_matrix_for_variance_temp(index_input)
    cls.init_indicator_matrix_for_variance_humidity(index_input)

    #calc for temp and humidity
    cls.inference_temp_variance();
    cls.inference_humidity_variance();


    print 'MAE for variance, temp, is ',cls.calculate_MAE_temp();
    print 'MAE for variance, humidity, is ',cls.calculate_MAE_humidity();


    # fileName3='tv'+id+'.csv';
    # fileName4='v'+id+'.csv';
    #
    # #write_file(cls.prediction_temp, fileName3);
    # write_file(cls.prediction_humidity, fileName4);

