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
    structure_matrix_temp=1;
    structure_matrix_humidity=1;

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

        #initialize the first column
        self.local_variance_for_sensors_temp[:,0]=self.init_first_column_variance_temp();

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


            #inference next column's variance
            if column_index < self.testing_set_width-1:

                for sensor_index in range(0, self.data_height):

                    #get structure
                    beta_structure=self.structure_matrix_temp[sensor_index, 1:];

                    beta_structure_square=beta_structure**2;

                    current_variance_vector=self.local_variance_for_sensors_temp[:, column_index]

                    var_next_new_sensor=self.global_variance_for_sensors_temp[sensor_index]+\
                                        np.dot(beta_structure_square, current_variance_vector);

                    self.local_variance_for_sensors_temp[sensor_index, column_index+1]=var_next_new_sensor;


        return;

    def init_indicator_matrix_for_variance_humidity(self, size):

        #initialize the first column
        self.local_variance_for_sensors_humidity[:,0]=self.init_first_column_variance_humidity();

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


            #inference next column's variance
            if column_index < self.testing_set_width-1:

                for sensor_index in range(0, self.data_height):

                    #get structure
                    beta_structure=self.structure_matrix_humidity[sensor_index, 1:];

                    beta_structure_square=beta_structure**2;

                    current_variance_vector=self.local_variance_for_sensors_humidity[:, column_index]

                    var_next_new_sensor=self.global_variance_for_sensors_humidity[sensor_index]+\
                                        np.dot(beta_structure_square, current_variance_vector);

                    self.local_variance_for_sensors_humidity[sensor_index, column_index+1]=var_next_new_sensor;


        return;




    # def init_local_variance_matrix_temp(self):
    #
    #     #initialize the first column of local variance matrix
    #     self.local_variance_for_sensors_temp[:,0]=self.global_variance_for_sensors_temp;
    #
    #     #iterate through variance matrix until the end of column
    #     for column_index in range(1, self.testing_set_width):
    #
    #         for row_index in range(0, self.data_height):
    #
    #             beta_vector=self.structure_matrix_temp[row_index, 1:];
    #
    #             variance_vector=self.local_variance_for_sensors_temp[:, column_index-1];
    #
    #             var_this_sensor=self.global_variance_for_sensors_temp[row_index]+np.dot(beta_vector**2, variance_vector);
    #
    #             self.local_variance_for_sensors_temp[row_index, column_index]=var_this_sensor;
    #
    #
    #     return;

    # def init_local_variance_matrix_humidity(self):
    #
    #     #initialize the first column of local variance matrix
    #     self.local_variance_for_sensors_humidity[:,0]=self.global_variance_for_sensors_humidity;
    #
    #     #iterate through variance matrix until the end of column
    #     for column_index in range(1, self.testing_set_width):
    #
    #         for row_index in range(0, self.data_height):
    #
    #             beta_vector=self.structure_matrix_humidity[row_index, 1:];
    #
    #             variance_vector=self.local_variance_for_sensors_humidity[:, column_index-1];
    #
    #             var_this_sensor=self.global_variance_for_sensors_humidity[row_index]+np.dot(beta_vector**2, variance_vector);
    #
    #             self.local_variance_for_sensors_humidity[row_index, column_index]=var_this_sensor;
    #
    #
    #     return;


    def learning_structure_temp(self):

        #structure must include intercept of the model, put it at the 0 position of a row
        self.structure_matrix_temp=np.zeros([self.data_height, self.data_height+1]);
        self.global_variance_for_sensors_temp=np.arange(self.data_height)*0.0;

        count=0;
        #find the structure for all sensors
        for sensor_index in range(0, self.data_height):

            #the learning_matrix has a height which is equal to number of columns of the training set - 1, because
            #the last column will not be used for training
            #the width of it should be data_height+2, where the last column is the output variable, which is THE 'y'.
            #the first column should be 1's.


            #at step 1
            learning_matrix=np.zeros([self.trainning_set_width-1, self.data_height+1]);

            #update the input_x and y.
            for index in range(0, self.trainning_set_width-1):

                learning_matrix[index, :-1]=self.trainning_temp[:,index];

                learning_matrix[index, -1]=self.trainning_temp[sensor_index, index+1];


            #write_file(learning_matrix,sensor_index);



            #training using linear regression, lasso
            clf=linear_model.Lasso(alpha=0.1, max_iter=1000);

            clf.fit(learning_matrix[:,:-1], learning_matrix[:, -1]);

            # #training using linear regression
            # clf=linear_model.LinearRegression();
            # clf.fit(learning_matrix[:,:-1], learning_matrix[:, -1]);

            #create theta vector
            coef=clf.coef_.tolist();
            coef.insert(0, clf.intercept_);

            coef=np.array(coef);

            #get the structure of the sensor and assign it to structure matrix
            self.structure_matrix_temp[sensor_index, :]=coef;

            #get the variance for this sensor
            SSE=0;

            #calculate SSE
            for sse_index in range(0, len(learning_matrix)):

                input_X=learning_matrix[sse_index, :-1];

                #add 1 to the beginning of the array.
                input_X=input_X.tolist();

                input_X.insert(0, 1);

                input_X=np.array(input_X);

                #calculate y_hat and y
                y_hat=np.dot(coef, input_X);
                y=learning_matrix[sse_index, -1];

                #accumulate SSE
                SSE=SSE+(y_hat-y)**2;

            #calculate variance, variance = SSE/n-k-1
            variance_for_this_sensor=SSE/(len(learning_matrix)-len(clf.coef_)-1);

            #get the variance of this sensor and store it into global variance;
            self.global_variance_for_sensors_temp[sensor_index]=variance_for_this_sensor;

            count+=1;


        return;


    def learning_structure_humidity(self):

        #structure will not include the intercept of the model
        self.structure_matrix_humidity=np.zeros([self.data_height, self.data_height+1]);
        self.global_variance_for_sensors_humidity=np.arange(self.data_height)*0.0;

        count=0;
        #find the structure for all sensors
        for sensor_index in range(0, self.data_height):

            #the learning_matrix has a height which is equal to number of columns of the training set - 1, because
            #the last column will not be used for training
            #the width of it should be data_height+2, where the last column is the output variable, which is THE 'y'.
            #the first column should be 1's.


            #at step 1
            learning_matrix=np.zeros([self.trainning_set_width-1, self.data_height+1]);

            #update the input_x and y.
            for index in range(0, self.trainning_set_width-1):

                learning_matrix[index, :-1]=self.trainning_humidity[:,index];

                learning_matrix[index, -1]=self.trainning_humidity[sensor_index, index+1];


            #write_file(learning_matrix,sensor_index);



            #training using linear regression, lasso
            clf=linear_model.Lasso(alpha=0.1, max_iter=1000);

            clf.fit(learning_matrix[:,:-1], learning_matrix[:, -1]);

            # #training using linear regression
            # clf=linear_model.LinearRegression();
            # clf.fit(learning_matrix[:,:-1], learning_matrix[:, -1]);


            #create theta vector
            coef=clf.coef_.tolist();
            coef.insert(0, clf.intercept_);

            coef=np.array(coef);

            #get the structure of the sensor and assign it to structure matrix
            self.structure_matrix_humidity[sensor_index, :]=coef;

            #get the variance for this sensor
            SSE=0;

            #calculate SSE
            for sse_index in range(0, len(learning_matrix)):

                input_X=learning_matrix[sse_index, :-1];

                #add 1 to the beginning of the array.
                input_X=input_X.tolist();

                input_X.insert(0, 1);

                input_X=np.array(input_X);

                #calculate y_hat and y
                y_hat=np.dot(coef, input_X);
                y=learning_matrix[sse_index, -1];

                #accumulate SSE
                SSE=SSE+(y_hat-y)**2;

            #calculate variance, variance = SSE/n-k-1
            variance_for_this_sensor=SSE/(len(learning_matrix)-len(clf.coef_)-1);

            #get the variance of this sensor and store it into global variance;
            self.global_variance_for_sensors_humidity[sensor_index]=variance_for_this_sensor;

            count+=1;


        return;

    def inference_temp_window(self):

        #initialize the prediction temp matrix
        self.prediction_temp=np.zeros([self.data_height, self.testing_set_width]);

        #initialize the first column
        self.prediction_temp[:,0]=self.init_input_first_column_temp();

        #print 'unitialized prediction_temp', self.prediction_temp;

        #initialize window
        for window_indicator_index in range(0, self.data_height):

            if self.indicator_matrix_for_window[window_indicator_index,0]==1:

                self.prediction_temp[window_indicator_index, 0]=self.testing_temp[window_indicator_index, 0];

        #print 'initialized prediction_temp', self.prediction_temp;

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(1, self.testing_set_width):

            #infrerence on row number
            for row_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_window[row_index, column_index]==0:

                    theta=self.structure_matrix_temp[row_index, :];

                    #initialized input_x
                    input_x=self.prediction_temp[:, column_index-1];

                    input_x=input_x.tolist();

                    input_x.insert(0,1);

                    input_x=np.array(input_x);

                    #calculate prediction
                    self.prediction_temp[row_index, column_index]=np.dot(theta, input_x);
                else:

                    self.prediction_temp[row_index, column_index]=self.testing_temp[row_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;

    def inference_humidity_window(self):

        #initialize the prediction temp matrix
        self.prediction_humidity=np.zeros([self.data_height, self.testing_set_width]);

        #initialize the first column
        self.prediction_humidity[:,0]=self.init_input_first_column_humidity();

        #print 'unitialized prediction_temp', self.prediction_temp;

        #initialize window
        for window_indicator_index in range(0, self.data_height):

            if self.indicator_matrix_for_window[window_indicator_index,0]==1:

                self.prediction_humidity[window_indicator_index, 0]=self.testing_humidity[window_indicator_index, 0];

        #print 'initialized prediction_temp', self.prediction_temp;

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(1, self.testing_set_width):

            #infrerence on row number
            for row_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_window[row_index, column_index]==0:

                    theta=self.structure_matrix_humidity[row_index, :];

                    #initialized input_x
                    input_x=self.prediction_humidity[:, column_index-1];

                    input_x=input_x.tolist();

                    input_x.insert(0,1);

                    input_x=np.array(input_x);

                    #calculate prediction
                    self.prediction_humidity[row_index, column_index]=np.dot(theta, input_x);
                else:

                    self.prediction_humidity[row_index, column_index]=self.testing_humidity[row_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;



    def inference_temp_variance(self):

        #initialize the prediction temp matrix
        self.prediction_temp=np.zeros([self.data_height, self.testing_set_width]);

        #initialize the first column
        self.prediction_temp[:,0]=self.init_input_first_column_temp();

        #print 'unitialized prediction_temp', self.prediction_temp;

        #initialize window
        for variance_indicator_index in range(0, self.data_height):

            if self.indicator_matrix_for_variance_temp[variance_indicator_index,0]==1:

                self.prediction_temp[variance_indicator_index, 0]=self.testing_temp[variance_indicator_index, 0];

        #print 'initialized prediction_temp', self.prediction_temp;

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(1, self.testing_set_width):

            #infrerence on row number
            for row_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_variance_temp[row_index, column_index]==0:

                    theta=self.structure_matrix_temp[row_index, :];

                    #initialized input_x
                    input_x=self.prediction_temp[:, column_index-1];

                    input_x=input_x.tolist();

                    input_x.insert(0,1);

                    input_x=np.array(input_x);

                    #calculate prediction
                    self.prediction_temp[row_index, column_index]=np.dot(theta, input_x);
                else:

                    self.prediction_temp[row_index, column_index]=self.testing_temp[row_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;



    def inference_humidity_variance(self):

        #initialize the prediction temp matrix
        self.prediction_humidity=np.zeros([self.data_height, self.testing_set_width]);

        #initialize the first column
        self.prediction_humidity[:,0]=self.init_input_first_column_humidity();

        #print 'unitialized prediction_temp', self.prediction_temp;

        #initialize window
        for variance_indicator_index in range(0, self.data_height):

            if self.indicator_matrix_for_variance_humidity[variance_indicator_index,0]==1:

                self.prediction_humidity[variance_indicator_index, 0]=self.testing_humidity[variance_indicator_index, 0];

        #print 'initialized prediction_temp', self.prediction_temp;

        #inference the whole test table, predicting starts from the second column.
        for column_index in range(1, self.testing_set_width):

            #infrerence on row number
            for row_index in range(0, self.data_height):

                #check window, if ==0, make prediction, if not, take reading.
                if self.indicator_matrix_for_variance_humidity[row_index, column_index]==0:

                    theta=self.structure_matrix_humidity[row_index, :];

                    #initialized input_x
                    input_x=self.prediction_humidity[:, column_index-1];

                    input_x=input_x.tolist();

                    input_x.insert(0,1);

                    input_x=np.array(input_x);

                    #calculate prediction
                    self.prediction_humidity[row_index, column_index]=np.dot(theta, input_x);
                else:

                    self.prediction_humidity[row_index, column_index]=self.testing_humidity[row_index, column_index];

            #print 'updated prediction is:\n',self.prediction_temp;

        return;




    def init_input_first_column_temp(self):

        mean_column=np.arange(self.data_height)*0.0;

        for index in range(0, self.data_height):

            mean_column[index]=np.mean(self.trainning_temp[index,:]);

        return mean_column;



    def init_input_first_column_humidity(self):

        mean_column=np.arange(self.data_height)*0.0;

        for index in range(0, self.data_height):

            mean_column[index]=np.mean(self.trainning_humidity[index,:]);

        return mean_column;



    def init_first_column_variance_temp(self):

        variance_column=np.arange(self.data_height)*0.0;

        for index in range(0, self.data_height):

            variance_column[index]=np.var(self.trainning_temp[index,:]);

        return variance_column;


    def init_first_column_variance_humidity(self):

        variance_column=np.arange(self.data_height)*0.0;

        for index in range(0, self.data_height):

            variance_column[index]=np.var(self.trainning_humidity[index,:]);

        return variance_column;



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

    print 'size is', index_input;
    print 'MAE for window, temp, is ',cls.calculate_MAE_temp();
    print 'MAE for window, humidity, is ',cls.calculate_MAE_humidity();

    # id=str(index_input);
    #
    # fileName1='tw'+id+'.csv';
    # fileName2='w'+id+'.csv';
    #
    # #write_file(cls.prediction_temp, fileName1);
    # write_file(cls.prediction_humidity, fileName2);


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


    # training_h_df=pd.read_csv('w5.csv');
    #
    # training_h_array=np.array(training_h_df);
    #
    # training_h_array=training_h_array[:, 1:];
    #
    # print training_h_array.shape;
    # print cls.prediction_humidity.shape;
    #
    # mae=cls.prediction_humidity-training_h_array;
    #
    # print 'test mae at 0',np.sum(mae);