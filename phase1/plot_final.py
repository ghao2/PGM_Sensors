import numpy as np;
import matplotlib;
import pandas as pd;
import matplotlib.pyplot as plt;

matplotlib.style.use('ggplot')

index_vector=np.array([0, 5, 10, 20, 25]);

for index in range(0, 5):

    index_input=index_vector[index];

    file_name='temp'+str(index_input)+'.txt';

    data=np.loadtxt(file_name, comments='#');

    print data.shape;

    xAxis=np.arange(10);
    yAxis1=data;

    plot1=plt.bar(left=xAxis, height=yAxis1, width=0.5);

    plot1[0].set_color('dimgrey')
    plot1[1].set_color('dimgrey')

    plot1[2].set_color('peru')
    plot1[3].set_color('peru')

    plot1[4].set_color('y')
    plot1[5].set_color('y')

    plot1[6].set_color('steelblue')
    plot1[7].set_color('steelblue')

    plot1[8].set_color('lime')
    plot1[9].set_color('lime')



    #add information to the graph
    plt.xlabel('different phase and windows');
    plt.ylabel('error');

    title_name='comparison of window method and variance method for temp at size '+str(index_input);

    plt.title(title_name);

    #move the ticks a little bit right.
    plt.xticks(xAxis,('p1 w','p1 v','p2 h-w','p2 h-v','p2 d-w',
                        'p2 d-v', 'p3 frd-w', 'p3 frd-v', 'p3 brd-w', 'p3 brd-v'));

    # #using legend to add label to the graph
    # plt.legend(handles=[plot1, plot2], labels=['window method','variance method']);

    plt.show();