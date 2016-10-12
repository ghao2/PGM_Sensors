import numpy as np;
import matplotlib;
import pandas as pd;
import matplotlib.pyplot as plt;

matplotlib.style.use('ggplot')

data=np.loadtxt('temp_reg_nonreg.txt', comments='#');

xAxis=data[0,:];
yAxis1=data[1,:];
yAxis2=data[2,:];
yAxis3=data[3,:];

plot1=plt.bar(left=xAxis, height=yAxis1, width=1, color='lime');
plot2=plt.bar(left=xAxis+1, height=yAxis2, width=1, color='lightslategrey');
plot3=plt.bar(left=xAxis+2, height=yAxis3, width=1, color='steelblue');

#add information to the graph
plt.xlabel('budget');
plt.ylabel('error');
plt.title('comparison of window, variance and regression method for temp');

#move the ticks a little bit right.
plt.xticks(xAxis+2,('0','5','10','20','25'));

#using legend to add label to the graph
plt.legend(handles=[plot1, plot2, plot3], labels=['regression method', 'window method','variance method']);

plt.show();