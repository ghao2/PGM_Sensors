import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib;

matplotlib.style.use('ggplot')


data=np.loadtxt('temperature_compare.txt', comments='#');

xAxis=data[0,:];
yAxis1=data[1,:];
yAxis2=data[2,:];

plot1=plt.bar(left=xAxis, height=yAxis1, width=2, color='lightsteelblue');
plot2=plt.bar(left=xAxis+2, height=yAxis2, width=2, color='lime');

#add information to the graph
plt.xlabel('budget');
plt.ylabel('error');
plt.title('comparison of window method and variance method for temperature');

#move the ticks a little bit right.
plt.xticks(xAxis+2,('0','5','10','20','25'));

#using legend to add label to the graph
plt.legend(handles=[plot1, plot2], labels=['window method','variance method']);

plt.show();