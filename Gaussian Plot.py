import matplotlib.pyplot as plt
from numpy import loadtxt,sqrt,pi,exp,prod,log

x_min,x_max = input("Enter Gaussian Parameters: x_min, x_max:")

#loading data and setting constants
a=loadtxt('A4Q1data.txt',float)
x=a[:,0]
y=a[:,1]
sigma = 0.3
n = len(y)
y_avg = (sum(y))/n

#search for the minimum chi-squared value,x_G then A snce given sigma
#x_G
for x_G in range (x_min, x_max):
    dev = x-x_G
    chi_squared = (sum(((dev)/sigma)**2))
    #check each chi squared value to see if its smallest
    if chi_squared < min_chi:
        chi_squared = min_chi
    print min_chi

#this creates a Gaussin Distrib. Plot, should overlay the data
def gaussian(x_peak, height_peak, width, x_min, x_max):
    return 

##PLOTTING
fig = plt.figure(1)
plt.plot(x,y,linestyle = 'none', marker = '.')#plots data as points
fig.suptitle('Phys 336 - Assignment 4 Gaussian Fit')
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('figure.png')
