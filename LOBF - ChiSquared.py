from numpy import loadtxt
import matplotlib.pyplot as plt
from numpy import linspace,sqrt,pi,exp,prod

a=loadtxt('A3.txt',float)#CHANGE FILE ON GNH
x=a[:,0]
y=a[:,1]
sigma=a[:,2]
N = len(x)#degrees of freedom
x_avg = (sum(x))/N #average x value

##FIT
delta = ((sum(1/sigma**2))*(sum(x**2/sigma**2)))-(sum(x/sigma**2))**2
a=((1/delta)*((sum(1/sigma**2)*sum(x*y/sigma**2))-((sum(x/sigma**2)*sum(y/sigma**2)))))
b=((1/delta)*((sum(x**2/sigma**2)*(sum(y/sigma**2)))-(sum(x/sigma**2)*(sum(x*y/sigma**2)))))

##UNCERTAINTIES
sigma_a = sqrt((1/delta)*(sum(x**2/sigma**2)))#uncertainties in x a and b coefficients (sigma)
sigma_b = sqrt((1/delta)*(sum(1/sigma**2)))
model=a*x+ b #equation for the 'model' i.e. line of best fit for the data
dev_mod = y-model #devaition of each value from the predicted model
chi_squared = (sum(((dev_mod)/sigma)**2))/(N-2) #reduced chi-squared value, degrees of freedom minus paratmers fitted
p_x = prod(((sigma*(sqrt(2*pi)))**(-1.))*(exp(-((dev_mod**2)/(2*(sigma))))))#probability for random data to have this fit

print 'a =', a #printing the required answers
print 'b =', b
print 'sigma_a = ', sigma_a 
print 'sigma_b = ', sigma_b
print 'Reduced Chi Squared = ', chi_squared
print 'P(x) = ', p_x
print (sum((dev_mod/sigma)**2))/(N-2)

##PLOTTING
#fig = plt.figure(1)
#plt.plot(x,y,linestyle = 'none', marker = 'o')#plots data as points
#plt.plot(x,model) #plots the predicted model
#fig.suptitle('Phys 336 - Assignment 3 Data')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()#DONT INCULDE IN FINAL CODE
#fig.savefig('figure.png')
