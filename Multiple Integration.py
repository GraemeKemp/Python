#need to integrate one at a time, use those values to integrate the next etc..
#gnh specifies the max value of x, integrate from 0 -> x_max

from numpy import exp
x_max = input('Specficify the maximum value for x to integrate over: ')

#constants
n=100 #number of intervals within the range to integrate
x_range=x_max/float(n)#size of the intervals
r_xmax = x_max*exp(-x_max**3)
y_range=(x_max**2)*exp(-2*x**3)/float(n)#constraining y and z to stop integration
z_range=(x_max**2)*exp(-2*x**3)/float(n)

total_x=0.#starts off the counters for area at zero
total_y=0.
total_z=0.

#define functions to integrate
def f_yz(y,z):
    return (y**2 + z**2)*(1+y+z**2)

#need to define a two functions such that int.f_yz = int.f_y*int.f_z, then they can be computed seperately
#integrate in y
for i in range(0,n):
    y_i=float(i)*y_range
    y_iplusone=y_i+y_range
    total_y=total_y+(f_yz(y_i,z)+f_yz(y_iplusone,z))
print 0.5*total_y*y_range

#integrate in z
for i in range(0,n):
    z_i=float(i)*z_range
    z_iplusone=z_i+z_range
    total_z=total_z+(f_yz(y,z_i)+f_yz(y,z_iplusone))
print 0.5*total_z*z_range

total = total_y*total_z#total 'area under the curve' is the volume of the integrated areas

print "Total moment of inertia = ", 0.5*total*x_range#x_range is simply the 'dx' portion of the integral

