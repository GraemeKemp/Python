#-------------------------------FIBONACCI-------------------------------------
n=input('How many Fibonacci numbers > ')
f1=1
print 1,f1
f2=1
print 2,f2
n=n-2
i=3
while n>0:
    next=f1+f2  # next fib numb
    print i,next
# reset numbers (last 2 Fib, plus 2 counters)
    f1=f2
    f2=next
    i=i+1
    n=n-1
#-----------------------------------------------------------------------------


#---------------------------FACTORIAL-----------------------------------------
n = input("Put in a number: ")

def factorial(n):
    if n==1:
        return 1
    else:
        return n*factorial(n-1)
print factorial(n)
#-----------------------------------------------------------------------------


#-------------------------------INTEGRATION-----------------------------------
#RECTANGLES
from numpy import sin,pi#this does sin(x) - we can put in any function(see Mid)
n=input("number of intervals >")
h=pi/float(n)
total=0.
for i in range(0,n):
    x_mid=(0.5+float(i))*h
    total=total+sin(x_mid)
print total*h,(total*h-2.)/2.0

#SIMPSONS
from numpy import sin,pi
n=input('number of intervals >')
h=pi/float(n)
print 'h=',h
# first element (with coefficent 1/3) is zero!
total=0.
for i in range(1,n-1):
    x_i=float(i)*h
    function=sin(x_i)
    if i%2 == 0:
        total=total+4.*function
    else:
        total=total+2.*function
print total*h/3.,((total*h/3.)-2.)/2.0

#TRAPEZOIDS
from numpy import sin,pi
n=input('number of intervals >')
h=pi/float(n)
total=0.
for i in range(0,n):
    x_i=float(i)*h
    x_iplusone=x_i+h
    total=total+(sin(x_i)+sin(x_iplusone))

print 0.5*total*h,(0.5*total*h-2.)/2.0
#---------------------------------------------------------------------------


#-----------------------------MULTIPLE INTEGRATION--------------------------
from numpy import arange,exp
from math import sqrt
def rmax(x):
    return (x*exp(-x**3))
def integrand(y,z):
    rx_sq=y*y+z*z
    return(rx_sq*(1.+y+z*z))
#choose approximate sizes of intervals
hx=0.01
hy=0.001
hz=0.001
xmax=input('xmax? ')
total=0.
# find number of slices along x axis
nx=int((xmax)/hx)
# choose an hx that gives integral over entire range of x
# (less important for x, but more important for y and z below)
hx=xmax/float(nx)

# start 'arange' at hx because at x=0 rmax will be zero and integral
#  in y and z will be zero; no need to do that part of integral
sumx=0.
for x in arange(hx,xmax+hx,hx):
# find number of strips along y axis and the size (hy_actual)
#  of each of these strips
    ymax=rmax(x)
    ny=int((2.*ymax)/hy)
    if ny<1:
        ny=1
    hy_actual=2*ymax/float(ny)

# iterate through all of the y strips
    sumy=0.
    for y in arange(-ymax,ymax+hy_actual,hy_actual):
# finder number of boxes along z axis in this y strip
        zmax_sq=ymax**2-y**2
        if zmax_sq>0:
            zmax=sqrt(zmax_sq)
            nz=int(2*zmax/hz)
            if nz<1:
                nz=1
            hz_actual=2*zmax/float(nz)
# iterate through the z boxes
#   note the *0.99  in the next line - the code was often adding one
#   extra box to the sum, probably due to rounding error.  this fixes it.
            sumz=0.
            for z in arange(-zmax,zmax+hz_actual*0.99,hz_actual):
#       note the 'if' statement checks if this term is an endpoint
#       (and therefore only gets half the weight of the other terms)
                if z<-zmax+hz_actual/2. or z>zmax-hz_actual/2.:
                    sumz=sumz+integrand(y,z)/2.
                else:
                    sumz=sumz+integrand(y,z)
            sumz=sumz*hz_actual
# add the integral of all of the boxes in this strip to to sum of all strips
# note the half-weight given to the end points
            if y<-ymax+hy_actual/2. or y>ymax-hy_actual/2.:
                sumy=sumy+sumz/2.
            else:
                sumy=sumy+sumz
    sumy=sumy*hy_actual
# the endpoints only get half weight
# (don't even chech the x=0 - integral is zero there)
    if x>xmax-hx/2.:
        sumx=sumx+sumy/2.
    else:
        sumx=sumx+sumy
sumx=sumx*hx
print sumx
#---------------------------------------------------------------------------


#--------------------------SOLVING DE'S EXAMPLE-----------------------------
dt=0.001#set a time interval,constants

def f(r,t):#define a function for the system of DE's (may need to decouple),might not need time
#EXAMPLE 2 - two coupled 1st ODE solved together
    x=r[0]#sets the variables in our vector 'r'
    v=r[1]#decoupled below
    dx=v #dx/dt=velocity
    dv=-x*k/m#dv/dt=acceleration (mass-spring damper)
    return array([dx,dv],float)#ALWAYS NEED THIS LINE
#----------------------------------------------------------------------------


#----------------------General Runge-Kutta Function---------------------------
#initial conditions and time interval (r = [x,y,z])
r = array([0.0,1.0,0.0],float)#initial conditions for the vector

#pre-allocating space for the final answers
tpoints = arange(t0,t_max,dt)#need to set time range, and dt
xpoints = []#these can be called in the plot function
ypoints = []
zpoints = []

#Runge-Kutta Function
for t in tpoints:
    xpoints.append(r[0])#adds the solution to each DE to the appropriate arrays
    ypoints.append(r[1])#can set as anything (velocity etc.)
    zpoints.append(r[2])
    k1 = dt*f(r, t)
    k2 = dt*f(r + 0.5*k1,t+0.5*dt)
    k3 = dt*f(r + 0.5*k2,t+0.5*dt)
    k4 = dt*f(r + k3,t+dt)
    r  += (k1+2*k2+2*k3+k4)/6.0
#-----------------------------------------------------------------------------


#----------------------SHOOTING METHOD MAIN PROGRAM---------------------------
def finalheight(vstart):#define a function to loop rk4
    r=array([0.0,vstart],float)
    #INSERT RK4 HERE 
    
bestheight = 1000.#boundary value
bestvelocity = 0.0#guess at initial condition
for vstart in arange(1.,100.,2.0):
    height = finalheight(vstart)#solution to DE given initial guess (solved w rk4)
    if abs(height)<bestheight:#if it doesn't reach the required BC
        bestheight = abs(height)#reset values, redo loop
        bestvelocity = vstart
print 'v=', bestvelocity, 'gives final height=',bestheight#else print
#-----------------------------------------------------------------------------


#--------------------------GAUSSIAN ELIMINATION-------------------------------
from numpy import array
from numpy.linalg import solve

A=array([[2,1],[1,0]],float)#sets up two matrices
v = array([-4,3],float)

x = solve(A,v)#solves Ax=v
#-----------------------------------------------------------------------------


#---------------------------RELAXATION METHOD---------------------------------
from math import exp
N = 16
x = 1.0
for i in range(N):#subs in new 'x' each time to converge on the answer
    x = 2 - exp(-x)
print x#prints x after the last loop
#-----------------------------------------------------------------------------


#-----------------------------NEWTON RHAPSON METHOD----------------------------
def f(x):#defined function
    return (5. - x**2.)
def fprime(x):#known derivative
    return (-2.*x)
accuracy = 1.0e-12#desired accuracy
x = 0.1#initial guess @ root
delta = 1.0#starting accuracy guess

while abs(delta)>accuracy:#GOOD WHILE LOOP
    delta = f(x)/fprime(x)
    x = x-delta
print x
#-----------------------------------------------------------------------------


#---------------------------------SECANT METHOD-------------------------------
def f(x):#defined function
    return (5. - x**2.)
accuracy = 1.0e-12#desired accuracy
x1 = 0.1#initial guesses
f1=f(x1)#ddefined function at initial guess
x2 = 0.2
f2 = f(x2)
delta = 1.0#starting accuracy guess

while abs(delta)>accuracy:
    delta = f2*(x2-x1)/(f2-f1)
    x3 = x2-delta
    x1=x2
    f1=f2
    x2=x3
    f2=f(x2)
print x3, delta
#-----------------------------------------------------------------------------


#-----------------------------RANDOM NUMBERS--------------------------------
from pylab import plot, show
N=1000
a=1664625
c=1013904223
m=4294967296
x=1
results = []#pre-allocated space
for i in range(N):
    x = (a*x +c)%m#random number generator 
    results.append(x)#add to the pre-allocated array

plot(results,'o')
show()
#---------------------------------------------------------------------------


#-----------------------------RNG GENERAL CODE--------------------------------
import random
from pylab import show, hist
xnsamples,xnbins = input('number of samples, number of bins')
nsamples = int(xnsamples)
nbins = int(xnbins)
x = []
for i in range(nsamples):
    x.append(random.random())

hist(x,nbins)
show()
#---------------------------------------------------------------------------


#-----------------------------RNG IN A BOX--------------------------------
from random import random#be careful with this!!
from pylab import show, plot
boxsize = 100.0
N=1000
x = []
y = []
for i in range(N):
    for j in range(N):
        x.append(random()*boxsize)#important!
        y.append(random()*boxsize)
plot(x,y)
show()
#---------------------------------------------------------------------------


#-----------------------------RNG WORKSPACE--------------------------------
from random import random
from pylab import show, plot,scatter
from math import sin, cos,pi
from numpy import array
radius = 10.0
N=1000
x = array([N,N,N],float)
y = array([N,N,N],float)
z = array([N,N,N],float)
for i in range(N):
    for j in range(N):
        r = random()*radius#placing boundaries on the random numbers
        th = random()*(2.0*pi)
        phi = (random()-0.5)*(2.0*pi)
        x[i,j,k] = r*cos(th)*cos(phi)
        y[i,j,k] = r*sin(phi)
        z[i,j,k] = r*sin(th)*cos(phi)
ax.scatter(x,y,z)
show()
#---------------------------------------------------------------------------


#--------------------------------MATRICES-----------------------------------
from numpy import mat
from numpy.linalg import solve
# We can define a matrix from a string
A=mat('1.0 0;-0.1,1')
# print one element of matrix and entire matrix
print "A[1,0] and A : ",A[1,0],"\n",A
# or we can define a matrix from an array
x1=1.
x2=5.
B=mat([[11.,x2-x1],[2.,3.]])
print "B: \n",B
print "A*B: \n",A*B
# we can make a vector (row)(from a string)
C=mat('3,2')
print "C: ",C
# or a vector (column) (from a string)
D=mat('4;5')
print "D: \n",D
# another way to extract one item from a matrix
print "first item in D: ",D.item(0)
# or a different way to make a vector column (using array)
E=mat([[6.],[7]])
print "column vector E: \n",E
# to multiply a matrix into a vector the vector must be a column
# eg. the transpose of a row
print "B transpose: \n",C.T
print "A times C transpose: \n",A*C.T
# or just a column vector
print "A times D: \n",A*D
#sometime we need the inverse of a matrix
print "A inverse: \n",A.I
# or we want to solve a matrix equation like  Ax=D
print "x=solve(A,D):  \n ",solve(A,D)
#--------------------------------------------------------------------------
