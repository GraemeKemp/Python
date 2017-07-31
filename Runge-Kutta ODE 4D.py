from numpy import arange,array
from pylab import plot,xlabel,ylabel,show,figure,legend

sigma = 10.0
r = 28.0
b = 2.667
interval = 0.01

t0 = 0.0
t_max = 50.0
h  = interval

def f(q,t):
    x = q[0]
    y = q[1]
    z = q[2]
    dx = sigma*(y - z)
    dy = x*(r-z) - y 
    dz = x*y - b*z
    return array([dx,dy,dz],float)
    
    
#pre-allocating space
tpoints = arange(t0,t_max,1)
xpoints = []
ypoints = []
zpoints = []

q = array([0.0,1.0,0.0],float)#initial conditions

#Runge-Kutta
for t in tpoints:
    xpoints.append(q[0])
    ypoints.append(q[1])
    zpoints.append(q[2])
    k1 = h*f(q, t)
    k2 = h*f(q + 0.5*k1,t+0.5*h)
    k3 = h*f(q + 0.5*k2,t+0.5*h)
    k4 = h*f(q + k3,t+h)
    q  += (k1+2*k2+2*k3+k4)/6.0


#PLOTTING
fig = figure(1)
line1,=plot(tpoints,ypoints,label = 'y vs t')
line2,=plot(xpoints,zpoints,label = 'z vs x')
fig.suptitle('Phys 236 - A6')
legend()
fig.savefig('figure1.png')

