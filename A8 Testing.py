from numpy import mat,arange,matrix,linspace
from numpy.linalg import solve
import matplotlib.pyplot as plt
from pylab import plot,xlabel,ylabel,show,figure,legend

#INPUTS FOR THE PROGRAM 
#----------------------------------------------------------------------
#FIRST TWO LENSES
pos_12 = 3.
#focal lengths
f_lens1 = 5.
f_lens2 = -20.
#aperture radius (size)
a_rad12 = 1.#size of the lens, rays outside will pass right by
pos_3 = 12.

#THIRD LENS
f_lens3 = 1.
a_rad3 = 0.5

x_max = 15.
#----------------------------------------------------------------------

#PRE-ALLOCATING SPACE - MAKING VECTORS
#----------------------------------------------------------------------
#initial angle from the x-axis
alpha = [-0.05, 0, 0.05, 0.1]
y_initial = arange(-1,1.2,0.2)

#putting the focal lengths and positions into arrays to be called later
focal = [(f_lens1 + f_lens2),f_lens3]#thin lenses in contact - (f1+f2)
d = [pos_12, (pos_3 - pos_12), (x_max - pos_3)]
#----------------------------------------------------------------------

#SETTING UP THE PHYSICS
#----------------------------------------------------------------------
#Translation Vectors
trans0 = mat([[1.,d[0]],[0.,1.]])
trans1 = mat([[1.,d[1]],[0.,1.]])
trans2 = mat([[1.,d[2]],[0.,1.]])
T = [trans0, trans1, trans2]

#Action of the thin lens
lens12 = mat([[1.,0.],[-1/focal[0],1]])
lens3 = mat([[1.,0.],[-1/focal[1],1]])
L = [lens12,lens3]
#----------------------------------------------------------------------

#MAIN LOOP
#----------------------------------------------------------------------
x_points = [0,pos_12,pos_3,x_max]
for a in alpha:
    for y in y_initial:
        initial_ray = mat([[y],[a]]) #initial vector for the ray (y initial)
        ray_afterL1 = T[1]*L[0]*T[0]*initial_ray #vector of the ray after crossing through the lens - enters L2
        ray_afterL2 = T[2]*L[1]*ray_afterL1 #vector after L2
    
        initial_points = initial_ray.item(0) #taking the y value of the initial ray
        y_at_L1 = a*d[0] + y #vector for the ray entering Lens 1 
        y_at_L2 = ray_afterL1.item(0) #takes the y value from the vector
        y_at_end = ray_afterL2.item(0)

        y_points = [initial_points,y_at_L1, y_at_L2, y_at_end]#puts the y values through the system into one array to be plotted
        #if/else statements to determine whether or not the ray passed through the lens using the aperture radius
        ##color changes for each different alpha
        if a==-0.05:
            if y_at_L1 > a_rad12 or y_at_L1 < -a_rad12 or y_at_L2 > a_rad3 or y_at_L2 < -a_rad3:
                plt.plot(x_points,y_points,'r--',label = 'alpha = -0.05')
            else:
                plt.plot(x_points,y_points,'r',label = 'alpha = -0.05')
        elif a==0:
            if y_at_L1 > a_rad12 or y_at_L1 < -a_rad12 or y_at_L2 > a_rad3 or y_at_L2 < -a_rad3:
                plt.plot(x_points,y_points,'b--',label = 'alpha = 0')
            else:
                plt.plot(x_points,y_points,'b',label = 'alpha = 0')
        elif a==0.05:
            if y_at_L1 > a_rad12 or y_at_L1 < -a_rad12 or y_at_L2 > a_rad3 or y_at_L2 < -a_rad3:
                plt.plot(x_points,y_points,'c--',label = 'alpha = 0.05')
            else:
                plt.plot(x_points,y_points,'c',label = 'alpha = 0.05')
        elif a==0.1:
            if y_at_L1 > a_rad12 or y_at_L1 < -a_rad12 or y_at_L2 > a_rad3 or y_at_L2 < -a_rad3:
                plt.plot(x_points,y_points,'m--',label = 'alpha = 0.1')
            else:
                plt.plot(x_points,y_points,'m',label = 'alpha = 0.1')
#-----------------------------------------------------------------------
#PLOTTING
fig = figure(1)            
plt.vlines(pos_12,-1,1,linestyle = '-',linewidth = 2)
plt.vlines(pos_3,-0.5,0.5, linestyle = '-', linewidth = 2)
plt.hlines(0,0,15)
plt.xlabel('Position - x')
plt.ylabel('Height - y')
fig.suptitle('Phys 236 - A8 - Lens Optics')
show()




