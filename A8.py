from numpy import mat,arange
from numpy.linalg import solve

#INPUTS FOR THE PROGRAM - keep the program on GNH, below are for testing
#----------------------------------------------------------------------
#FIRST TWO LENSES
pos_12 = 3.
#focal lengths
f_lens1 = 5.
f_lens2 = -20.
#aperture radius (size)
a_rad12 = 1.
pos_3 = 12.

#THIRD LENS
f_lens3 = 1.
a_rad3 = 0.5

x_max = 15.#plots start at x=0
#----------------------------------------------------------------------

#PRE-ALLOCATING SPACE - MAKING VECTORS
#----------------------------------------------------------------------
#these make my intial vector for y
alpha = [-0.05, 0, 0.05, 0.1] #alpha is angle wrt to x-axis
y_i = arange(-1.,1.2,0.2)#initial y positions

#inputs for the action and translation vectors
focal = [f_lens1,f_lens2,f_lens3]
d = [pos_12, (pos_3 - pos_12), (x_max - pos_3)]#careful with this
#----------------------------------------------------------------------

#ACTUAL CODE TO SOLVE THE SYSTEM
#----------------------------------------------------------------------
#action of the thin lens, i
L = mat([[1.,0.],[-1/focal[i],1]])

#translation
T = mat([[1.,d[i]],[0.,1.]])

#rays
y_init_vector = mat([[alpha[0]],[y_i[0]]])
y_final = T[1]*L*T[0]*y_init_vector
#----------------------------------------------------------------------

print 'y_final', y_final



