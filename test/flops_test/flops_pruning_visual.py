import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_cube():  
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)
    x = np.cos(Phi)*np.sin(Theta)+0.5
    y = np.sin(Phi)*np.sin(Theta)+0.5
    z = np.cos(Theta)/np.sqrt(2)+0.5
    return x,y,z


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(np.abs(x), np.abs(y), np.abs(z),cmap=plt.cm.coolwarm,alpha=0.2)

colors = ['Blues', 'BuGn', 'BuPu',  
 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',  
 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',  
 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']

a, b, c = get_cube()
len_list = []
len_num = 10
for i in range(len_num):
    o, p, q = 10, 10, 10
    while(o**2+p**2>100) or (o**2+p**2<4):
        o = random.uniform(2,8)
        p = random.uniform(2,8)
    len_list.append([o, p, np.sqrt(100-o**2-p**2)])
    
for i in range(len_num):
    ax.plot_surface(a*len_list[i][0], b*len_list[i][1], c*len_list[i][2], cmap=colors[i])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(0,10)
ax.set_ylim(0,13)
ax.set_zlim(0,9.8)
plt.title('Visualization of pruning results on 3D data',fontsize='xx-large')
plt.show()