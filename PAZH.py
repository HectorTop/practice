import matplotlib.pyplot as plt
import scipy
import numpy as np
import scipy.stats
import random
from scipy.stats import gaussian_kde

data = np.array([[7.310e-05, 3.000e-06, 2.582e-01, 1.330e-02],
       [3.261e-04, 6.710e-05, 2.487e-01, 2.480e-02],
       [1.405e-04, 1.430e-05, 2.690e-01, 1.430e-02],
       [1.685e-04, 1.350e-05, 2.584e-01, 7.500e-03],
       [2.367e-04, 2.820e-05, 2.569e-01, 1.650e-02],
       [1.589e-04, 1.280e-05, 2.515e-01, 1.130e-02],
       [7.600e-05, 3.500e-06, 2.560e-01, 8.800e-03],
       [1.727e-04, 1.400e-05, 2.544e-01, 7.700e-03],
       [1.863e-04, 1.540e-05, 2.549e-01, 1.540e-02],
       [6.260e-05, 3.500e-06, 2.471e-01, 1.350e-02],
       [1.976e-04, 2.140e-05, 2.448e-01, 1.050e-02],
       [1.312e-04, 1.050e-05, 2.498e-01, 9.800e-03],
       [1.550e-04, 2.210e-05, 2.499e-01, 1.250e-02],
       [7.730e-05, 5.200e-06, 2.487e-01, 8.000e-03],
       [2.918e-04, 4.950e-05, 2.644e-01, 9.200e-03],
       [3.755e-04, 4.470e-05, 2.624e-01, 1.300e-02],
       [1.693e-04, 6.900e-06, 2.540e-01, 1.300e-02],
       [1.427e-04, 9.800e-06, 2.614e-01, 1.080e-02],
       [3.943e-04, 5.220e-05, 2.493e-01, 1.180e-02],
       [1.599e-04, 1.190e-05, 2.507e-01, 1.380e-02],
       [1.628e-04, 1.440e-05, 2.516e-01, 8.700e-03],
       [1.757e-04, 1.200e-05, 2.526e-01, 1.780e-02],
       [1.673e-04, 3.040e-05, 2.491e-01, 1.880e-02],
       [2.267e-04, 2.030e-05, 2.618e-01, 1.580e-02],
       [1.838e-04, 1.860e-05, 2.515e-01, 9.100e-03],
       [2.667e-04, 2.830e-05, 2.542e-01, 1.010e-02],
       [1.378e-04, 9.600e-06, 2.627e-01, 1.000e-02],
       [6.430e-05, 4.000e-06, 2.463e-01, 1.220e-02],
       [2.082e-04, 2.550e-05, 2.594e-01, 1.100e-02],
       [1.323e-04, 8.800e-06, 2.469e-01, 1.410e-02],
       [1.817e-04, 3.450e-05, 2.542e-01, 1.820e-02],
       [2.004e-04, 2.630e-05, 2.621e-01, 1.090e-02],
       [2.226e-04, 3.120e-05, 2.568e-01, 1.050e-02],
       [2.104e-04, 2.380e-05, 2.534e-01, 1.280e-02],
       [3.455e-04, 4.180e-05, 2.601e-01, 1.050e-02],
       [1.504e-04, 2.780e-05, 2.458e-01, 1.750e-02],
       [5.590e-05, 5.900e-06, 2.483e-01, 1.420e-02],
       [1.854e-04, 1.770e-05, 2.504e-01, 1.360e-02],
       [2.177e-04, 2.030e-05, 2.551e-01, 1.030e-02],
       [1.282e-04, 9.300e-06, 2.617e-01, 1.200e-02],
       [6.480e-05, 3.000e-06, 2.483e-01, 7.000e-03],
       [6.500e-05, 3.200e-06, 2.522e-01, 7.800e-03],
       [2.559e-04, 3.650e-05, 2.515e-01, 1.600e-02],
       [4.074e-04, 6.540e-05, 2.548e-01, 1.870e-02],
       [2.244e-04, 2.570e-05, 2.522e-01, 1.400e-02],
       [2.050e-04, 1.880e-05, 2.561e-01, 7.500e-03],
       [3.202e-04, 6.390e-05, 2.355e-01, 1.680e-02],
       [1.921e-04, 1.790e-05, 2.614e-01, 1.330e-02],
       [1.796e-04, 2.450e-05, 2.542e-01, 2.150e-02],
       [4.320e-05, 2.800e-06, 2.589e-01, 9.700e-03],
       [9.660e-05, 5.600e-06, 2.441e-01, 8.100e-03],
       [2.381e-04, 2.970e-05, 2.632e-01, 6.800e-03],
       [1.076e-04, 4.800e-06, 2.656e-01, 8.600e-03],
       [1.493e-04, 8.500e-06, 2.502e-01, 1.230e-02],
       [1.310e-05, 1.400e-06, 2.375e-01, 7.400e-03],
       [1.980e-05, 9.900e-06, 2.484e-01, 9.700e-03],
       [9.710e-05, 3.700e-06, 2.547e-01, 6.900e-03],
       [6.770e-05, 2.200e-06, 2.560e-01, 6.100e-03],
       [5.880e-05, 2.200e-06, 2.467e-01, 4.400e-03],
       [1.390e-04, 6.400e-06, 2.553e-01, 7.000e-03],
       [1.888e-04, 7.700e-06, 2.586e-01, 3.400e-03],
       [2.027e-04, 1.230e-05, 2.620e-01, 7.600e-03],
       [2.100e-05, 1.000e-06, 2.532e-01, 0.0061],
       [7.970e-05, 7.600e-06, 2.459e-01, 7.500e-03],
       [3.310e-05, 2.300e-06, 2.535e-01, 6.800e-03],
       [1.146e-04, 8.300e-06, 2.495e-01, 6.700e-03],
       [5.990e-05, 4.100e-06, 2.450e-01, 5.600e-03],
       [8.920e-05, 6.600e-06, 2.594e-01, 7.500e-03],
       [1.087e-04, 4.000e-06, 2.534e-01, 2.400e-03],
       [7.000e-05, 2.500e-06, 2.476e-01, 1.030e-02],
       [4.730e-05, 3.900e-06, 2.431e-01, 6.100e-03],
       [8.080e-05, 3.600e-06, 2.531e-01, 5.900e-03],
       [3.720e-05, 2.700e-06, 2.504e-01, 5.200e-03],
       [3.210e-05, 1.500e-06, 2.448e-01, 6.500e-03],
       [1.813e-04, 2.530e-05, 2.522e-01, 4.100e-03]])

def generator(med1,med2,disp1,disp2,min,max,A):
    x = np.random.rand() * (max - min) + min
    y = np.random.rand()
    while y>(disp1/0.4*gauss(x,med1,disp1)+disp2/0.4*gauss(x,med2,disp2)*A)/2:
        x = np.random.rand()*(max-min)+min
        y = np.random.rand()
        print(x,y, (disp1/0.4*gauss(x,med1,disp1)+disp2/0.4*gauss(x,med2,disp2)*A)/2)

    return x
def gauss(x,dot,d):
    outval = (0.4/d)*np.exp(-0.5*(((x-dot)/d)**2))
    return outval

nbins = 30
n = 75
k = 46
b = 0.245
dist = [0 for i in range(n+1)]
dkoord = [[0 for i in range(n)] for j in range(2)]
newdata = np.zeros(shape=(120,2))
datax = np.zeros(n)
datay = np.zeros(n)
usyx = np.zeros(n)
usyy = np.zeros(n)
realdisty = np.zeros(n)
realdistx = np.zeros(n)
xinit = np.zeros(2)
yinit = np.zeros(2)
xinit[0] = 0
xinit[1] = 0.0003
yinit[0] = 0.245
yinit[1] = 0.26
for i in range(n):
    datax[i] = data[i][0]
    datay[i] = data[i][2]
    usyx[i] = data[i][1]
    usyy[i] = data[i][3]
print(datax)
print(datay)
print(usyx)
print(usyy)
dkoordsecx = np.zeros(n)
dkoordsecy = np.zeros(n)
for i in range(n):
   #stretching for dx=dy
   stretchkoeff = data[i][3]/data[i][1]
   dkoordsecx[i] = ((datay[i]-b)/k - datax[i])/usyx[i]
   dkoordsecy[i] = (k*datax[i] + b - datay[i])/usyy[i]
   newdata[i][0] = data[i][0]*stretchkoeff
   newdata[i][1] = data[i][3]
   ks = k/stretchkoeff
   dist[i] = (ks*newdata[i][0] - data[i][2] + b)/(ks**2+1)**(1/2)/data[i][3]
   dkoord[0][i]=dist[i]*ks*((1/(1 + ks**2)**(1/2)))
   dkoord[1][i]=dist[i]*((1/(1 + ks**2)**(1/2)))
   realdistx[i] = dist[i] * data[i][3] * k * ((1 / (1 + k ** 2) ** (1 / 2)))/stretchkoeff
   realdisty[i] = dist[i] * data[i][3] * ((1 / (1 + k ** 2) ** (1 / 2)))
#normality test

p = scipy.stats.kstest(dist,'norm')



numsynth = 75
synthx = np.array([random.random()*0.0003 for i in range(numsynth)])
synthy = np.zeros(numsynth)
synthf = np.zeros(numsynth)
synthdy = np.zeros(numsynth)
synthdx = np.zeros(numsynth)
synthdistk = np.zeros(shape = (2,numsynth))
synthdist = 0
for i in range(numsynth):
    synthy[i] =  np.random.normal(50*synthx[i]-0.0015,0.01,1)
    synthdx[i] = generator(9.2*10**(-6),3.2*10**(-5),0.00001,0.00001,min(usyx),max(usyx),1/2.9)
    synthdy[i] = generator(0.0052,0.0116,0.003,0.003,min(usyy),max(usyy),1/2.7)
    synthdist = (50*synthx[i]-synthy[i])/np.sqrt(50**2+1)
    synthdistk[0][i] = (synthdist/np.sqrt(1+50**2))/synthdx[i]
    synthdistk[1][i] = (synthdist*(50**2)/np.sqrt(1+50**2))/synthdy[i]
    synthf[i] = 50*synthx[i]

figure,ax1 = plt.subplots(2,4)
ax1[0][0].scatter(dkoord[0][:],dkoord[1][:],s=1)
ax1[0][0].title.set_text("distance in sigma units")
ax1[0][0].set_xlim(-1,1)
ax1[0][0].set_ylim(-3,3)
ax1[0][0].set_xlabel('distance in x')
ax1[0][0].set_ylabel('distance in y')

ax1[0][2].scatter(synthdistk[0][:],synthdistk[1][:],s=1)
ax1[0][2].title.set_text("residuals by axises SYNTH")
ax1[0][2].set_xlim(-1,1)
ax1[0][2].set_ylim(-3,3)
ax1[0][2].set_xlabel('distance in x')
ax1[0][2].set_ylabel('distance in y')

ax1[0][3].errorbar(synthx,synthy,synthdy,synthdx,ls='none')
ax1[0][3].plot(synthx,synthf)
ax1[0][3].title.set_text("synth data")
ax1[0][3].set_xlim(0,0.0003)
ax1[0][3].set_ylim(-0.03,0.05)
ax1[0][3].set_xlabel('x')
ax1[0][3].set_ylabel('y')

ax1[0][1].errorbar(datax,datay,usyy,usyx,ls='none')
ax1[0][1].plot(xinit,yinit)
ax1[0][1].title.set_text("our data")
ax1[0][1].set_xlim(0,0.0003)
ax1[0][1].set_ylim(0.22,0.3)
ax1[0][1].set_xlabel('x')
ax1[0][1].set_ylabel('y')

graph = np.zeros(1000)
graphsynth = np.zeros(1000)
x = np.zeros(1000)
for i in range(1000):
    x[i] = i * 0.0002 - 0.1
    for j in range(n):
        graph[i] += gauss(x[i],dkoord[0][j],1.06*(n**(-0.2)*np.std(dkoord[0][:])))
    for j in range(75):
        graphsynth[i] += gauss(x[i],synthdistk[0][j],1.06*(100**(-0.2))*np.std(synthdistk[0][:]))

ax1[1][0].plot(x,graph)
ax1[1][0].title.set_text("distance to x in sigma units")
ax1[1][2].plot(x,graphsynth)
ax1[1][2].title.set_text("distance to x in sigma units synth")

for i in range(1000):
    graph[i] = 0
    x[i] = i * 0.004 - 2
    for j in range(n):
        graph[i] += gauss(x[i],dkoord[1][j],1.06*(n**(-0.2)*np.std(dkoord[1][:])))
    for j in range(75):
        graphsynth[i] += gauss(x[i], synthdistk[1][j], 1.06 * (100 ** (-0.2)) * np.std(synthdistk[1][:]))

ax1[1][1].plot(x,graph,x,max(graph)*np.std(dkoord[1][:])/0.4*gauss(x,0,np.std(dkoord[1][:])))
ax1[1][1].title.set_text("distance to y in sigma units")
ax1[1][3].plot(x,graphsynth)
ax1[1][3].title.set_text("distance to y in sigma units synth")

plt.show()
