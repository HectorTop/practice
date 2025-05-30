import matplotlib.pyplot as plt
import scipy
import numpy as np
import scipy.stats
import random

data = list([[1.50000000e-05,   1.00000000e-06,   2.36600000e-01, 1.37000000e-02],
    [2.00000000e-05,   1.00000000e-06,   2.49700000e-01, 0.69000000e-02],
    [2.80000000e-05,   1.00000000e-06,   2.68900000e-01, 1.20000000e-02],
    [3.20000000e-05,   1.00000000e-06,  2.47100000e-01, 6.80000000e-03],
    [3.50000000e-05,   1.00000000e-06 ,  2.43100000e-01, 1.05000000e-02],
    [4.00000000e-05 ,  1.00000000e-06,   2.41500000e-01, 1.25000000e-02],
    [4.20000000e-05,   3.00000000e-06,   2.48800000e-01, 9.60000000e-03],
    [5.50000000e-05,   2.00000000e-06,   2.54000000e-01, 1.48000000e-02],
    [6.10000000e-05,   1.00000000e-06,   2.48400000e-01, 2.60000000e-03],
    [6.40000000e-05,   2.00000000e-06,   2.47000000e-01, 1.59000000e-02],
    [7.20000000e-05,   2.00000000e-06,   2.54800000e-01, 1.28000000e-02],
    [7.70000000e-05,   2.00000000e-06,   2.47300000e-01, 6.40000000e-03],
    [1.01000000e-04 ,  2.00000000e-06,   2.54900000e-01, 2.32000000e-02],
    [1.17000000e-04 ,  3.00000000e-06,   2.53500000e-01, 1.47000000e-02],
    [1.52000000e-04,   4.00000000e-06,   2.54600000e-01, 0.63000000e-02],
    [18.9e-5,   0.4e-5,   0.2618,   0.0096],
    [19.2e-5,   0.5e-5,   0.2692,   0.0158],
    [9.8e-5, 0.2e-5,  0.2462, 0.0197],
    [3.5e-5, 0.1e-5,   0.2596, 0.0105],
    [20.4e-5, 0.2e-5,  0.2464, 0.0089],
    [1.34e-04, 1.67e-05, 0.2592, 0.0175 ],
    [2.03e-04, 1.99e-05, 0.2486, 0.0123 ],
    [2.46e-04, 6.33e-05, 0.2638, 0.0123 ],
    [1.58e-04, 2.04e-05, 0.2471, 0.0109 ],
    [2.66e-04, 3.62e-05, 0.2629, 0.0087 ],
    [1.58e-04, 2.64e-05, 0.2553, 0.0080],
    [2.16e-04, 1.44e-05, 0.2583, 0.0054 ],
    [1.43e-04, 2.11e-05, 0.2702, 0.0244 ],
    [1.47e-04, 1.90e-05, 0.2462, 0.0079 ],
    [7.49e-05, 2.10e-06, 0.2426, 0.0099 ],
    [1.18e-04, 4.60e-06, 0.2726, 0.0083],
    [1.47e-04, 1.64e-05, 0.2659, 0.0105 ],
    [1.79e-04, 3.58e-05, 0.2582, 0.0134],
    [1.72e-04, 1.96e-05, 0.2541, 0.0128],
    [9.80e-05, 9.00e-06, 0.2551, 0.0153],
    [1.43e-04, 3.79e-05, 0.2737, 0.0241],
    [1.18e-04, 8.00e-06, 0.2411, 0.0124 ],
    [8.92e-05, 8.00e-06, 0.2518, 0.0069],
    [2.36e-04, 2.47e-05, 0.2751, 0.0108],
    [1.52e-04, 6.70e-06, 0.2512, 0.0085 ],
    [8.72e-05, 1.19e-05, 0.2520, 0.0072 ],
    [1.61e-04, 1.06e-05, 0.2434, 0.0089 ],
    [1.75e-04, 1.66e-05, 0.2496, 0.0090 ],
    [2.14e-04, 3.02e-05, 0.2546, 0.0097],
    [9.75e-05, 3.50e-06, 0.2470, 0.0104 ],
    [1.24e-04, 4.10e-06, 0.2707, 0.0127],
    [1.14e-04, 7.90e-06, 0.2520, 0.0075 ],
    [1.95e-04, 3.78e-05, 0.2430, 0.0051 ],
    [1.13e-04, 7.10e-06, 0.2483, 0.0082 ],
    [1.26e-04, 1.41e-05, 0.2490, 0.0156 ],
    [2.08e-04, 1.60e-05, 0.2611, 0.0155 ],
    [1.23e-04, 8.90e-06, 0.2475, 0.0092],
    [1.83e-04, 1.14e-05, 0.2686, 0.0162],
    [1.10e-04, 6.90e-06, 0.2445, 0.0074 ],
    [1.16e-04, 1.46e-05, 0.2501, 0.0059 ],
    [2.17e-04, 1.46e-05, 0.2737, 0.0059],
    [1.44e-04, 6.20e-06, 0.2429, 0.0087],
    [2.56e-04, 3.93e-05, 0.2612, 0.0137],
    [1.76e-04, 1.96e-05, 0.2391, 0.0061],
    [1.08e-04, 3.90e-06, 0.2543, 0.0091 ],
    [1.30e-04, 1.33e-05, 0.2506, 0.0215],
    [1.33e-04, 2.66e-05, 0.2435, 0.0338],
    [1.84e-04, 2.46e-05, 0.2561, 0.0148],
    [2.71e-04, 7.26e-05, 0.2531, 0.0082],
    [1.27e-04, 1.43e-05, 0.2499, 0.0196],
    [1.44e-04, 1.10e-05, 0.2688, 0.0109],
    [1.89e-04, 1.48e-05, 0.2502, 0.0135],
    [8.23e-05, 5.10e-06, 0.2589, 0.0067],
    [1.82e-04, 2.26e-05, 0.2502, 0.0088],
    [2.69e-04, 2.48e-05, 0.2496, 0.0127],
    [1.16e-04, 1.03e-05, 0.2444, 0.0095],
    [1.37e-04, 3.00e-05, 0.2671, 0.0161],
    [2.22e-04, 2.57e-05, 0.2512, 0.0047],
    [1.11e-04, 1.12e-05, 0.2499, 0.0105],
    [1.36e-04, 2.69e-05, 0.2689, 0.0082],
    [1.76e-04, 1.51e-05, 0.2428, 0.0067],
    [1.32e-04, 1.10e-05, 0.2548, 0.0121],
    [1.52e-04, 3.52e-05, 0.2440, 0.0118],
    [1.32e-04, 3.48e-05, 0.2596, 0.0280],
    [1.67e-04, 3.26e-05, 0.2628, 0.0248],
    [2.46e-04, 4.80e-05, 0.2587, 0.0037],
    [2.47e-04, 6.62e-05, 0.2480, 0.0061],
    [2.92e-04, 1.02e-04, 0.2534, 0.0055],
    [2.28e-04, 1.68e-05, 0.2679, 0.0049],
    [1.39e-04, 1.71e-05, 0.2565, 0.0174],
    [1.60e-04, 3.00e-06, 0.2575, 0.0096],
    [1.14e-04, 4.50e-06, 0.2463, 0.0133],
    [1.97e-04, 3.73e-05, 0.2579, 0.0159],
    [1.42e-04, 1.10e-05, 0.2409, 0.0068],
    [5.56e-05, 8.60e-06, 0.2377, 0.0150 ],
    [1.37e-04, 4.50e-06, 0.2504, 0.0078],
    [1.05e-04, 2.65e-05, 0.2419, 0.0172],
    [1.78e-04, 9.20e-06, 0.2626, 0.0077],
    [5.21e-05, 7.00e-06, 0.2364, 0.0156],
    [1.86e-04, 4.13e-05, 0.2682, 0.0108],
    [2.58e-04, 3.09e-05, 0.2676, 0.0089],
    [1.58e-04, 2.24e-05, 0.2682, 0.0182],
    [1.81e-04, 2.27e-05, 0.2598, 0.0079],
    [1.94e-04, 2.69e-05, 0.2603, 0.0071],
    [4.56e-05, 1.12e-05, 0.2442, 0.0173],
    [4.72e-05, 2.48e-05, 0.2548, 0.0163],
    [1.45e-04, 3.04e-05, 0.2420, 0.0165],
    [1.50e-04, 1.17e-05, 0.2474, 0.0104],
    [1.52e-04, 1.00e-05, 0.2668, 0.0167],
    [1.59e-04, 1.83e-05, 0.2515, 0.0087],
    [1.53e-04, 1.56e-05, 0.2436, 0.0066],
    [2.34e-04, 3.45e-05, 0.2510, 0.0073],
    [1.69e-04, 1.15e-05, 0.2432, 0.0044],
    [1.85e-04, 7.60e-06, 0.2562, 0.0048],
    [1.20e-04, 9.50e-06, 0.2443, 0.0073],
    [1.96e-04, 3.09e-05, 0.2494, 0.0084],
    [1.30e-04, 1.08e-05, 0.2470, 0.0085],
    [2.56e-04, 3.43e-05, 0.2490, 0.0074],
    [1.91e-04, 2.42e-05, 0.2541, 0.0080],
    [1.94e-04, 1.70e-05, 0.2618, 0.0059],
    [1.72e-04, 1.32e-05, 0.2664, 0.0079],
    [1.14e-04, 1.22e-05, 0.2465, 0.0133],
    [1.12e-04, 1.31e-05, 0.2482, 0.0158],
    [1.57e-04, 5.26e-05, 0.2572, 0.0116 ],
    [2.74e-04, 4.04e-05, 0.2633, 0.0063]])

def generator(med1,med2,disp1,disp2,min,max,A):
    x = np.random.rand() * (max - min) + min
    y = np.random.rand()
    while y>(disp1/0.4*gauss(x,med1,disp1)+disp2/0.4*gauss(x,med2,disp2)*A)/2:
        x = np.random.rand()*(max-min)+min
        y = np.random.rand()
        #print(x,y, (disp1/0.4*gauss(x,med1,disp1)+disp2/0.4*gauss(x,med2,disp2)*A)/2)
    return x

def gauss(x,dot,d):
    outval = (0.4/d)*np.exp(-0.5*(((x-dot)/d)**2))
    return outval

n = 120
k = 46
b = 0.245
dist = [0 for i in range(n+1)]
dkoord = [[0 for i in range(n)] for j in range(2)]
newdata = np.zeros(shape=(120,2))
datax = np.zeros(120)
datay = np.zeros(120)
usyx = np.zeros(120)
usyy = np.zeros(120)
xinit = np.zeros(2)
yinit = np.zeros(2)
realdistx = np.zeros(120)
realdisty = np.zeros(120)
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
    dkoordsecx[i] = ((datay[i] - b) / k - datax[i]) / usyx[i]
    dkoordsecy[i] = (k * datax[i] + b - datay[i]) / usyy[i]
    #stretching for dx=dy
    stretchkoeff = data[i][3]/data[i][1]
    newdata[i][0] = data[i][0]*stretchkoeff
    newdata[i][1] = data[i][3]
    ks = k/stretchkoeff
    dist[i] = (ks*newdata[i][0] - data[i][2] + b)/(ks**2+1)**(1/2)/data[i][3]
    dkoord[0][i]=dist[i]*ks*((1/(1 + ks**2)**(1/2)))
    dkoord[1][i]=dist[i]*((1/(1 + ks**2)**(1/2)))
    realdistx[i] = dist[i]*data[i][3]*k*((1/(1 + k**2)**(1/2)))/stretchkoeff
    realdisty[i] = dist[i] * data[i][3] * ((1 / (1 + k ** 2) ** (1 / 2)))

synthx = np.array([random.random()*0.0003 for i in range(100)])
synthy = np.zeros(100)
synthf = np.zeros(100)
synthdy = np.zeros(100)
synthdx = np.zeros(100)
synthdistk = np.zeros(shape = (2,100))
synthdist = 0
for i in range(100):
    synthy[i] = np.random.normal(50 * synthx[i] - 0.0025, 0.01, 1)
    synthdx[i] = generator(7.2 * 10 ** (-6), 2.8 * 10 ** (-5), 0.00001, 0.000008, min(usyx), max(usyx), 1 / 3.1)
    synthdy[i] = generator(0.0045, 0.0111, 0.003, 0.003, min(usyy), max(usyy), 1 / 2.6)
    synthdist = (50*synthx[i]-synthy[i])/np.sqrt(50**2+1)
    synthdistk[0][i] = (synthdist/np.sqrt(1+50**2))/synthdx[i]
    synthdistk[1][i] = (synthdist*(50**2)/np.sqrt(1+50**2))/synthdy[i]
    synthf[i] = 50*synthx[i]

figure,ax1 = plt.subplots(2,4)
ax1[0][0].scatter(dkoord[0][:],dkoord[1][:],s=1)
ax1[0][0].title.set_text("residuals by axises")
ax1[0][0].set_xlim(-1,1)
ax1[0][0].set_ylim(-3,3)
ax1[0][0].set_xlabel('distance in x')
ax1[0][0].set_ylabel('distance in y')

ax1[0][2].scatter(synthdistk[0][:],synthdistk[1][:],s=1)
ax1[0][2].title.set_text("residuals by axises")
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
    for j in range(100):
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
    for j in range(100):
        graphsynth[i] += gauss(x[i], synthdistk[1][j], 1.06 * (100 ** (-0.2)) * np.std(synthdistk[1][:]))

ax1[1][1].plot(x,graph,x,max(graph)*np.std(dkoord[1][:])/0.4*gauss(x,0,np.std(dkoord[1][:])))
ax1[1][1].title.set_text("distance to y in sigma units")
ax1[1][3].plot(x,graphsynth)
ax1[1][3].title.set_text("distance to y in sigma units synth")
figure.tight_layout()
plt.show()