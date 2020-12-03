import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation
import random

def findY(coef, t):
    return coef[0] * t ** 3 + coef[1] * t ** 2 + coef[2] * t + coef[3]


def findYt(coef, t):
    return coef[0] * t ** 2 + coef[1] * t + coef[2]


def coor(coef, t):
    Ylist = findY(coef, t)
    return Ylist


def coorDer(coef, t):
    Ylsit = findYt(coef, t)
    return Ylsit


def graph(xP, yP, t, y, coef,critY,critX):
    fig = plt.figure(figsize = (5.5,5))
    plt.grid()
    s = f'{xP[0]}, {yP[0]}'
    s2 = f'{xP[3]}, {yP[3]}'
    plt.ylabel("y")
    plt.xlabel("x")
    plt.scatter(xP[0],yP[0])
    plt.scatter(xP[3],yP[3])
    plt.text(xP[0],yP[0],s)
    plt.text(xP[3],yP[3],s2)
    xa = xP[3]
    xb = xP[0]
    ya = critY['max']
    yb = critY['min']
    plt.xlim(xb-50, xa+50)
    plt.ylim(yb-50, ya+50)
    plt.plot(t, y, color = "red")
    der = coorDer(np.polyder(coef),t)
    #plt.plot(t, der)
    plt.scatter(critX['max'], critY['max'], label = 'Máximo')
    plt.scatter(critX['min'], critY['min'], label = 'Mínimo')
    c1 = f"{round(critX['max'],2)}, {round(critY['max'],2)}"
    c2 = f"{round(critX['min'],2)}, {round(critY['min'],2)}"
    plt.text(critX['max'], critY['max'],c1)
    plt.text(critX['min'], critY['min'],c2)
    plt.text(xP[3],yP[3],s2)
    plt.legend()
    st.pyplot(plt)
    plt.close()

def graphT(xP, yP, t, y, yO,critY,critX,YcritZ,TangenCritz):
    fig = plt.figure(figsize = (5.5,5))
    plt.grid()
    s = f'{xP[0]}, {yP[0]}'
    s2 = f'{xP[3]}, {yP[3]}'
    plt.ylabel("y")
    plt.xlabel("x")
    plt.scatter(xP[0],yP[0])
    plt.scatter(xP[3],yP[3])
    plt.text(xP[0],yP[0],s)
    plt.text(xP[3],yP[3],s2)
    xa = xP[3]
    xb = xP[0]
    ya = critY['max']
    yb = critY['min']
    plt.xlim(xb-50, xa+50)
    plt.ylim(yb-50, ya+50)
    plt.plot(t, y,)
    plt.scatter(YcritZ['x'],YcritZ['y'], color = 'brown')
    plt.plot(t , yO, color = "red")
    result = isinstance(TangenCritz, int)
    if result != True:
        plt.plot(t,TangenCritz)
    plt.scatter(critX['max'], critY['max'], label = 'Máximo')
    plt.scatter(critX['min'], critY['min'], label = 'Mínimo')
    c1 = f"{round(critX['max'],2)}, {round(critY['max'],2)}"
    c2 = f"{round(critX['min'],2)}, {round(critY['min'],2)}"
    plt.text(critX['max'], critY['max'],c1)
    plt.text(critX['min'], critY['min'],c2)
    plt.legend()
    st.pyplot(plt)

def deriv(coef, t):
    curv = np.polyder(coef)
    y = coorDer(curv, t)
    return y

def secondD(curv, t):
    curvD= np.polyder(curv)
    secondY = curvD[0] * t + curvD[1]
    return secondY

def getTangen(coef, t, deltaT):
    curv = np.polyder(coef)
    secondY = secondD(curv,t) #coordenadas segunda d
    derivY = deriv(coef,t) # coordenadas primera d
    m = findYt(curv, deltaT) #valor de la pendiente
    yValue = findY(coef, deltaT) #coordenadas de la tangente
    b = yValue - m * deltaT
    index = (np.abs(derivY - m)).argmin() #encuentra el indice del valor más cercano a m
    K = np.abs(secondY[index])/(1 + derivY[index]**2)**(3/2)
    st.write(f"Función: $y = {round(m,3)}x+{round(b,3)}$")
    st.text(f"Radio de la curvatura en X: {1/K}")
    return m * t + b
     
def getCrit(coef, t, derY, y):
    critX = {'min': 0, 'max': 0}
    critY = {'min': 0, 'max': 0}
    indexM , indexMax = 0, 0
    for i in range(len(y)):
        if i != len(y) - 1:
            if derY[i-1] < 0 and derY[i+1] > 0:
                critX['min'] = t[i]
                indexM = i
            if derY[i-1] > 0 and derY[i+1] < 0:
                critX['max'] = t[i]
                indexMax = i
    critY['min'], critY['max'] = y[indexM], y[indexMax]
    return critY, critX

def K(t,coef):
    coefD = np.polyder(coef)
    secondY = secondD(coefD,t)
    derivY = deriv(coef,t)
    K = []
    for i in range(len(t)):
        K.append(np.abs(secondY[i])/(1 + derivY[i]**2)**(3/2))
    return K

def Kcheck(K,y,t):
    critZone = {'y': [], 'x': []}
    for i in range(len(K)):
        if 1/K[i] < 50:
            critZone['y'].append(y[i])
            critZone['x'].append(t[i])
    return critZone

def animDrift(tangenCritz, xP,t,y,coef,critY,derrapeY,derrapeX):
    fig, ax = plt.subplots()
    ln, = plt.plot([],[], 'ro')
    ax.plot(t,y)

    def tanPoint(x):
        curv = np.polyder(coef)
        m  = findYt(curv,derrapeX)
        b = derrapeY - m *derrapeX
        return m * x + b
    def init():
        ax.set_xlim(xP[0], xP[3])
        if tangenCritz[0] > 1700:
            ax.set_ylim(0, tangenCritz[0] + 1000)
        else:
            ax.set_ylim(tangenCritz[0] - 1000, critY['max'])
        return ln
    def update(frame):
        if frame < derrapeX:
            ln.set_data(frame, coor(coef,frame))
        else:
            ln.set_data(frame,tanPoint(frame))
    ani = FuncAnimation(fig,update, frames = t , init_func = init, blit = False)
    ani.save("drift_animation.gif", writer = "imagemagick", fps = 60)

def anim(xP,t,y,coef,critY):
    fig, ax = plt.subplots()
    ln, = plt.plot([],[], 'ro')
    ax.plot(t,y)

    def init():
        ax.set_xlim(xP[0],xP[3])
        ax.set_ylim(critY['min']-50,critY['max']+50)
        return ln

    def update(frame):
        ln.set_data(frame, coor(coef,frame))

    ani = FuncAnimation(fig,update, frames = t, init_func = init, blit = False)
    ani.save("animation.gif", writer = "imagemagick", fps = 60)

def find_nearest(array, value): #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def graphV(tangenCritz,xP,t,y,coef,critY,derrapeY,derrapeX):
    fig = plt.figure(figsize = (7,7))
    plt.grid()
    vmaxC = 114
    vmax = 320
    vx1 = [0,610,1400]
    vy1 = [vmax,vmaxC,vmax]
    cov1 = np.polyfit(vx1,vy1,2)
    vx2 = [1400,2090,2600]
    vy2 = [vmax,vmaxC,vmax]
    cov2 = np.polyfit(vx2,vy2,2)
    y1 = random.randint(-20,20) + (cov1[0]*t**2 + cov1[1] * t + cov1[2])
    y2 = random.randint(-20,20) + (cov2[0]*t**2 + cov2[1] * t + cov2[2])
    y1 = np.split(y1,2)[0]
    y2 = np.split(y2,2)[1]
    yf = np.append(y1,y2)
    plt.plot(t,yf)
    plt.ylabel("m/s")
    plt.xlabel("X")
    plt.title("Velocidad vs Desplazamiento(x)")
    st.pyplot(plt)