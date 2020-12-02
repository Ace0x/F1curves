import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import Curves
import base64

def main():
    xP, yP, t, y, coef = Points() 
    critY, critX = Curves.getCrit(coef,t,Curves.deriv(coef,t),y)
    st.write(f"Función: ${round(coef[0],10)}x^3{round(coef[1],10)}x^2+{round(coef[2],10)}x+{round(coef[3],10)}$")
    Curves.graph(xP, yP, t, y, coef,critY,critX)
    tangent(xP, yP, t, coef, y,critY,critX)
    animate(xP,yP,t,y,coef,critY)

def Points():
    x1 = st.number_input("x1", value=100)
    y1 = st.number_input("y1", value=1700)
    x2 = st.number_input("x2", value=499)
    y2 = st.number_input("y2", value=1850)
    x3 = st.number_input("x3", value=1500)
    y3 = st.number_input("y3", value=1710)
    x4 = st.number_input("x4", value=2600)
    y4 = st.number_input("y4", value=1800)
    xP = [x1, x2, x3, x4]
    yP = [y1, y2, y3, y4]
    t = np.linspace(xP[0], xP[3])
    coef = np.polyfit(xP, yP, 3)
    y = Curves.coor(coef, t)
    return xP, yP, t, y, coef

def tangent(xP, yP, t, coef, yO,critY,critX):
    deltaT = st.slider("X", float(xP[0]), float(xP[3]), step = 0.001)
    y = Curves.getTangen(coef, t, deltaT)
    YcritZ = critZone(t,yO,coef)
    tangenCritz = 0
    if len(YcritZ['x']) != 0:
        amp = st.slider("Punto de derrape:", min_value = 1, max_value=len(YcritZ['x']))
        tangenCritz = Curves.getTangen(coef,t,YcritZ['x'][amp-1])
    Curves.graphT(xP, yP, t, y, yO,critY,critX,YcritZ,tangenCritz)

def critZone(t,y,coef):
    K = Curves.K(t,coef)
    YcritZ = Curves.Kcheck(K,y,t)
    return YcritZ

def animate(xP,yP,t,y,coef,critY):
    Curves.anim(xP,yP,t,y,coef,critY)
    file = open("animation.gif", "rb")
    cont = file.read()
    data_url = base64.b64encode(cont).decode("utf-8")
    file.close()
    st.markdown( f'<img src="data:image/gif;base64,{data_url}" alt="animation gif">',
    unsafe_allow_html=True,)

main()
