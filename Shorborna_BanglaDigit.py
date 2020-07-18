from keras.models import load_model
from tkinter import *
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import websocket
try:
    import thread
except ImportError:
    import _thread as thread
import time
import json
import cv2
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
from autokiddobot import AutoKiddobot



model=load_model('SaveModelData\\vowel_bangDigit.h5')
model.summary()
y_pred='noname'
mask=0

def cameraon():
    global mask
    cap = cv2.VideoCapture(1)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Video feed', mask)
        k = cv2.waitKey(1)
        if k%256 == 27: #ESC Pressed
            break
         
    cap.release()
    cv2.destroyAllWindows()
thread.start_new_thread(cameraon, ())     
  

# Capture an image from whiteboard and predict
def predict():
    global mask
    global y_pred
    blurred = cv2.blur(mask, (3,3))
    canny = cv2.Canny(blurred, 50, 200)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = mask[y1:y2, x1:x2]
    cimg=cv2.resize(cropped,(28,28))
    ret,thresh_img=cv2.threshold(cimg,127,255,cv2.THRESH_BINARY)
    img1=np.array(thresh_img)
    t=np.array([img1])
    arr4=t.reshape(1,28,28,1)
    pred= model.predict_classes(arr4)
    print(pred)

    if(pred==[0]):
        y_pred='অ'
    elif(pred==[1]):
        y_pred='আ'
    elif(pred==[2]):
        y_pred='ই'
    elif(pred==[3]):
        y_pred='ঈ'
            
    elif(pred==[4]):
        y_pred='উ'
    elif(pred==[5]):
        y_pred='ঊ'
    elif(pred==[6]):
        y_pred='ঋ'
    elif(pred==[7]):
        y_pred='এ '
            
    elif(pred==[8]):
        y_pred='ঐ'
    elif(pred==[9]):
        y_pred='ও'
            
    elif(pred==[10]):
        y_pred='ঔ'
    elif(pred==[11]):
        y_pred='শূন্য'
    elif(pred==[12]):
        y_pred='এক'
    elif(pred==[13]):
        y_pred='দুই'
            
    elif(pred==[14]):
        y_pred='তিন'
    elif(pred==[15]):
        y_pred='চার'
    elif(pred==[16]):
        y_pred='পাঁচ'
    elif(pred==[17]):
        y_pred='ছয়'

    elif(pred==[18]):
        y_pred='সাত'
    elif(pred==[19]):
        y_pred='আট'
    elif(pred==[20]):
        y_pred='নয়'
    print("Prediction ==",y_pred)

# Convert string array to numpy array
def cvrtarr(line):
    arr=np.array(line)
    n=len(arr)
    for i in range(n):
        p1=arr[i].strip().split(',')
        pts=[]
        for item in p1:
            item=float(item)
            pts.append(item)
        pts=np.array(pts)
    pts=np.array(pts)
    return pts

# Seperate x coordinate and y coordinate
def xycvrt(data):
    x=[]
    y=[]
    for i in range(len(data)):
        if(i%2)==0:
            x.append(data[i])
        elif(1%2)!=0:
            y.append(data[i])

    x=np.array(x)
    y=np.array(y)
    return x,y

# Robot moving and lifting condition
def move(x,y):
    bot=AutoKiddobot()
    bot.start()
    for i, (ix, iy) in enumerate(zip(x,y)):
        l=len(x)
        if(i+1==l):
            break
        dx=x[i]-x[i+1]
        dy=y[i]-y[i+1]
        if(abs(dx and dy)>8):
            bot.pen_up()
            time.sleep(1)
            bot.pen_down()
            bot.go_to_xy(x[i+1],y[i+1])
        else:
            bot.pen_down()
            bot.go_to_xy(ix,iy)

# Read predicted shape from text file and draw by the robot       
def write():
    global y_pred
    bot=AutoKiddobot()
    bot.start()
    fn='Coordinate\\'+y_pred+'.txt'
    with open (fn) as file:
        line=file.readlines()
    data=cvrtarr(line)
    x,y=xycvrt(data)
    bot.go_to_xy(x[0],y[0])
    time.sleep(1)
    move(x,y)       
    bot.pen_up()
    bot.go_to_xy(0,0)
    time.sleep(2)
    bot.close()


# Speach Recognition
ws=None
command=None
def on_message(ws, message):
    global command
    print("Recived=",message)
    print("Datatype=",type(message))
    if 'datas' in message:
        command=message

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    print('Connected')

def run(*args):
        ws.run_forever()


websocket.enableTrace(True)
ws = websocket.WebSocketApp("ws://192.168.43.208:8000",  on_message = on_message,  on_error = on_error, on_close = on_close)
ws.on_open = on_open

thread.start_new_thread(run, ())

# Voice Input
def speak():

    datain={'type':'asr'}
    datain = json.dumps(datain)
    ws.send(datain)
    print("Sent=",datain)

# Conversation Structure     
def interact():
    global y_pred
    global command
    command=None
    speak()
    while command  is None:
        time.sleep(.5)
    print("Data found=",command)

    y=json.loads(command)
    
    v=y["data"] 
    print("voice=",y["data"])
    
    con1= u"লিখতে পারো"
    
    con2= u"কি লিখতে পারো"

    con3=u"এটা কি লিখেছি"
    
    con4= u"লিখ"


    
    if v==con1:
        time.sleep(1)
        data={'type':'tts', 'data': u' পারিই  '}
        data = json.dumps(data)
        ws.send(data)
        
    elif v==con2:
        time.sleep(1)
        data={'type':'tts', 'data': u' স্বরবর্ণ ও বাংলা সংখ্যা লিখতে পারি '}
        data = json.dumps(data)
        ws.send(data)

    elif v==con3:
        time.sleep(1)
        data={'type':'tts', 'data': y_pred}
        data = json.dumps(data)
        ws.send(data)
        
    elif v==con4:
        time.sleep(1)
        data={'type':'tts', 'data': u'আচ্ছা '}
        data = json.dumps(data)
        ws.send(data)

        time.sleep(1)
        write()

        
        
    else:
        time.sleep(1)
        data={'type':'tts', 'data': ' Sorry  '}
        data = json.dumps(data)
        ws.send(data)




if __name__ == '__main__':
	root=Tk()
	button1=Button(root,text='Capture',command=predict).place(x=15,y=50)
	button2=Button(root,text='Interaction',command=interact).place(x=100,y=50)
 
	root.mainloop()

