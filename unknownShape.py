from tkinter import *

import numpy as np 
import cv2
import os
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

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert


arr2=[]
path=[]
visited=[]

# DFS function using recursion
def DFS(x, y, n, m):
    global arr2
    global visited
    global path
    if (x >= n or y >= m):

        return
    if(x < 0 or y < 0):

        return
    if(arr2[x][y]==1):

        return
    if(visited[x][y] == True):

        return
    
    visited[x][y] = True
    path.append([x,y])

# Check 8 side of each node
    DFS(x-1, y-1, n, m)
    DFS(x-1, y,  n, m)
    DFS(x-1, y+1, n, m)
    DFS(x, y-1,  n, m)
    DFS(x, y+1,  n, m)
    DFS(x+1, y-1,  n, m)
    DFS(x+1, y,  n, m)
    DFS(x+1, y+1,  n, m)


# Camera run
def cameraon():
    global mask

    cap = cv2.VideoCapture(1)
    while True:
        ret, img = cap.read()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Video feed', mask)
        k = cv2.waitKey(1)
        if k%256 == 27: #ESC Pressed
            break
         
    cap.release()
    cv2.destroyAllWindows()
thread.start_new_thread(cameraon, ()) 

# Capture the shape from whiteboard as an image
def capture():
    global mask
    global arr2
    global visited
    global path

    path=[]

    blurred = cv2.blur(mask, (3,3))
    canny = cv2.Canny(blurred, 50, 200)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = mask[y1:y2, x1:x2]
    cimg=cv2.resize(cropped,(120,120))
    image=invert(cimg)

	# perform skeletonization
    skeleton = skeletonize(image)

    #RGB to grayscale 
    skeleton=cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
    
    # grayscale to binary image
    ret,thresh_img=cv2.threshold(skeleton,127,255,cv2.THRESH_BINARY)

    # Image to array
    arr=np.asarray(thresh_img)

    #Convert (0,255) to (0,1) 
    arr2=[]
    for i in range(len(arr)):
        l=[]
        for j in range(len(arr[i])):
            if(arr[i][j]==255):
                j=0
            else: j=1
	            
            l.append(j)
	        
        arr2.append(l)
	    
    arr2=np.array(arr2)

# DFS implementation
    m=120
    n=120
    visited=[]
    for i in range(0,n):
        v=[]
        for j in range(0,m):
            v.append(0)
        visited.append(v)
    visited=np.array(visited)

    f = 0
    for i in range(n):
        for j in range(m):
            if(arr2[i][j]==0):
                DFS(i,j,n,m)
                f = 1
            if(f==1):
                break
        if(f==1):
            break

    path=np.asarray(path)
    path=path.reshape(-1)
    path=np.array(path)
    print("Coordinate=",path)


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
        if(abs(dx and dy)>2):
            bot.pen_up()
            time.sleep(1)
            bot.pen_down()
            bot.go_to_xy(y[i+1],x[i+1])
        else:
            bot.pen_down()
            bot.go_to_xy(iy,ix)

# Robot draw the path tarjectory
def write():
	global path
	bot=AutoKiddobot()
	bot.start()
	x,y=xycvrt(path)
	bot.go_to_xy(y[0],x[0])
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
ws = websocket.WebSocketApp("ws://192.168.43.15:8000",  on_message = on_message,  on_error = on_error, on_close = on_close)
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

    # con3=u"এটা কি লিখেছি"
    
    con4= u"লিখ"


    
    if v==con1:
        time.sleep(1)
        data={'type':'tts', 'data': u' পারিই  '}
        data = json.dumps(data)
        ws.send(data)
        
    elif v==con2:
        time.sleep(1)
        data={'type':'tts', 'data': u' তুমি যা লিখবে তাই লিখতে পারবো'}
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
	button2=Button(root,text='Capture',command=capture).place(x=100,y=50)
	button3=Button(root,text='Interact',command=interact).place(x=100,y=110)
 
	root.mainloop()

