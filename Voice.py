import websocket
try:
    import thread
except ImportError:
    import _thread as thread
import time
import json
from tkinter import *
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

# def connect():
# 	websocket.enableTrace(True)
# 	ws = websocket.WebSocketApp("ws://192.168.43.23:8000",  on_message = on_message,  on_error = on_error, on_close = on_close)
# 	ws.on_open = on_open

websocket.enableTrace(True)
ws = websocket.WebSocketApp("ws://192.168.43.23:8000",  on_message = on_message,  on_error = on_error, on_close = on_close)
ws.on_open = on_open

thread.start_new_thread(run, ())





def tell():
	data={'type':'tts', 'data': u' পারিই  '}
	data = json.dumps(data)
	ws.send(data)
	print("Sent=",data)

def speak():
	
	datain={'type':'asr'}
	datain = json.dumps(datain)
	ws.send(datain)
	print("Sent=",datain)

def interact():
	global command
	# for i in range(0,10):
	# 	print("Hello")
	# 	time.sleep(1)
	command=None
	speak()
	while command  is None:
		print("Waiting for data")
		time.sleep(.5)
	print("Data found=",command)

	y=json.loads(command)

	print("voice=",y["data"])


if __name__ == '__main__':
	# print("Connecting")

	# connect()
	# time.sleep(20)
	# thread.start_new_thread(run, ())
	# print("Speaking")
	# speak()
	# time.sleep(0.5)
	# tell()

	root=Tk()
	button1=Button(root,text='Speak',command=speak).place(x=10,y=110)
	button2=Button(root,text='Tell',command=tell).place(x=100,y=110)
	button3=Button(root,text='interact',command=interact).place(x=50,y=50)

	root.mainloop()

# ws.close()
