# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:06:03 2018

@author: PARS PARDAZ CO.1
"""
#----------- Multithreading --------------
import threading
import socket
from scipy.io import wavfile
import binascii
import sounddevice as sd

def tcp_server():
    frames=[]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 5005))
    s.listen(1)
    print("TCP Server is Listening . . .")
    conn, addr = s.accept()
    print ('TCP Server: Connection address:', addr)
    while 1:
        data = conn.recv(80)
        if not data: break
        #print ("TCP Server: received data:", data)
        #print (binascii.hexlify(data))
        frames.append(data)
        print("\nlen=", len(frames))
        if(len(frames)==460): break
#        if(len(frames)>10000):
#            sd.play(frames, 8000)
#            frames=[]
        
        #conn.send(data)  # echo
    conn.close()
    print("\nServer Finished!")
    sd.play(frames, 8000)
    
def tcp_client():
    fs, data = wavfile.read('d:\waves\noah.wav')
    TCP_IP = '127.0.0.1'
    TCP_PORT = 5005
    BUFFER_SIZE = 1024
    MESSAGE = "Hello, World!"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    
    
    s.send(bytes(MESSAGE, "utf-8"))
    data = s.recv(BUFFER_SIZE)
    s.close()
    print ("TCP Client: received data:", data)
    print("\nClient Finished")

    
t = threading.Thread(target=tcp_server)
t.daemon = True
t.start()

#client_t = threading.Thread(target=tcp_client)
#client_t.daemon=True
#client_t.start()

# ---------- TCP Client ---------------
import socket
import numpy as np
from scipy.io import wavfile
import threading
import sounddevice as sd
from time import sleep

fs, data = wavfile.read('d:\\waves\\Noah_Denoised.wav')
#sd.play(data, 8000)

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE = "Bye, World!"
frame_len=80

frame_numbers=int(len(data)/frame_len)

i=0
l_index=i*frame_len
h_index=(i+1)*frame_len
frame=data[l_index:h_index]


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
for i in range(frame_numbers):
    l_index=i*frame_len
    h_index=(i+1)*frame_len
    frame=data[l_index:h_index]
    array2send=np.int16(frame)
    s.send(array2send)
    sleep(0.02)
#s.send(bytes(MESSAGE, "utf-8"))

#data = s.recv(BUFFER_SIZE)
#s.close()

#print ("TCP Client: received data:", data)

# ---------- TCP Server ---------------
    '''
import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 20  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
#s.close()

conn, addr = s.accept()
print ('Connection address:', addr)
while 1:
    data = conn.recv(BUFFER_SIZE)
    if not data: break
    print ("received data:", data)
    conn.send(data)  # echo
conn.close()

#================================================================
#------------------------ UDP Client ----------------------------
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = "Hello, World!"

print ("UDP target IP:", UDP_IP)
print ("UDP target port:", UDP_PORT)
print ("message:", MESSAGE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))
#------------------ UDP Server ----------------------------
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
     data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
     print ("received message:", data)
'''     