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
import numpy as np
import pyaudio

def calc_int16(low_byte, high_byte):
    if high_byte<127:
        return high_byte*256+low_byte
    else:
        low_2s = 256 - low_byte
        high_2s = 255 - high_byte
        if low_2s==256:
            low_2s = 0
            high_2s = high_2s + 1
        return -1*(high_2s*256+low_2s)
    
def tcp_server():
    player = pyaudio.PyAudio()
    # Open Output Stream (basen on PyAudio tutorial)
    stream = player.open(format = 8,
        channels =1,
        rate = 8000,
        output = True)
    frames=[]
    findex=0
    xin =np.int16( [0] * 50000)
    temp_buffer = np.int16([0]*40)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 5005))
    s.listen(1)
    print("TCP Server is Listening . . .")
    conn, addr = s.accept()
    print ('TCP Server: Connection address:', addr)
    while 1:
        data = conn.recv(80)
        
        #if not data: break
        #print ("TCP Server: received data:", data)
        #print (binascii.hexlify(data))
        #frames.append(data)
        
        stream.write (data)
        
#        for index in range(40):
#            low_byte = data[2*index]
#            high_byte= data[2*index+1]        
#            xin[findex*40+index]=calc_int16(low_byte, high_byte)
#            temp_buffer[index] = xin[findex*40+index]
            
            #print("[", findex*40+index, "]: ", temp_buffer[index] )
            
        #print("findex=", findex)
        #sd.play(temp_buffer, 8000)
        #stream.write(temp_buffer)
        #if(len(frames)==460): break
#        if(len(frames)>10000):
#            sd.play(frames, 8000)
#            frames=[]
#        if(findex==1998):      break
        findex = findex+1
        
        #conn.send(data)  # echo
    conn.close()
    print("\nServer Finished!")
    #print("[0]: ",frames[0])
    #print (binascii.hexlify(frames[0]))
#    print("f=", findex*80)
#    print("Xin= ", xin[0], xin[1], xin[2], xin[3])
    
    #arr=np.int16(frames)
    #sd.play(xin, 8000)
    
t = threading.Thread(target=tcp_server)
t.daemon = True
t.start()
    
'''    
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
'''
    


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
import pyaudio


fs, data = wavfile.read('d:\\waves\\Noah_Denoised.wav')
#sd.play(data, 8000)

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE = "Bye, World!"
frame_len=40
frame_numbers=int(len(data)/frame_len)

p = pyaudio.PyAudio()
stream = p.open(format=8,
                channels=1,
                rate=8000,
                input=True,
                frames_per_buffer=40)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

#i=0
#l_index=i*frame_len
#h_index=(i+1)*frame_len
#frame=data[l_index:h_index]
#array2send=np.int16(frame)
#s.send(array2send)
    
#for i in range(frame_numbers):
#    l_index=i*frame_len
#    h_index=(i+1)*frame_len
#    frame=data[l_index:h_index]
#    array2send=np.int16(frame)
#    s.send(array2send)
    #sleep(0.05)

#for i in range(1000):
while 1:
    data = stream.read(40)
#    array2send=np.int16(frame)
    s.send(data)




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