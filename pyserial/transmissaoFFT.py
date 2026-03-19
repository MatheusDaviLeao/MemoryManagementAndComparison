
import serial
import threading
from collections import deque
import time
import librosa
import os
import numpy as np
from byteToSamples import byteToSamples 
from calculaFFT import calculaFFT
from calculaMFCC import calculaMFCC
from compararEvento import compararEvento
record_start = None

#porta
PORT = '/tmp/ttyS'
BAUDRATE = 1000000

#caracteristicas da transmissão
SAMPLE_RATE = 48000
OUTPUT_FILE = "evento.npy"
FFT_SIZE = 2048
CHUNK = 2048
lock = threading.Lock()
lastEventTime = 0


#buffer 5 segundos (evento)
BUFFER_SECONDS = 5
BUFFER_SIZE = 110
buffer = deque(maxlen=BUFFER_SIZE)

#buffer 15 segundos (comparação)
BUFFER2_SECONDS = 15
BUFFER2_SIZE = 330
buffer2 = deque(maxlen=BUFFER2_SIZE)

#buffer 5 segundos (FFT)
BUFFER3_SECONDS = 5
BUFFER3_SIZE = 110
buffer3 = deque(maxlen=BUFFER3_SIZE)

recording = False
ser = serial.Serial(PORT,BAUDRATE,timeout=1)
mel_filter = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=FFT_SIZE,
    n_mels=32
)
print("ENTER → gravar")
print("Ctrl+C → sair")
def toggle_recording():
    global recording, record_start, future_buffer, lastEventTime, future_buffer2

    while True:
        input()

        if not recording:
            print("🔴 Gravando (incluindo últimos 5 s)...")

            # salva o buffer antes de começar
            #codigo salva buffer
            future_buffer = []
            future_buffer2 = []
            recording = True
            record_start = time.time()

threading.Thread(target=toggle_recording, daemon=True).start()
threading.Thread(target=compararEvento,args=(buffer2,lock,lambda:lastEventTime),daemon=True).start()
try:
    while True:
        data = ser.read(CHUNK)
        if not data:
            continue
        #modificar data posteriormente no restante do codigo
        if len(data) < FFT_SIZE:
            continue
        mfcc = calculaMFCC(data,mel_filter)
        mfcc = mfcc[:8]
        with lock:
            buffer.append(mfcc)
            buffer2.append(mfcc)
            buffer3.append(data)
            if recording:
                future_buffer.append(mfcc)
                future_buffer2.append(data)
                if time.time - record_start >= 5:
                    evento = list(buffer) + future_buffer
                    evento = np.array(evento)
                    fft = list(buffer3) + future_buffer2
                    fft = np.array(fft)
                    np.save("evento_tmp.npy", evento)
                    os.replace("evento_tmp.npy", "evento.npy")
                    np.save("fft_tmp.npy", fft)
                    os.replace("fft_tmp.npy", "fft.npy")
                    lastEventTime = time.time()
                    recording = False
                    print("evento de 10s salvo")
except(KeyboardInterrupt):
    print("\nEncerrando")

ser.close()

print("Arquivo salvo: ", OUTPUT_FILE)