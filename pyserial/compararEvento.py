from dtaidistance import dtw
import numpy as np
import time
import os
def compararEvento(buffer2,lock,get_lastEventTime): 
    COOLDOWN = 10   
    detectando = False
    LIMIAR_RAPIDO = 60
    LIMIAR_DTW = 15
    contador = 0
    evento_cache = None
    last_mtime = 0
    while True:
        time.sleep(0.5)
        if time.time() - get_lastEventTime() < COOLDOWN:
            continue
        if not os.path.exists("evento.npy"):
            continue
        mtime = os.path.getmtime("evento.npy")
        if evento_cache is None or mtime != last_mtime:
            print("evento atualizado!")
            evento_cache = np.load("evento.npy")[:,:8]
            last_mtime = mtime
        evento = evento_cache
        with lock:
            snapshot = list(buffer2)
        if len(snapshot) < len(evento):
            continue    
        atual_np = np.array(snapshot)[:,:8]
        evento_flat = evento.flatten()
        evento_flat = (evento_flat - np.mean(evento_flat)) / (np.std(evento_flat) + 1e-9)
        melhor_dist = float('inf')
        for i in range(0, len(atual_np)-len(evento)+1,20):
            janela = atual_np[i:i+len(evento)]
            janela_flat = janela.flatten()
            janela_flat = (janela_flat - np.mean(janela_flat)) / (np.std(janela_flat) + 1e-9)




            dist_simples = np.linalg.norm(evento_flat - janela_flat)
            if dist_simples < LIMIAR_RAPIDO:
                dist = dtw.distance_fast(evento_flat, janela_flat)
                if dist < melhor_dist:
                    melhor_dist = dist
        if melhor_dist < LIMIAR_DTW:
            contador += 1
        else:
            contador = 0
        if contador >= 5:
            if not detectando:
                print("OPA! SOM SEMELHANTE!!!")
                detectando = True
        else: detectando = False