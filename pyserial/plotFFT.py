import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import os
fig, ax = plt.subplots()
plt.ion()  # modo interativo
plt.show(block=False)
fft_cache = None
last_mtime = 0


img = None

while True:
    if not os.path.exists("fft.npy"):
        plt.pause(0.1)
        continue

    mtime = os.path.getmtime("fft.npy")

    if fft_cache is None or mtime != last_mtime:
        print("🔄 FFT atualizada!")

        fft_cache = np.load("fft.npy")
        print(fft_cache.shape)
        print("min", np.min(fft_cache))
        print("max", np.max(fft_cache))
        last_mtime = mtime
        print(np.std(fft_cache))
        print(np.std(np.abs(fft_cache[0])))
        fft_mag = np.abs(fft_cache)
        fft_mag = 20 * np.log10(fft_mag + 1e-9)
        fft_mag = fft_mag[:,:300]
        vmax = np.max(fft_mag)
        vmin = vmax - 50


        print("min pos log", np.min(fft_mag))
        print("max pos log", np.max(fft_mag))
        # normalização (opcional, mas melhora visual)
        #fft_mag = (fft_mag - np.mean(fft_mag)) / (np.std(fft_mag) + 1e-9)

        ax.clear()

        img = ax.imshow(
            fft_mag.T,
            aspect='auto',
            origin='lower',
            cmap='inferno',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title("FFT ao longo do tempo")
        ax.set_xlabel("Tempo (frames)")
        yticks = np.linspace(0, fft_mag.shape[1], 6)
        ylabels = [f"{int(y * 44100 / 2048)} Hz" for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
