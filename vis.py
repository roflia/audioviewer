import sys
import time
import atexit
import warnings

import threading
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio

DEFAULT_DEVICE = 'Stereo Mix (Realtek'
DEFAULT_FPS = 30
DEFAULT_WINDOWSIZE = 5
DRAWRATE = DEFAULT_WINDOWSIZE*1

class Listener:
    def __init__(self, device):
        self.DEVICE = device
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100 
        self.FPS = DEFAULT_FPS
        self.CHANNELS = 2 
        self.INTERVAL = int(1000/self.FPS)
        self.CHUNK = int(self.RATE/self.FPS)
        self.WINDOWSIZE = DEFAULT_WINDOWSIZE
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                              input=True,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              frames_per_buffer=self.CHUNK,
                              stream_callback=self.new_window)
        self.lock = threading.Lock()
        self.stop = False
        self.windows = [np.zeros(self.CHUNK*self.WINDOWSIZE) for i in range(self.CHANNELS)]
        atexit.register(self.close)

    def new_window(self, data, *args, **kwargs):
        data = np.fromstring(data, 'int16')
        with self.lock:
            for i in range(self.CHANNELS):
                self.windows[i] = np.concatenate((self.windows[i],data[i::self.CHANNELS]), axis=None)
                self.windows[i] = self.windows[i][self.CHUNK:]
                if self.stop: return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_windows(self):
        with self.lock:
            return self.windows

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()



class Widget:
    def __init__(self, args):
        self.init_pyaudio(args)
        self.init_fig(args)

    def init_pyaudio(self, args):
        if not args: DEVICE = DEFAULT_DEVICE
        else: DEVICE = args[0]
        self.l = Listener(DEVICE)
        self.l.start()
        
    def init_fig(self, args):
        Lcolor = 'red'; Rcolor = 'blue';
        n = self.l.CHUNK*self.l.WINDOWSIZE
        self.freq_vect = np.fft.rfftfreq(n, 1./self.l.RATE)
        self.time_vect = np.arange(n, dtype=np.float32) / self.l.RATE * 1000

        self.fig = plt.figure()
        # plot settings
        self.axRaw = plt.subplot2grid((5,1),(0,0), rowspan=2)
        self.axFrq = plt.subplot2grid((5,1),(2,0), rowspan=3) 
        self.axRaw.set_ylim(-32768, 32768)
        self.axRaw.set_xlim(0, self.time_vect.max())
        self.axFrq.set_ylim(0, 1)
        self.axFrq.set_xlim(0, self.freq_vect.max())

        # line objects        
        self.lineRawL, = self.axRaw.plot(self.time_vect[::DRAWRATE], np.zeros(len(self.time_vect))[::DRAWRATE], Lcolor)
        self.lineRawR, = self.axRaw.plot(self.time_vect[::DRAWRATE], np.zeros(len(self.time_vect))[::DRAWRATE], Rcolor)
        self.lineFrqL, = self.axFrq.plot(self.freq_vect[::DRAWRATE], np.zeros(len(self.freq_vect))[::DRAWRATE], Lcolor)
        self.lineFrqR, = self.axFrq.plot(self.freq_vect[::DRAWRATE], np.zeros(len(self.freq_vect))[::DRAWRATE], Rcolor)
    
        plt.tight_layout()
        anim = FuncAnimation(self.fig, self.start_listening, interval=int(1000/DEFAULT_FPS)) 
        self.fig.show()

    def start_listening(self, *args, **kwargs):
        windows = self.l.get_windows()
        self.lineRawL.set_ydata(windows[0][::DRAWRATE]) 
        self.lineRawR.set_ydata(windows[1][::DRAWRATE]) 
        
        fftL = np.fft.rfft(windows[0], self.l.CHUNK*self.l.WINDOWSIZE, axis=0)
        fftR = np.fft.rfft(windows[1], self.l.CHUNK*self.l.WINDOWSIZE, axis=0)         
        self.lineFrqL.set_ydata(fftL[::DRAWRATE])
        self.lineFrqR.set_ydata(fftR[::DRAWRATE])
        
        

if __name__ == '__main__':
    app = Widget(sys.argv[1:])
