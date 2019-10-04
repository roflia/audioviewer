import sys
import time
import atexit
import warnings

import threading
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
import pyaudio

warnings.filterwarnings('ignore')
DEFAULT_DEVICE = 'Stereo Mix (Realtek'
DEFAULT_FPS = 30
DEFAULT_WINDOWSIZE = 5
RESOLUTION = 512

FCOLOR = 'black'
BCOLOR = 'white'
mpl.rcParams['text.color'] = BCOLOR
mpl.rcParams['figure.facecolor'] = FCOLOR
mpl.rcParams['figure.edgecolor'] = BCOLOR
mpl.rcParams['legend.facecolor'] = FCOLOR
mpl.rcParams['legend.edgecolor'] = BCOLOR
mpl.rcParams['axes.facecolor'] = FCOLOR
mpl.rcParams['axes.edgecolor'] = BCOLOR
mpl.rcParams['axes.labelcolor'] = BCOLOR
mpl.rcParams['ytick.color'] = BCOLOR
mpl.rcParams['xtick.color'] = BCOLOR
mpl.rcParams['axes.spines.bottom'] = False 
mpl.rcParams['axes.spines.right'] = False 
mpl.rcParams['axes.spines.left'] = False 
mpl.rcParams['lines.linewidth'] = 1 
mpl.rcParams['toolbar'] = 'None'

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
        self.DR = n//RESOLUTION
        THRESH = 1/self.l.RATE
        self.freq_vect = np.fft.rfftfreq(n, 1./self.l.RATE)
        self.time_vect = np.arange(n, dtype=np.float32) / self.l.RATE * 1000
        self.fftMaxes = [THRESH,THRESH]        

        # plot objects        
        self.fig = plt.figure()
        self.axRaw = plt.subplot2grid((5,1),(0,0), rowspan=2)
        self.axFrq = plt.subplot2grid((5,1),(2,0), rowspan=3) 
        self.lineRawL, = self.axRaw.plot(self.time_vect[::self.DR], np.zeros(len(self.time_vect))[::self.DR], Lcolor)
        self.lineRawR, = self.axRaw.plot(self.time_vect[::self.DR], np.zeros(len(self.time_vect))[::self.DR], Rcolor)
        self.lineFrqL, = self.axFrq.plot(self.freq_vect[::self.DR], np.ones_like(self.freq_vect)[::self.DR], Lcolor)
        self.lineFrqR, = self.axFrq.plot(self.freq_vect[::self.DR], np.ones_like(self.freq_vect)[::self.DR], Rcolor)
        # top plot
        self.axRaw.set_title("RAW", loc='left')
        self.axRaw.set_ylim(-32768, 32768)
        self.axRaw.set_xlim([0, self.time_vect.max()])
        self.axRaw.set_yticks([0])
        self.axRaw.set_xticks([self.time_vect.max()])
        self.axRaw.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "{:d}ms".format(int(x)) )
        )
        # bot plot
        self.axFrq.set_title("FRQ (rFFT)", loc='left')
        self.axFrq.set_yscale('log')
        self.axFrq.set_ylim(THRESH*5, 1)
        self.axFrq.set_xlim(0, 20000)
        self.axFrq.set_xticks([0, 5000, 10000, 15000, 20000])
        self.axFrq.set_yticks([0.01, 1])
        self.axFrq.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: "{:.2f}".format(y))
        )
        self.axFrq.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "{:d}KHz".format(int(x/1000)) if (x != 0) else "{:d}".format(int(x)))
        )
    

        plt.tight_layout()
        anim = FuncAnimation(self.fig, self.start_listening, interval=int(1000/DEFAULT_FPS)) 
        plt.show()

    def start_listening(self, *args, **kwargs):
        try:
            windows = self.l.get_windows()
            #if np.average(np.abs(windows)) < 5: return

            self.lineRawL.set_ydata(windows[0][::self.DR][::-1]) 
            self.lineRawR.set_ydata(windows[1][::self.DR][::-1]) 
            
            fftL = np.fft.rfft(windows[0], self.l.CHUNK*self.l.WINDOWSIZE, axis=0)
            fftR = np.fft.rfft(windows[1], self.l.CHUNK*self.l.WINDOWSIZE, axis=0)        
            if max(fftL)>self.fftMaxes[0]: self.fftMaxes[0] = max(fftL) 
            if max(fftR)>self.fftMaxes[1]: self.fftMaxes[1] = max(fftR) 
            self.lineFrqL.set_ydata(fftL[::self.DR]/self.fftMaxes[0])
            self.lineFrqR.set_ydata(fftR[::self.DR]/self.fftMaxes[1])
        except e:
            print(e) 
        

if __name__ == '__main__':
    app = Widget(sys.argv[1:])
