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
warnings.filterwarnings('ignore') # suppress numpy warning... why is this showing up?
    
# WIDGET OPTIONS
DEFAULT_DEVICE = 'Stereo Mix (Realtek'
DEFAULT_FPS = int(60)
DEFAULT_WINDOWSIZE = int(2*2)
RESOLUTION = int(512)
NBINS = int(16*4)

# WIDGET COLOR
FCOLOR = 'black'
BCOLOR = 'white'
mpl.rcParams['text.color'] = BCOLOR
mpl.rcParams['figure.facecolor'] = FCOLOR
mpl.rcParams['figure.edgecolor'] = BCOLOR
mpl.rcParams['figure.subplot.left'] = 0.06 
mpl.rcParams['figure.subplot.right'] = 0.94 
mpl.rcParams['figure.subplot.bottom'] = 0.06
mpl.rcParams['figure.subplot.top'] = 0.94
mpl.rcParams['figure.subplot.wspace'] = 0 
mpl.rcParams['figure.subplot.hspace'] = 0 

mpl.rcParams['legend.facecolor'] = FCOLOR
mpl.rcParams['legend.edgecolor'] = FCOLOR
mpl.rcParams['legend.framealpha'] = 0
mpl.rcParams['axes.facecolor'] = FCOLOR
mpl.rcParams['axes.titlepad'] = 0 
mpl.rcParams['axes.edgecolor'] = BCOLOR
mpl.rcParams['xtick.major.pad'] = 0 
mpl.rcParams['ytick.major.pad'] = 0 
mpl.rcParams['ytick.color'] = BCOLOR
mpl.rcParams['xtick.color'] = BCOLOR
mpl.rcParams['axes.spines.bottom'] = False 
mpl.rcParams['axes.spines.right'] = False 
mpl.rcParams['axes.spines.left'] = False 
mpl.rcParams['lines.linewidth'] = 1 
mpl.rcParams['toolbar'] = 'None'

class Listener:
    def __init__(self, device):
        self.FORMAT = pyaudio.paInt16
        self.DEVICE = device
        self.RATE = 44100 
        self.FPS = DEFAULT_FPS
        self.CHANNELS = 2 
        self.INTERVAL = int(1000/self.FPS)
        self.CHUNK = int(self.RATE/self.FPS)
        self.WINDOWSIZE = DEFAULT_WINDOWSIZE
        self.p = pyaudio.PyAudio()
        #self.p.get_device_info_by_index(self.)
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

    def fig_resized(self, *args, **kwargs):
        self.size = self.fig.get_size_inches()*self.fig.dpi

    def init_pyaudio(self, args):
        if not args: self.DEVICE = DEFAULT_DEVICE
        else: self.DEVICE = args[0]
        self.l = Listener(self.DEVICE)
        self.l.start()
        self.n = self.l.CHUNK*self.l.WINDOWSIZE
        
    def init_fig(self, args):
        Lcolor = 'red'; Rcolor = 'blue';
        self.DR = self.n//RESOLUTION
        THRESH = 1/self.l.RATE
        self.freq_vect = np.fft.fftfreq(self.n, 1./self.l.RATE)[:int(self.n/2)]
        self.time_vect = np.arange(self.n, dtype=np.float32) / self.l.RATE * 1000
        self.fftMaxes = [THRESH,THRESH]        
        self.t_ref = time.time()

        # plot objects        
        self.fig = plt.figure()
        self.fig.set_size_inches(1300/200,800/200)
        self.size = self.fig.get_size_inches()*self.fig.dpi # size in pixels
        self.cid = self.fig.canvas.mpl_connect('resize_event', self.fig_resized)
        self.BARWIDTH = self.size[0]*0.5/NBINS*self.fig.dpi/2

        self.cutoff = 0
        self.barY = self.freq_vect[::len(self.freq_vect)//NBINS]
        self.barX = np.linspace(20,max(self.freq_vect),NBINS)
        if len(self.barY) != len(self.barX): self.cutoff = abs(len(self.barY)-len(self.barX))
        self.barY = self.barY[:-self.cutoff]
        assert len(self.barY) == len(self.barX)
 
        self.axRaw = plt.subplot2grid((10,1),(0,0), rowspan=3)
        self.axFrq = plt.subplot2grid((10,1),(4,0), rowspan=4) 
        self.axBar = plt.subplot2grid((10,1),(9,0), rowspan=1) 
        self.lineRawL, = self.axRaw.plot(self.time_vect[::self.DR*2], np.zeros(len(self.time_vect))[::self.DR*2], Lcolor, label='$L_{r}$')
        self.lineRawR, = self.axRaw.plot(self.time_vect[::self.DR*2], np.zeros(len(self.time_vect))[::self.DR*2], Rcolor, label='$R_{r}$')
        self.lineFrqL, = self.axFrq.plot(self.freq_vect[::self.DR], np.ones_like(self.freq_vect)[::self.DR], Lcolor, label='$L_{f}$')
        self.lineFrqR, = self.axFrq.plot(self.freq_vect[::self.DR], np.ones_like(self.freq_vect)[::self.DR], Rcolor, label='$R_{f}$')

        self.barFrqL = self.axBar.bar(self.barX, self.barY, width=self.BARWIDTH, color='magenta', alpha=0.8)
        self.barFrqR = self.axBar.bar(self.barX, self.barY, width=self.BARWIDTH, color='cyan', alpha=0.8)

        # top plot
        ylim = 32768/2
        self.axRaw.set_title("RAW - {})".format(self.DEVICE), loc='left')
        self.axRaw.set_ylim(-ylim, ylim)
        self.axRaw.set_xlim([0, self.time_vect.max()])
        self.axRaw.set_yticks([0])
        self.axRaw.set_xticks([self.time_vect.max()])
        self.axRaw.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "{:d}ms".format(int(x)) )
        )
        # bot plot
        self.axFrq.set_title("FRQ", loc='left')
        self.axFrq.set_ylim(THRESH*5, 1)
        self.axFrq.set_xlim(20, 20000)
        self.axFrq.set_xticks([])
        self.axFrq.set_yticks([0.01, 1])
        self.axFrq.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: "{:.2f}".format(y))
        )
        self.axBar.set_yscale('log')
        self.axBar.set_title('',pad=0)
        self.axBar.spines['top'].set_visible(False)
        self.axBar.set_ylim(THRESH*5, 1)
        self.axBar.set_xlim(20, 20000)
        self.axBar.set_xticks([20, 5000, 20000])
        self.axBar.set_yticks([])
        self.axBar.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "{:d}KHz".format(int(x/1000)) if (x != 0) else "{:d}".format(int(x)))
        )

        # xy location of FPS text
        sc = self.size[0]/self.fig.dpi/6.5
        #print(sc) #DO RELATIVE TXT LOCATION TODO

        self.textFPSx = (0.90*sc)*max(self.time_vect)
        self.textFPSy = 1.10*ylim
        self.textFPS = self.axRaw.text(self.textFPSx, self.textFPSy, 'FPS: 000')
    
        plt.legend(handles=[self.lineFrqL, self.lineFrqR], loc=((0.90*sc), (4.5*sc)))
        #plt.subplots_adjust(bottom=0.1, left=0.1)
        #plt.tight_layout()
        anim = FuncAnimation(self.fig, self.start_listening, self.listen_data, \
            interval=int(1000/DEFAULT_FPS), blit=True)
        plt.show()

    def listen_data(self):
        while True:
            windows = self.l.get_windows()
            yield windows

    def start_listening(self, windows, *args, **kwargs):
        try:
            # show top axes lines
            self.lineRawL.set_ydata(windows[0][::self.DR*2][::-1]) 
            self.lineRawR.set_ydata(windows[1][::self.DR*2][::-1]) 
            # show bottom axes lines
            norm = None #"ortho"
            fftL = np.fft.fft(windows[0], self.n, axis=0, norm=norm)
            fftR = np.fft.fft(windows[1], self.n, axis=0, norm=norm)        
            if fftL.max()>self.fftMaxes[0]: self.fftMaxes[0] = fftL.max() 
            if fftR.max()>self.fftMaxes[1]: self.fftMaxes[1] = fftR.max() 
            fftLNorm = abs(fftL)[:int(self.n/2)]/self.fftMaxes[0]
            fftRNorm = abs(fftR)[:int(self.n/2)]/self.fftMaxes[1]
            self.lineFrqL.set_ydata(fftLNorm[::self.DR])
            self.lineFrqR.set_ydata(fftRNorm[::self.DR])
            # update bar rectangles
            for bar, h in zip(self.barFrqL, fftLNorm[::len(fftLNorm)//NBINS][:-self.cutoff]):
                bar.set_height(h)
            for bar, h in zip(self.barFrqR, fftRNorm[::len(fftRNorm)//NBINS][:-self.cutoff]):
                bar.set_height(h)
            # calc FPS and show
            delta_t = time.time()-self.t_ref
            self.t_ref = time.time()
            self.textFPS.set_text('FPS: {0:0=3d}'.format(int(1/delta_t)))
            # return value
            artObj = tuple([self.lineRawL, self.lineRawR, self.lineFrqL, \
                self.lineFrqR ]) + tuple(self.barFrqL) + tuple(self.barFrqR)
            return artObj
            
        except Exception as e:
            print(e) 
        

if __name__ == '__main__':
    app = Widget(sys.argv[1:])
