import sys
import time
import atexit
import warnings

import threading
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyaudio

DEFAULT_DEVICE = 'Stereo Mix (Realtek'
DEFAULT_FPS = 30
DEFAULT_WINDOWSIZE = 5

class Listener:
    def __init__(self, device)
        self.DEVICE = device
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100 
        self.FPS = DEFAULT_FPS
        self.INTERVAL = 1000/self.FPS
        self.CHUNK = self.RATE/self.FPS 
        self.WINDOWSIZE = DEFAULT_WINDOWSIZE
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(input=True,
                              channels=2
                              format=self.FORMAT,
                              rate=self.RATE,
                              frames_per_buffer=self.CHUNK,
                              stream_callback=self.new_window)
        self.lock = threading.Lock()
        self.stop = False
        self.windows = [np.zeros(CHUNK*WINDOWSIZE) for i in range(self.CHANNELS)]
        atexit.register(self.close)

    def new_window(self, data)
        data = np.fromstring(data, 'int16')
        with self.lock:
            for i in range(self.CHANNELS):
                self.windows[i].append(data[i::self.CHANNELS])
                self.windows[i] = self.windows[i][self.CHUNK:]
                if self.stop: return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
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
        self.init_fig()
        self.start_listening()

    def init_pyaudio(self, args):
        # get DEVICE from arg
        if not args: DEVICE = DEFAULT_DEVICE
        else: DEVICE = args[0]
        l = Listener(DEVICE)
        l.start()
        
    def init_fig(self):
        pass

    def start_listening(self):
        pass


if __name__ == '__main__':
    Widget(sys.argv[1:])
