import pyaudio
import time
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal



FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])

SPEAKERS = p.get_default_output_device_info()["hostApi"]

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames 
    audio_data = np.fromstring(in_data, dtype=np.float32)
    dry_data = np.append(dry_data,audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pyaudio.paContinue)

def main():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_host_api_specific_stream_info=SPEAKERS,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(10)
        stream.stop_stream()
    stream.close()

    numpydata = np.hstack(fulldata)
    plt.plot(numpydata)
    plt.title("Wet")
    plt.show()


    numpydata = np.hstack(dry_data)
    plt.plot(numpydata)
    plt.title("Dry")
    plt.show()


    p.terminate()


main()

