import sounddevice as sd
import soundfile as sf
import queue
import threading
import sys

import matplotlib.pyplot as plt

filename = "merge5.wav"

buffersize = 24
blocksize = 2048
device = None
threshold = 0.8

q = queue.Queue(maxsize=buffersize)
event = threading.Event()

from collections import deque
import numpy as np
import librosa
import pickle
def confidence(model, scaler, data, sr, time):
    feature = []
    y = np.array(data)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr= np.mean(librosa.feature.zero_crossing_rate(y))

    feature.extend([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr])

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    for i, m in enumerate(mfcc):
        feature.append(np.mean(m))

    feature = np.array(feature)

    X = feature.reshape(1,-1)
    X = scaler.transform(X)
    prob = model.predict_proba(X)
    np.set_printoptions(suppress=True)
    prob = prob.flatten()
    column_name = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 
                       26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 
                       40, 41, 42, 43, 44, 45, 46, 5, 6, 7, 8, 9]
    name = np.array(column_name)

    confidence = np.column_stack((name, prob))
    confidence = confidence[confidence[:, 1].argsort()[::-1]]

    if confidence[0,1] > threshold:
        verdict = "  Known voice   "
        # print(f"Known voice: speaker {int(confidence[0,0])}")

    else:
        verdict = " Unknown voice  "

    if time<10:
        print(f"|    {time}    |{verdict}|   speaker {int(confidence[0,0])}   |    {confidence[0,1]:0.2f}    |")
    else:
        print(f"|    {time}   |{verdict}|   speaker {int(confidence[0,0])}   |    {confidence[0,1]:0.2f}    |")

    # print(confidence[0, 1],"\t", confidence[0,0])

def callback(outdata, frames, time, status):
    assert frames == blocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):].fill(0)
        raise sd.CallbackStop
    else:
        # outdata[:] = data.reshape(-1,1)
        outdata[:] = data.reshape(-1,1)

def start(filename):
    with sf.SoundFile(filename) as f:
            for _ in range(buffersize):
                data = f.read(blocksize)
                if not len(data):
                    break
                q.put_nowait(data)  # Pre-fill queue
            stream = sd.OutputStream(
                samplerate=f.samplerate, blocksize=blocksize,
                device=device, channels=1,
                callback=callback, finished_callback=event.set)
            
            with stream:
                Deq = deque(maxlen=buffersize)              # List of data audio input 
                timeout = blocksize * buffersize / f.samplerate
                tracking = 0

                with open('models\model_10.pkl', 'rb') as a:
                    clf_10 = pickle.load(a)

                with open('models\scaler.pkl', 'rb') as b:
                    scaler = pickle.load(b)

                flow, sr = librosa.load(filename)
                current_frame = 0
                start = current_frame
                time = 0
                while len(data):
                    data = f.read(blocksize)
                    q.put(data)
                    tracking += 1
                    Deq.append(data)

                    end = current_frame + blocksize
                    current_frame += blocksize
                    current_flow = flow[start:end]

                    if(tracking == int(f.samplerate/blocksize)*1):
                        tracking = 0
                        real_shit = np.array(Deq)
                        real_shit = real_shit.flatten()
                        time = time + 1
                        # confidence(clf_10, scaler, real_shit, f.samplerate)
                        confidence(clf_10, scaler, current_flow, sr , time)
                        start = current_frame

                print("+---------+----------------+----------------+------------+")
                event.wait()  # Wait until playback is finished

print("+---------+----------------+----------------+------------+")
print("|Time(sec)|     Result     |   Best match   | Confidence |")
print("+=========+================+================+============+")
start(filename)