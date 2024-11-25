import librosa
import pandas as pd
import numpy as np
import os
import wave
import struct
import soundfile as sf

list_dicts = []
audio_root_dir = "46_speaker_original\Train"

for audio_subfolder in os.listdir(audio_root_dir):
  audio_subfolder_path = os.path.join(audio_root_dir, audio_subfolder)

  for audio_file in os.listdir(audio_subfolder_path):
    file_data = {}
    audio_file_path = os.path.join(audio_subfolder_path, audio_file)
    y, sr = librosa.load(audio_file_path)
    # sr = 16000
    file_data['label'] = int(audio_subfolder[-2:])
    file_data['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    file_data['rmse'] = np.mean(librosa.feature.rms(y=y))
    file_data['spec_cent'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))    
    file_data['spec_bw'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    file_data['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    file_data['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    for i, m in enumerate(mfcc):
      file_data['m'+str(i)] = np.mean(m)
    list_dicts.append(file_data)

  print('class',audio_subfolder,'completed')

data_frame = pd.DataFrame(list_dicts)
data_frame.to_csv(f'Train_speaker_original.csv')
 