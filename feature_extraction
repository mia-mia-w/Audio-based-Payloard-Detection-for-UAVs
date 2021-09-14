import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue

def extract_feature(file_name=None):
    if file_name:
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True:
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    # tempogram
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=X, sr=sample_rate, hop_length=hop_length)
    tempograms = librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate, hop_length=hop_length)
    tempogram = np.mean(tempograms.T, axis=0)

    # fourier_tempogram
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=X, sr=sample_rate, hop_length=hop_length)
    fourier_tempograms = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sample_rate,
                                                           hop_length=hop_length)
    fourier_tempograms = fourier_tempograms.real
    fourier_tempogram = np.mean(fourier_tempograms.T, axis=0)

    # when changing the method, the shape is also need to be changed
    # mfcc:40; chroma: 12; mel: 128; contrast: 7; tonnetz: 6; tempogram: 384; fourier_tempogram: 193
    return tonnetz

def parse_audio_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,6)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                # when trying other combinations, change the methods here
                try: tonnetz = extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn,e))
                    continue
                # when trying other combinations, change the methods here
                ext_features = np.hstack([tonnetz])
                features = np.vstack([features,ext_features])
                # labels = np.append(labels, fn.split('/')[1])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

# def parse_predict_files(parent_dir,file_ext='*.ogg'):
#     features = np.empty((0,193))
#     filenames = []
#     for fn in glob.glob(os.path.join(parent_dir, file_ext)):
#         # when trying other combinations, change the methods here
#         mfccs, mel, chroma, contrast, tonnetz = extract_feature(fn)
#         ext_features = np.hstack([mfccs, mel, chroma, contrast, tonnetz])
#         features = np.vstack([features,ext_features])
#         filenames.append(fn)
#         print("extract %s features done")
#     return np.array(features), np.array(filenames)

def main():
    # Get features and labels
    features, labels = parse_audio_files('../test_audio_loaded_unloaded')
    np.save('../feature_results/loaded_vs_unloaded/feat_tonnetz.npy', features)
    np.save('../feature_results/loaded_vs_unloaded/label_tonnetz.npy', labels)

    # Predict new
    # features, filenames = parse_predict_files('predict')
    # np.save('predict_feat.npy', features)
    # np.save('predict_filenames.npy', filenames)

if __name__ == '__main__': main()
