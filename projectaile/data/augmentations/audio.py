import librosa
import numpy as np

'''
    add_noise : add random noise to the audio signal
'''
def add_noise(audio, noise_dir, snr):
    return

'''
    db_filter
'''
def db_filter():
    return

'''
    smooth
'''
def smooth(audio):
    b, a = signal.butter(5, 0.45)
    condn = signal.lfilter_zi(b, a)
    smoothed, _ = signal.lfilter(b, a, audio, zi=condn*audio[0])
    return smoothed

'''
    resample : resample the audio to a different sample rate
    
    audio : audio signal samples
    sr : sample rate of the audio
    target_sr : target sample rate
    
    returns : resampled audio with new sample rate as target sample rate
'''
def resample(audio, sr, target_sr):
    return librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)