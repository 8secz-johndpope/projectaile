import librosa
import numpy as np


'''
    mel_spectrogram : Returns the mel scale spectrogram from either the audio signal
                    or the spectrogram
    
    aud : audio signal ( not required if spec is provided )
    spec : spectrogram ( not required if aud is provided )
    sr : sample rate ( sample rate of the audio signal )
    n_fft : length of fft window
    hop_length : number of samples between successive frames
    power : exponent to obtain the mel spectrogram, 1 for energy, 2 for power
    
    returns : mel spectrogram
'''
def mel_spectrogram(aud=None, spec=None, sr=22050, n_fft=2048, hop_length=512, power=2.0):
    if aud is None and spec is None:
        print('Need atleast one of the audio signal or the spectrogram')
        exit(0)
    
    if spec:
        return librosa.feature.melspectrogram(sr=sr, S=spec, n_fft=n_fft, hop_length=hop_length, power=power)
    
    if aud and type(aud) == np.ndarray:
        return librosa.feature.melspectrogram(y=aud, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power)
    elif aud and type(aud) == str:
        aud, sr = librosa.load(aud, sr=sr)
        return librosa.feature.melspectrogram(y=aud, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power)
    

'''
    normalize : Normalizes the spectrogram between min and ref db
    
    spec : spectrogram
    min_db : minimum db value
    ref_db : reference db value
    
    returns : normalized spectrogram
'''
def normalize(spec, min_db=-100, ref_db=20):
    spec = 20.0 * np.log10(np.maximum(1e-5, spec)) - ref_db
    return np.clip(spec / -min_db, -1.0, 0.0) + 1.0


'''
    denormalize : retrieves original spectrogram from normalized spectrogram
    
    spec : normalized spectrogram
    min_db : minimum db value
    ref_db : reference db value
    
    returns : original spectrogram
'''
def denormalize(spec, min_db=-100, ref_db=20):
    spec = (np.clip(spec, 0.0, 1.0) - 1.0) * -min_db
    spec = spec + ref_db
    spec = np.power(10.0, spec * 0.05)
    return spec


'''
    phase_addition : add phase information back to the generated wave from spectrogram
    
    aud : waveform
    phase_iters : number of iterations for phase additions
    n_fft : number of fft frames
    
    returns : phase added signal
'''
def phase_addition(aud, phase_iters=10, n_fft=1024):
    sample_len = len(aud)
    inp = np.random.randn(sample_len)

    for i in range(phase_iters):
        angle = librosa.stft(inp, n_fft=n_fft)

        if np.shape(angle)[1] < aud.shape[1]:
            angle = np.concatenate((angle, np.zeros((aud.shape[0], aud.shape[1]-np.shape(angle)[1]))), axis=-1)

        full = aud * np.exp(1j * np.angle(angle))
        inp = librosa.istft(full)

    return inp


'''
    length_normalize : normalize the length of audio signal
    
    audio : the audio signal
    sr : sample rate
    max_time : maximum time of the audio in seconds
    
    return : length normalized audio signal
'''
def length_normalize(audio, sr=16000, max_time=0.3):
    sample_len = int(max_time * sr)
    if len(audio) < sample_len:
        sample = []
        while len(sample) < sample_len:
            sample += list(audio)

        audio = np.array(sample, dtype=np.float32)

    audio = audio[:sample_len]
    return audio


'''
------------others---------------
'''