import os
import librosa
import numpy as np


class AUDIO_LOADER:
    def __init__(self, config):
        self.config = config
        
    def get_batch(self, x):
        features, labels = [], []
        for i in x:
            aud, _ = librosa.load(x, sr=self.config)
            features.append(aud)
            
            