import librosa
import numpy as np

from hparams import HParams

class AudioProcessor:
    def __init__(self, hparams: HParams = None):
        self.hparams = hparams if hparams is not None else HParams()

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.hparams.sample_rate)
        return wav

    def wav_to_mel(self, wav):
        # STFT
        stft = librosa.stft(
            wav,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length
        )

        spectrogram = np.abs(stft)

        # Mel filter
        mel_filter = librosa.filters.mel(
            sr=self.hparams.sample_rate,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_mels,
            fmin=self.hparams.fmin,
            fmax=self.hparams.fmax
        )

        mel = np.dot(mel_filter, spectrogram)

        # log scaling
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

        return mel

    def wav_to_mel_from_path(self, path):
        wav = self.load_wav(path)
        return self.wav_to_mel(wav)
    
    def mel_to_wav(self, mel, n_iter=200):
        # Reverse log to exp
        mel = np.exp(mel)

        # Mel to linear spectrogram
        mel_filter = librosa.filters.mel(
            sr=self.hparams.sample_rate,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_mels,
            fmin=self.hparams.fmin,
            fmax=self.hparams.fmax
        )

        inv_mel_filter = np.linalg.pinv(mel_filter)
        spectrogram = np.dot(inv_mel_filter, mel)

        # Griffin-Lim
        wav = librosa.griffinlim(
            spectrogram,
            n_iter=n_iter,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length
        )

        return wav