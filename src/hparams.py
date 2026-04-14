from dataclasses import dataclass

@dataclass
class HParams:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000
