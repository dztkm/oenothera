from dataclasses import dataclass

@dataclass
class HParams:
    # Audio
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000

    # Model
    text_num_embeddings: int = 256
    embedding_size: int = 256
    encoder_embedding_size: int = 256
    dim_feedforward: int = 1024
    encoder_kernel_size: int = 5
    postnet_embedding_size: int = 512
    postnet_kernel_size: int = 5
    max_mel_time: int = 2000
