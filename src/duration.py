import os
import numpy as np
import torch
from typing import Optional

def get_naive_durations(text_length: int, mel_frames: int) -> torch.Tensor:
    """
    Calculates a naive proportional duration for each text token.
    This assumes every character takes roughly an equal amount of frames.
    While inaccurate for high-quality speech, it serves as a baseline 
    to make the pipeline runnable without external aligners.
    """
    if text_length == 0:
        return torch.tensor([])
        
    duration = np.full(text_length, mel_frames // text_length)
    remainder = mel_frames % text_length
    
    # Distribute the remainder evenly across the first few tokens
    for i in range(remainder):
        duration[i] += 1
        
    return torch.LongTensor(duration)

def load_preextracted_durations(file_id: str, alignments_dir: str) -> torch.Tensor:
    """
    Placeholder for loading real durations extracted via MFA (.npy or .txt).
    If a pre-extracted duration file exists, it returns those. 
    Otherwise returns None.
    """
    dur_path = os.path.join(alignments_dir, f"{file_id}.npy")
    if os.path.exists(dur_path):
        dur = np.load(dur_path)
        return torch.LongTensor(dur)
    return None

def extract_durations_for_item(text_seq: list[int], mel_tensor: torch.Tensor, file_id: Optional[str] = None, alignments_dir: Optional[str] = None) -> torch.Tensor:
    """
    Attempts to load real durations; falls back to naive proportional durations.
    """
    mel_frames = mel_tensor.shape[1] if len(mel_tensor.shape) > 1 else mel_tensor.shape[0]
    text_length = len(text_seq)
    
    if alignments_dir and file_id:
        real_durations = load_preextracted_durations(file_id, alignments_dir)
        if real_durations is not None:
            # Ensure summation matches exactly for the pipeline
            if real_durations.sum() == mel_frames and len(real_durations) == text_length:
                return real_durations
            else:
                s = real_durations.sum()
                diff = mel_frames - s
                if diff > 0:
                    real_durations[-1] += diff
                elif diff < 0:
                    real_durations[-1] = max(0, real_durations[-1] + diff)
                return real_durations

    return get_naive_durations(text_length, mel_frames)

if __name__ == "__main__":
    # Test naive duration
    seq_len = 10
    mel_len = 54
    durs = get_naive_durations(seq_len, mel_len)
    print(f"Sequence length: {seq_len}, Mel Frames: {mel_len}")
    print(f"Generated Naive Durations: {durs}")
    print(f"Sum of durations: {durs.sum().item()} (Should equal Mel Frames: {mel_len})")
