import os
import argparse
import datetime
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from text import text_to_sequence
from model import TransformerTTS
from hparams import HParams
from audioProcessor import AudioProcessor

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hp = HParams()
    model = TransformerTTS(hp).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess text
    sequence = text_to_sequence(args.text)
    text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)

    # Inference yielding mel
    print("Running inference...")
    mel_postnet, stop_tokens = model.inference(text_tensor)
    
    # Extract the mel (batch=0) and transpose to (n_mels, TIME)
    # The mel output is (N, TIME, n_mels)
    mel = mel_postnet[0].transpose(0, 1).cpu().numpy()

    # Generate wav using Griffin-Lim
    print("Synthesizing audio...")
    ap = AudioProcessor(hp)
    wav = ap.mel_to_wav(mel)

    # Convert to 16-bit PCM for standard WAV output
    wav_normalized = wav / np.max(np.abs(wav))
    wav_int16 = np.int16(wav_normalized * 32767)

    # Save logic
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"synthesis_{timestamp}.wav"
    output_path = os.path.join(args.output_dir, filename)

    wavfile.write(output_path, hp.sample_rate, wav_int16)
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model synthesis pipeline")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, default="synthesis_output", help="Directory for output wav")
    args = parser.parse_args()
    infer(args)
