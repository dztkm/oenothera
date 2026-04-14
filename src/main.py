import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from audioProcessor import AudioProcessor 
from hparams import HParams
import pandas as pd
import os

def load_metadata(metadata_path, wavs_dir):
    data = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            file_id = parts[0]
            text = parts[1]

            audio_path = os.path.join(wavs_dir, f"{file_id}.wav")

            data.append({
                "id": file_id,
                "text": text,
                "audio_path": audio_path
            })

    return pd.DataFrame(data)


# Example usage
df = load_metadata("data/metadata.csv", "data/wavs")
print(df.head())

ap = AudioProcessor(HParams())

# Get single file
sample = df.iloc[0]
audio_path = sample["audio_path"]

# Audio to mel
mel = ap.wav_to_mel_from_path(audio_path)

print("Mel shape:", mel.shape)



plt.imshow(mel, aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.colorbar()
plt.show()

# Mel to audio
reconstructed_wav = ap.mel_to_wav(mel)

# Save
# input_wav, sr = sf.read(audio_path)
# sf.write("input.wav", input_wav, ap.hparams.sample_rate)
sf.write("reconstructed.wav", reconstructed_wav, ap.hparams.sample_rate)


