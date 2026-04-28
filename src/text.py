import re

# Define character set for English character-level mapping
# PAD is 0, EOS could be 1, but we usually just map characters directly.
# Let's define a simple vocabulary
_pad        = '_'
_punctuation = '!\'(),.:;? '
_letters = 'abcdefghijklmnopqrstuvwxyz'

# The complete symbol set
symbols = [_pad] + list(_punctuation) + list(_letters)

# Mappings from symbol to ID and ID to symbol
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def get_vocab_size():
    return len(symbols)

def clean_text(text: str) -> str:
    """
    Cleans raw text: lowercase, removes unhandled characters.
    """
    text = text.lower()
    # Replace weird characters with standard ones (very simplified)
    text = re.sub(r'[\-\[\]&]', ' ', text)
    # Keep only characters in our dictionary
    text = ''.join([c for c in text if c in _symbol_to_id])
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def text_to_sequence(text: str) -> list[int]:
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    """
    cleaned = clean_text(text)
    sequence = [_symbol_to_id[c] for c in cleaned if c in _symbol_to_id]
    return sequence

def sequence_to_text(sequence: list[int]) -> str:
    """
    Converts a sequence of IDs back to a string.
    """
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += str(_id_to_symbol[symbol_id])
    return result

if __name__ == "__main__":
    sample_text = "Hello, world! This is a test for Mini SpeedySpeech."
    seq = text_to_sequence(sample_text)
    print(f"Original: {sample_text}")
    print(f"Sequence: {seq}")
    print(f"Recovered: {sequence_to_text(seq)}")
    print(f"Vocab size: {get_vocab_size()}")
