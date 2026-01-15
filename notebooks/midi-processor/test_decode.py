import midiprocessor as mp

# Read one token file
token_file = '0a0b677f20c5fc6eb9b69c55adec1920.mid.txt'
with open(token_file, 'r') as f:
    token_str = f.read().strip()

print(f"Token string (first 100 chars): {token_str[:100]}")

# Create decoder
decoder = mp.MidiDecoder('REMIGEN')

# Decode
tokens_list = token_str.split()
print(f"Total tokens: {len(tokens_list)}")
print(f"First 20 tokens: {tokens_list[:20]}")

# Decode to MIDI
midi_obj = decoder.decode_from_token_str_list(tokens_list)

# Save
midi_obj.dump('test_output.mid')
print("✅ Successfully decoded to test_output.mid!")