import wave
import numpy as np
import os
import matplotlib.pyplot as plt
import binascii

# Function to generate a 256-bit key and save it as a hexadecimal string
def generate_key(filename):
    # Generate a 256-bit key (32 bytes)
    key = np.random.bytes(32)  # 256 bits = 32 bytes
    
    # Convert the key to hexadecimal format
    key_hex = binascii.hexlify(key).decode('utf-8')
    
    key_txt_file = f"{filename}.txt"
    
    # Save the key as a hexadecimal string
    with open(key_txt_file, "w") as f:
        f.write(key_hex)
    print(f"Key saved as hexadecimal to '{key_txt_file}'")
    
    return key

# Function to load the key from a .txt file and convert it back to bytes
def load_key(filename):
    key_txt_file = f"{filename}.txt"
    if not os.path.exists(key_txt_file):
        raise FileNotFoundError(f"Key file '{key_txt_file}' not found.")
    
    # Read the key as a hexadecimal string from the file
    with open(key_txt_file, "r") as f:
        key_hex = f.read().strip()
    
    # Convert the hexadecimal string back to bytes
    key = binascii.unhexlify(key_hex)
    
    return key

# Encryption function using XOR with a dynamic evolving key
def encrypt_signal(signal, key):
    # Convert the key into np.int16 type for XOR compatibility
    key_int16 = np.frombuffer(key, dtype=np.int16)
    
    # Make a copy of the key to modify it (key should be mutable)
    mutable_key = key_int16.copy()
    
    # Set the segment size equal to the length of the key (in 16-bit words)
    segment_size = len(mutable_key)
    
    # Store the encrypted signal
    encrypted_signal = np.zeros_like(signal, dtype=np.int16)
    
    # Iterate over the signal in segments of the length of the key
    for i in range(0, len(signal), segment_size):
        # Get the current segment of the signal
        segment = signal[i:i + segment_size]
        
        # XOR the segment with the current key
        encrypted_segment = segment ^ mutable_key[:len(segment)]
        
        # Store the encrypted segment in the output signal
        encrypted_signal[i:i + segment_size] = encrypted_segment
        
        # Update the mutable key for the next segment (new key is the result of XOR)
        mutable_key[:len(encrypted_segment)] = encrypted_segment
    
    return encrypted_signal


# Decrypt the signal using XOR with a dynamically evolving key (same process as encryption)
def decrypt_signal(encrypted_signal, key):
    # Convert the key into np.int16 type for XOR compatibility
    key_int16 = np.frombuffer(key, dtype=np.int16)
    
    # Make a copy of the key to modify it (key should be mutable)
    mutable_key = key_int16.copy()
    
    # Set the segment size equal to the length of the key (in 16-bit words)
    segment_size = len(mutable_key)
    
    # Store the decrypted signal
    decrypted_signal = np.zeros_like(encrypted_signal, dtype=np.int16)
    
    # Iterate over the encrypted signal in segments of the length of the key
    for i in range(0, len(encrypted_signal), segment_size):
        # Get the current segment of the encrypted signal
        segment = encrypted_signal[i:i + segment_size]
        
        # XOR the segment with the current key to decrypt it
        decrypted_segment = segment ^ mutable_key[:len(segment)]
        
        # Store the decrypted segment in the output signal
        decrypted_signal[i:i + segment_size] = decrypted_segment
        
        # Update the mutable key for the next segment (new key is the result of XOR)
        mutable_key[:len(decrypted_segment)] = segment
    
    return decrypted_signal

# User input for file name
while True:
    try:
        print("Enter the name of the .wav file (without extension): ")
        file = input().strip()
        wav_obj = wave.open(f"{file}.wav", 'rb')
        break
    except FileNotFoundError:
        print(f"File '{file}.wav' not found. Please enter a valid file name.")

# Process wav file
sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
t_audio = n_samples / sample_freq

# Read and prepare signal
signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
times = np.linspace(0, n_samples / sample_freq, num=n_samples)
n_channels = wav_obj.getnchannels()
wav_obj.close()

# Generate and save the key to a .txt file for this file
key = generate_key(file)

# Encrypt signal
encrypted_signal = encrypt_signal(signal_array, key)

# Save encrypted signal to a new .wav file
with wave.open(f"{file}_encrypted.wav", 'wb') as encrypted_wav:
    encrypted_wav.setnchannels(n_channels)
    encrypted_wav.setsampwidth(2)  # 16-bit audio
    encrypted_wav.setframerate(sample_freq)
    encrypted_wav.writeframes(encrypted_signal.tobytes())

print(f"Encrypted audio saved as '{file}_encrypted.wav'")

# Load the key from the .txt file and decrypt the signal
key = load_key(file)
decrypted_signal = decrypt_signal(encrypted_signal, key)

# Save decrypted signal to another .wav file
with wave.open(f"{file}_decrypted.wav", 'wb') as decrypted_wav:
    decrypted_wav.setnchannels(n_channels)
    decrypted_wav.setsampwidth(2)  # 16-bit audio
    decrypted_wav.setframerate(sample_freq)
    decrypted_wav.writeframes(decrypted_signal.tobytes())

print(f"Decrypted audio saved as '{file}_decrypted.wav'")

# Plot original, encrypted, and decrypted signals
if n_channels == 2:
    # Stereo audio - plot each channel separately
    l_channel = signal_array[0::2]
    r_channel = signal_array[1::2]
    l_channel_encrypted = encrypted_signal[0::2]
    r_channel_encrypted = encrypted_signal[1::2]
    l_channel_decrypted = decrypted_signal[0::2]
    r_channel_decrypted = decrypted_signal[1::2]
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 2, 1)
    plt.plot(times, l_channel)
    plt.title(f'{file}.wav - Left Channel (Original)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 2, 2)
    plt.plot(times, r_channel)
    plt.title(f'{file}.wav - Right Channel (Original)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 2, 3)
    plt.plot(times, l_channel_encrypted)
    plt.title(f'{file}.wav - Left Channel (Encrypted)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 2, 4)
    plt.plot(times, r_channel_encrypted)
    plt.title(f'{file}.wav - Right Channel (Encrypted)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 2, 5)
    plt.plot(times, l_channel_decrypted)
    plt.title(f'{file}.wav - Left Channel (Decrypted)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 2, 6)
    plt.plot(times, r_channel_decrypted)
    plt.title(f'{file}.wav - Right Channel (Decrypted)')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
else:
    # Mono audio - plot single signal
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(times, signal_array)
    plt.title(f'{file}.wav - Original Signal')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 1, 2)
    plt.plot(times, encrypted_signal)
    plt.title(f'{file}.wav - Encrypted Signal')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    
    plt.subplot(3, 1, 3)
    plt.plot(times, decrypted_signal)
    plt.title(f'{file}.wav - Decrypted Signal')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)

plt.tight_layout()
plt.show()
