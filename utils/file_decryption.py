from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import hashlib

from io import BytesIO
from utils.constants import DECRYPTION_KEY as KEY
from utils.constants import AES_KEY as passphrase

BUFFER_SIZE = 8192


def decrypt_xor_response_content(raw_file):
    try:
        input_stream = BytesIO(raw_file)
        output = BytesIO()
    
        key_index = 0
        buffer = bytearray(BUFFER_SIZE)
    
        while True:
            bytes_read = input_stream.readinto(buffer)
            if bytes_read == 0:
                break
            for i in range(bytes_read):
                buffer[i] ^= KEY[key_index % len(KEY)]
                key_index += 1
            output.write(buffer[:bytes_read])
    
        decrypted_data = output.getvalue()
        return decrypted_data
    except Exception as error:
        raise error


def decrypt_aes_response_content(encrypted_data_b64: bytes) -> bytes:
    # Step 1: Decode Base64
    encrypted_data = base64.b64decode(encrypted_data_b64)

    # Step 2: Extract IV (first 12 bytes)
    iv = encrypted_data[:12]
    ciphertext = encrypted_data[12:]

    # Step 3: Generate AES key from passphrase
    key = hashlib.sha256(passphrase.encode('utf-8')).digest()

    # Step 4: Decrypt the data
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext, None)  # No associated data

    return plaintext
