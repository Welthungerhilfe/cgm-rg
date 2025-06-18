from io import BytesIO
from utils.constants import DECRYPTION_KEY as KEY

BUFFER_SIZE = 8192


def decrypt_file_data(raw_file):
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
