from PIL import Image
import numpy as np
from tensorflow import keras
from keras import layers


# Create the LSTM model for binary conversion
def create_lstm_model(input_shape):
    model = keras.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output a single binary value
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_for_message(message):
    """Train an LSTM model to convert message characters to binary."""
    # Map characters to integers
    char_to_int = {chr(i): i for i in range(256)}
    encoded_message = [char_to_int[char] for char in message]
    encoded_message = np.array(encoded_message).reshape(-1, 1)  # Reshape for LSTM input

    # Prepare the output labels (binary representation)
    binary_output = []
    for char in encoded_message:
        binary_output.append(1 if char[0] % 2 == 1 else 0)  # Example: odd/even mapping
    binary_output = np.array(binary_output).reshape(-1, 1)

    # Create and train the LSTM model
    model = create_lstm_model((1, 1))
    model.fit(encoded_message, binary_output, epochs=10, verbose=0)
    return model

def message_to_binary_with_lstm(message, model):
    """Convert a string message to a binary string using a trained LSTM model."""
    char_to_int = {chr(i): i for i in range(256)}
    encoded_message = [char_to_int[char] for char in message]
    encoded_message = np.array(encoded_message).reshape(-1, 1)

    binary_message_output = []
    for char in encoded_message:
        prediction = model.predict(np.array([[char]]))
        binary_message_output.append(int(np.round(prediction[0][0])))

    binary_string = ''.join(map(str, binary_message_output)) + '00000000'  # Add null byte
    return binary_string

def binary_to_message_with_lstm(binary_string, model):
    """Convert a binary string to a message using a trained LSTM model."""
    binary_list = [int(bit) for bit in binary_string]
    binary_array = np.array(binary_list).reshape(-1, 1)

    decoded_chars = []
    for binary_value in binary_array:
        prediction = model.predict(np.array([[binary_value]]))
        decoded_char = chr(int(np.round(prediction[0][0])))
        decoded_chars.append(decoded_char)

    decoded_message = ''.join(decoded_chars)
    return decoded_message

def encode_image(image_path, message, output_path, model):
    """Encode a message into an image using LSB steganography."""
    # Load the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB format
    width, height = img.size

    # Convert the message to binary using LSTM
    binary_message = message_to_binary_with_lstm(message, model)
    message_length = len(binary_message)

    # Check if the image can hold the message
    if message_length > width * height * 3:
        raise ValueError("Message is too long to be encoded in this image.")

    # Encode the message into the image
    data_index = 0
    for y in range(height):
        for x in range(width):
            if data_index < message_length:
                r, g, b = img.getpixel((x, y))

                # Modify the least significant bit of each color channel
                if data_index < message_length:
                    r = (r & ~1) | int(binary_message[data_index])
                    data_index += 1
                if data_index < message_length:
                    g = (g & ~1) | int(binary_message[data_index])
                    data_index += 1
                if data_index < message_length:
                    b = (b & ~1) | int(binary_message[data_index])
                    data_index += 1

                img.putpixel((x, y), (r, g, b))

            if data_index >= message_length:
                break

    img.save(output_path)
    print(f"Message encoded and saved to {output_path}")

def decode_image(image_path, model):
    """Decode a message from an image using LSB steganography."""
    img = Image.open(image_path)
    img = img.convert('RGB')
    width, height = img.size

    binary_message = ''
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            binary_message += str(r & 1)
            binary_message += str(g & 1)
            binary_message += str(b & 1)

    binary_message = binary_message.split('00000000')[0]
    decoded_message = binary_to_message_with_lstm(binary_message, model)
    return decoded_message

if __name__ == "__main__":
    image_path = input("Enter the path to the cover image: ")
    message = input("Enter the message to encode: ")
    output_path = "encoded_image.png"

    lstm_model = train_lstm_for_message(message)

    # Encode the message
    encode_image(image_path, message, output_path, lstm_model)

    # Decode the message
    decoded_message = decode_image(output_path, lstm_model)
    print(f"Decoded Message: {decoded_message}")
