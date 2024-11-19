from flask import Flask, render_template, request, redirect, url_for
import os
from encode_decode import encode_image, decode_image, train_lstm_for_message, load_model_from_file, save_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths for image storage
UPLOAD_FOLDER = 'static/uploads/'
ENCODED_FOLDER = 'static/encoded/'

# Make sure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENCODED_FOLDER, exist_ok=True)

# File upload settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize the LSTM model (load or train)
LSTM_MODEL_PATH = 'models/lstm_model.h5'

# Load or train model
if os.path.exists(LSTM_MODEL_PATH):
    model = load_model_from_file(LSTM_MODEL_PATH)
else:
    model = train_lstm_for_message("Sample Message")  # You could train with any sample message
    save_model(model, LSTM_MODEL_PATH)

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files or 'message' not in request.form:
        return redirect(request.url)
    
    image = request.files['image']
    message = request.form['message']
    
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Encode the message into the image
        encoded_image_path = os.path.join(ENCODED_FOLDER, f"encoded_{filename}")
        encode_image(image_path, message, encoded_image_path, model)
        
        return redirect(url_for('encoded_image', filename=f"encoded_{filename}"))
    return "Invalid file format"

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image = request.files['image']
    
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Decode the message from the image
        decoded_message = decode_image(image_path, model)
        
        return render_template('decoded_message.html', message=decoded_message)
    return "Invalid file format"

@app.route('/encoded_image/<filename>')
def encoded_image(filename):
    return render_template('encoded_image.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
