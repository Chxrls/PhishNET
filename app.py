import os
import time
import zipfile
import shutil
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global state to track what is currently loaded in memory
current_model_name = None
tokenizer = None
model = None
label_map = {0: "Safe", 1: "Phishing"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    # List folders in the models directory
    models = [item for item in os.listdir(app.config['UPLOAD_FOLDER']) 
              if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], item))]
    return jsonify({'models': models})

@app.route('/upload', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a .zip file'}), 400
    
    filename = secure_filename(file.filename)
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(zip_path)
    
    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4])
    os.makedirs(extract_folder, exist_ok=True)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    os.remove(zip_path) 
    
    # Verify required files
    files = os.listdir(extract_folder)
    has_config = 'config.json' in files
    has_weights = 'model.safetensors' in files or 'pytorch_model.bin' in files
    
    if not (has_config and has_weights):
        shutil.rmtree(extract_folder)
        return jsonify({'error': 'Verification failed. Missing config.json or weights.'}), 400
        
    return jsonify({'message': f'Model {filename[:-4]} uploaded and verified.'})

@app.route('/predict', methods=['POST'])
def predict():
    global current_model_name, tokenizer, model
    data = request.get_json()
    email_text = data.get('email_text', '')
    requested_model = data.get('model', '')

    if not email_text or not requested_model:
        return jsonify({'error': 'Missing text or model selection'}), 400

    start_time = time.time()
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], requested_model)
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    loaded_fresh = False
    # Only load if a new model is selected or nothing is loaded yet
    if current_model_name != requested_model:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            current_model_name = requested_model
            loaded_fresh = True
        except Exception as e:
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

    # Inference
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1).squeeze()
    
    confidence, predicted_class = torch.max(probabilities, dim=0)
    prediction_label = label_map[predicted_class.item()]
    
    inference_time = round(time.time() - start_time, 3)

    return jsonify({
        'prediction': prediction_label,
        'confidence': round(confidence.item() * 100, 2),
        'time_taken': inference_time,
        'loaded_fresh': loaded_fresh
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)