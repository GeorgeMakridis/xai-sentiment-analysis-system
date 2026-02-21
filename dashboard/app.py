from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import os
import requests
import json
import uuid
from werkzeug.utils import secure_filename
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
CORS(app)

# Configuration
XAI_SERVICE_URL = os.environ.get('XAI_SERVICE_URL', 'http://xai_service:8000')
AI_OUTPUTS_SERVICE_URL = os.environ.get('AI_OUTPUTS_SERVICE_URL', 'http://ai_outputs:8001')
SHARED_DATA_DIR = '/app/shared_data'
UPLOAD_FOLDER = os.path.join(SHARED_DATA_DIR, 'uploads')
MODELS_FOLDER = os.path.join(SHARED_DATA_DIR, 'models')
RESULTS_FOLDER = os.path.join(SHARED_DATA_DIR, 'results')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Mock user database (in production, use a real database)
USERS = {
    'admin': 'password123',
    'user1': 'password123'
}

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session.get('user_id', 'admin')
    return render_template('index.html', username=username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['user_id'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    data_type = request.form.get('data_type', 'timeseries')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Trigger initial ingestion process
    try:
        response = requests.post(f"{XAI_SERVICE_URL}/ingest", json={
            'file_path': file_path,
            'user_id': session['user_id'],
            'data_type': data_type
        })
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'message': 'Data uploaded and ingested successfully', 
                'file_path': file_path,
                'data_summary': result.get('data_summary', {})
            })
        else:
            return jsonify({'error': 'Ingestion failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Ingestion failed: {str(e)}'}), 500

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(MODELS_FOLDER, filename)
    file.save(file_path)
    
    # Trigger XAI analysis
    try:
        response = requests.post(f"{XAI_SERVICE_URL}/analyze", json={
            'model_path': file_path,
            'user_id': session['user_id']
        })
        
        if response.status_code == 200:
            result_data = response.json()
            
            # Store results in AI outputs service for chat functionality
            try:
                store_response = requests.post(f"{AI_OUTPUTS_SERVICE_URL}/store-results", json={
                    'user_id': session['user_id'],
                    'results': result_data
                })
                if store_response.status_code != 200:
                    print(f"Warning: Failed to store results in AI outputs service: {store_response.text}")
                else:
                    print(f"Successfully stored results in AI outputs service for user {session['user_id']}")
            except Exception as e:
                print(f"Warning: Could not store results in AI outputs service: {e}")
            
            return jsonify({
                'message': 'Model uploaded and analysis completed',
                'results': result_data
            })
        else:
            return jsonify({'error': 'Analysis failed'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/get-results')
def get_results():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Try to get results from AI outputs service first
        try:
            response = requests.get(f"{AI_OUTPUTS_SERVICE_URL}/results/{session['user_id']}")
            if response.status_code == 200:
                ai_results = response.json()
                # Check if AI outputs service returned actual results with images
                if 'images' in ai_results and ai_results['images']:
                    return jsonify(ai_results)
                # If no images, fall back to XAI service results
                print(f"AI outputs service returned metadata only, falling back to XAI results")
        except Exception as e:
            print(f"AI outputs service not available: {e}")
        
        # If AI outputs service fails, try to get results from XAI service
        try:
            # Check if there are any result files in the shared volume
            results_dir = os.path.join(SHARED_DATA_DIR, 'results')
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
                
                # Find the most recent result file for this user
                user_results = [f for f in result_files if session['user_id'] in f]
                if user_results:
                    # Get the most recent file
                    latest_result = sorted(user_results)[-1]
                    result_file = os.path.join(results_dir, latest_result)
                    
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    return jsonify(results)
                else:
                    return jsonify({'message': 'No results found. Please upload and analyze a model first.'})
            else:
                return jsonify({'message': 'No results found. Please upload and analyze a model first.'})
                
        except Exception as e:
            print(f"Error reading results from file: {e}")
            return jsonify({'message': 'No results found. Please upload and analyze a model first.'})
            
    except Exception as e:
        return jsonify({'error': f'Failed to fetch results: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response = requests.post(f"{AI_OUTPUTS_SERVICE_URL}/chat", json={
            'question': question,
            'user_id': session['user_id']
        })
        
        if response.status_code == 200:
            return jsonify(response.json())
        elif response.status_code == 404:
            # Handle case where no results are found for the user
            error_data = response.json()
            return jsonify({'error': error_data.get('error', 'No results found. Please upload and analyze a model first.')}), 404
        else:
            return jsonify({'error': 'Chat failed'}), 500
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/api/create-sp100-model', methods=['POST'])
def create_sp100_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    model_type = data.get('model_type')
    
    if not model_type:
        return jsonify({'error': 'No model type provided'}), 400
    
    # Path to SP100 data
    sp100_data_path = os.path.join(SHARED_DATA_DIR, 'uploads', 'sp100_daily_prices.csv')
    
    if not os.path.exists(sp100_data_path):
        return jsonify({'error': 'SP100 data not found'}), 404
    
    try:
        # Trigger model creation with SP100 data
        response = requests.post(f"{XAI_SERVICE_URL}/create-model", json={
            'data_path': sp100_data_path,
            'model_type': model_type,
            'user_id': session['user_id']
        })
        
        if response.status_code == 200:
            result_data = response.json()
            return jsonify({
                'message': f'SP100 {model_type.replace("_", " ").title()} model created successfully',
                'results': result_data
            })
        else:
            return jsonify({'error': 'Model creation failed'}), 500
    except Exception as e:
        return jsonify({'error': f'Model creation failed: {str(e)}'}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    model_type = data.get('model_type')
    data_type = data.get('data_type')
    
    if not model_type or not data_type:
        return jsonify({'error': 'Missing model_type or data_type'}), 400
    
    try:
        # Trigger model training/analysis
        response = requests.post(f"{XAI_SERVICE_URL}/train-model", json={
            'model_type': model_type,
            'data_type': data_type,
            'user_id': session['user_id']
        })
        
        if response.status_code == 200:
            result_data = response.json()
            return jsonify({
                'message': f'{model_type.replace("_", " ").title()} model trained successfully',
                'results': result_data
            })
        else:
            return jsonify({'error': 'Model training failed'}), 500
    except Exception as e:
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/data-statistics', methods=['POST'])
def data_statistics():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    user_id = session['user_id']
    
    try:
        response = requests.post(f"{XAI_SERVICE_URL}/data-statistics", json={
            'user_id': user_id
        })
        
        if response.status_code == 200:
            result = response.json()
            
            # Store data statistics results in AI outputs service for chat functionality
            try:
                store_response = requests.post(f"{AI_OUTPUTS_SERVICE_URL}/store-results", json={
                    'user_id': user_id,
                    'results': {
                        'type': 'data_statistics',
                        'images': result.get('images', []),
                        'data_type': result.get('data_type', 'unknown'),
                        'insights': result.get('insights', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                })
                if store_response.status_code != 200:
                    print(f"Warning: Failed to store data statistics in AI outputs service: {store_response.text}")
                else:
                    print(f"Successfully stored data statistics in AI outputs service for user {user_id}")
            except Exception as e:
                print(f"Warning: Could not store data statistics in AI outputs service: {e}")
            
            return jsonify({
                'message': 'Data statistics generated successfully',
                'images': result.get('images', []),
                'data_type': result.get('data_type', 'unknown'),
                'user_id': user_id
            })
        else:
            return jsonify({'error': 'Failed to generate data statistics'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to generate data statistics: {str(e)}'}), 500

@app.route('/api/preprocess-data', methods=['POST'])
def preprocess_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    preprocessing_options = {
        'target_column': data.get('target_column'),
        'selected_features': data.get('selected_features', []),
        'remove_duplicates': data.get('remove_duplicates', True),
        'handle_missing_values': data.get('handle_missing_values', True),
        'normalize_data': data.get('normalize_data', True),
        'encode_categorical': data.get('encode_categorical', True),
        'text_preprocessing': data.get('text_preprocessing', {}),
        'user_id': session['user_id']
    }
    
    try:
        # Send preprocessing options to XAI service
        response = requests.post(f"{XAI_SERVICE_URL}/preprocess-data", json=preprocessing_options)
        
        if response.status_code == 200:
            result_data = response.json()
            return jsonify({
                'message': 'Data preprocessed successfully',
                'preprocessed_data': result_data
            })
        else:
            return jsonify({'error': 'Data preprocessing failed'}), 500
    except Exception as e:
        return jsonify({'error': f'Data preprocessing failed: {str(e)}'}), 500

@app.route('/api/enhanced-xai', methods=['POST'])
def enhanced_xai():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        model_path = data.get('model_path')
        data_path = data.get('data_path')
        user_id = data.get('user_id', session['user_id'])
        
        if not model_path or not data_path:
            return jsonify({'error': 'Missing model_path or data_path'}), 400
        
        response = requests.post(f"{XAI_SERVICE_URL}/enhanced-xai", json={
            'model_path': model_path,
            'data_path': data_path,
            'user_id': user_id
        })
        
        if response.status_code == 200:
            result = response.json()
            return jsonify(result)
        else:
            return jsonify({'error': 'Enhanced XAI analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Enhanced XAI analysis failed: {str(e)}'}), 500

@app.route('/api/download-finbert', methods=['POST'])
def download_finbert():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        response = requests.post(f"{XAI_SERVICE_URL}/download-finbert", json={
            'user_id': session['user_id']
        })
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'message': 'FinBERT model downloaded successfully',
                'model_info': result.get('model_info', {})
            })
        else:
            return jsonify({'error': 'FinBERT download failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'FinBERT download failed: {str(e)}'}), 500

@app.route('/api/download-mnist', methods=['POST'])
def download_mnist():
    """Download MNIST dataset for testing image data functionality"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        sample_size = data.get('sample_size', 1000)
        
        response = requests.post(
            f"{XAI_SERVICE_URL}/download-mnist", 
            json={
                'user_id': session['user_id'],
                'sample_size': sample_size
            },
            timeout=300  # 5 minutes timeout for download
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'message': result.get('message', 'MNIST dataset downloaded successfully'),
                'file_path': result.get('file_path'),
                'sample_file': result.get('sample_file'),
                'metadata_file': result.get('metadata_file'),
                'status': result.get('status'),
                'sample_size': result.get('sample_size')
            })
        else:
            error_data = response.json() if response.content else {}
            return jsonify({'error': error_data.get('error', 'MNIST download failed')}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'MNIST download timed out. Please try again with a smaller sample size.'}), 504
    except Exception as e:
        return jsonify({'error': f'MNIST download failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 