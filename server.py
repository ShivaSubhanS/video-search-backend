import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from threading import Lock

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch_dtype
).to(device)

# Load sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Video settings
video_base_dir = "videos"
fps = 25.0
captions_with_timestamps = []
lock = Lock()

if not os.path.exists(video_base_dir):
    os.makedirs(video_base_dir)

def compute_embedding(caption):
    return sentence_model.encode(caption, convert_to_tensor=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    video_path = os.path.join(video_base_dir, file.filename)
    
    # Save video
    file.save(video_path)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 400
    
    global captions_with_timestamps
    captions_with_timestamps = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 15 == 0:
            current_timestamp = frame_count / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Generate caption with BLIP
            inputs = processor(images=pil_image, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Compute embedding
            embedding = compute_embedding(caption)
            
            # Generate thumbnail
            _, buffer = cv2.imencode('.jpg', frame)
            thumbnail = base64.b64encode(buffer).decode('utf-8')
            
            with lock:
                captions_with_timestamps.append({
                    'caption': caption,
                    'timestamp': current_timestamp,
                    'embedding': embedding.tolist(),
                    'thumbnail': thumbnail
                })
        
        frame_count += 1
    
    cap.release()
    return jsonify({'message': 'Video processed successfully'})

@app.route('/search', methods=['POST'])
def search_captions():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    query_embedding = compute_embedding(query).to(device)
    results = []
    
    with lock:
        for item in captions_with_timestamps:
            embedding = torch.tensor(item['embedding']).to(device)
            similarity = util.cos_sim(query_embedding, embedding).item()
            if similarity > 0.5:
                results.append({
                    'caption': item['caption'],
                    'timestamp': item['timestamp'],
                    'similarity': similarity,
                    'thumbnail': item['thumbnail']
                })
    
    return jsonify(sorted(results, key=lambda x: x['similarity'], reverse=True))

if __name__ == '__main__':
    app.run(debug=False, port=5000)