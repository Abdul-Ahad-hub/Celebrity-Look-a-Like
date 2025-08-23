from flask import Flask, request, jsonify, render_template
import face_recognition
import os
from werkzeug.utils import secure_filename
import numpy as np
from flask import send_from_directory
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'celebs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        known_faces, known_names = pickle.load(f)
        
else:

    known_faces = []
    known_names = []

    for celeb_folder in os.listdir(DATASET_FOLDER):
        celeb_path = os.path.join(DATASET_FOLDER, celeb_folder)
        if os.path.isdir(celeb_path):
            
            for file in os.listdir(celeb_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(celeb_path, file)
                     
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_faces.append(encodings[0])
                        known_names.append(celeb_folder)
                      
                        break
                    
                except Exception as e:
                    print(f"Failed to process: {e}")


    with open("encodings.pkl", "wb") as f:
        pickle.dump((known_faces, known_names), f)
        

@app.route('/')
def home():
    return render_template('index.html')

 
@app.route('/match', methods=['POST'])
def match():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = face_recognition.load_image_file(file_path)
    unknown_encodings = face_recognition.face_encodings(image)

    if not unknown_encodings:
        return jsonify({'error': 'No face found'}), 400

    unknown_encoding = unknown_encodings[0]
    distances = face_recognition.face_distance(known_faces, unknown_encoding)

     
    top_indices = np.argsort(distances)[:3]
    matches = []
    for idx in top_indices:
        similarity = max(0, 1 - distances[idx]) * 100  
        celeb_name = known_names[idx]
    
        celeb_dir = os.path.join(DATASET_FOLDER, celeb_name)
         
        image_files = [f for f in os.listdir(celeb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            
            celeb_image = f"/celebs/{celeb_name}/{image_files[0]}"
       
        
# image path in static/
        matches.append({
            "name": celeb_name,
            "similarity": round(similarity, 2),
            "image": celeb_image
        })

    return jsonify({
        "match": True,
        "results": matches
    })

@app.route('/celebs/<celeb_name>/<filename>')
def serve_celeb_image(celeb_name, filename):
    return send_from_directory(os.path.join(DATASET_FOLDER, celeb_name), filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
