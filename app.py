from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import zipfile
import shutil
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}
MODEL_PATH = 'model.h5'  # Place your trained model here

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L').resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def load_images_from_folder(folder):
    import glob
    X, y = [], []
    for digit in range(10):
        digit_folder = os.path.join(folder, str(digit))
        for img_path in glob.glob(os.path.join(digit_folder, '*')):
            img = Image.open(img_path).convert('L').resize((28, 28))
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(digit)
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = to_categorical(y, 10)
    return X, y

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            pred = model.predict(img_array)
            digit = int(np.argmax(pred))
            return render_template('result.html', filename=filename, digit=digit)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            return redirect(request.url)
        if file:
            zip_path = os.path.join('static', 'uploads', secure_filename(file.filename))
            file.save(zip_path)
            extract_path = os.path.join('static', 'uploads', 'train_data')
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            X, y = load_images_from_folder(extract_path)
            model = Sequential([
                Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
                MaxPooling2D((2,2)),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=5)
            model.save('model.h5')
            os.remove(zip_path)
            shutil.rmtree(extract_path)
            return 'Model trained and saved as model.h5! <a href="/">Back to Classifier</a>'
    return render_template('train.html')

if __name__ == '__main__':
    app.run(debug=True)


