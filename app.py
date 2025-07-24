import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  
model = load_model('skin_cancer_cnn.h5')

TARGET_SIZE = (224, 224)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(img_path)

            img = preprocess_image(img_path)
            result = model.predict(img)[0][0] 
            prediction = "Cancer" if result >= 0.5 else "Non-Cancer"
            image_url = img_path

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    port = 5000
    print(f"Running on port {port}")
    app.run(debug=True, port=port)
=======
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("skin_cancer_cnn.h5") 
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Malignant" if prediction > 0.5 else "Benign"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            result = predict_image(file_path)
            return render_template("index.html", image=file.filename, prediction=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
