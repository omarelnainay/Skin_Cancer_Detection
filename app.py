from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("skin_cancer_cnn.h5")  # Load your trained model

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
