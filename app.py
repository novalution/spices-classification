from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


app = Flask(__name__)

dic = {0: 'jahe',
       1: 'kencur',
       2: 'kunyit',
       3: 'lengkuas',
       4: 'temulawak'}

model = load_model(
    'C:/Users/ASUS/Documents/tugas/SKRIPSI NEW/model/model.h5')
model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224, 224, 3)
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route("/index", methods=['GET', 'POST'])
def utama():
    return render_template("index.html")


@app.route("/klasifikasi")
def klasifikasi():
    return render_template("klasifikasi.html")


@app.route("/resep1")
def resep1():
    return render_template("resep-1.html")


@app.route("/resep2")
def resep2():
    return render_template("resep-2.html")


@app.route("/resep3")
def resep3():
    return render_template("resep-3.html")


@app.route("/resep4")
def resep4():
    return render_template("resep-4.html")


@app.route("/resep5")
def resep5():
    return render_template("resep-5.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "C:/Users/ASUS/Documents/tugas/SKRIPSI FIX BANGET/static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

        return render_template("klasifikasi.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
