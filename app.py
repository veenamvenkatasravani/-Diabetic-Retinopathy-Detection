import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for

model = load_model("model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "User_Images"

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

classes = [
    'No DR',
    'Mild',
    'Moderate',
    'Severe',
    'Proliferative DR'
]

# HOME
@app.route('/')
def index():
    return render_template('index.html')

# LOGIN PAGE
@app.route('/login', methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form['email']
        password = request.form['password']

        if email == "" or password == "":
            return "Please enter Email and Password"

        else:
            return redirect(url_for('prediction'))

    return render_template('login.html')


# REGISTER PAGE
@app.route('/register', methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if name == "" or email == "" or password == "":
            return "Please fill all fields"

        else:
            return redirect(url_for('login'))

    return render_template('register.html')


# LOGOUT PAGE
@app.route('/logout')
def logout():
    return render_template('logout.html')


# PREDICTION PAGE
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


# PREDICT LOGIC
@app.route('/predict', methods=["POST"])
def predict():

    f = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)

    img = tf.keras.utils.load_img(filepath, target_size=(299,299))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0

    prediction = model.predict(x)
    prediction_index = np.argmax(prediction, axis=1)[0]

    result = classes[prediction_index]

    return render_template('prediction.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=False)
