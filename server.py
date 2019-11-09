#!/usr/bin/python

import os
# from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, send_file
from flask import *
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import detect_object as do
from base64 import b64encode

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def send_image(image):
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/upload_image", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file was received"})
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Wrong file name"})
        if file and file.filename.lower().endswith('.dcm'):
            filename = secure_filename(file.filename)
            print(file)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({"filename": filename})
    return jsonify({"error": "Unknown error"})


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.fromarray(do.get_image_from_dicom(file_path))
    return send_image(image)


@app.route('/show_image/<filename>')
def show_image(filename):
    return render_template("show_image.html", uploaded_image=filename)


@app.route('/show_result/<filename>')
def show_result(filename):
    return render_template("show_result.html", uploaded_image=filename)


@app.route("/process_image", methods=['GET', 'POST'])
def process_image():
    filename = request.json['filename']
    try:
        image = do.process_image(filename)
        # image = do.get_image_from_dicom(filename)
        img_io = BytesIO()
        Image.fromarray(image).save(img_io, 'PNG')
        img_io.seek(0)
        return b64encode(img_io.read())
    except IOError:
        return jsonify({"error": "Unknown error"})


if __name__ == "__main__":
    app.run(debug=True)
