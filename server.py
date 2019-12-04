#!/usr/bin/python

import os
# from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory, send_file
from flask import *
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import detect_object as do
from base64 import b64encode

from print_objects import print_objects_on_graphs

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def serialize_image(image):
    img_io = BytesIO()
    Image.fromarray(image).save(img_io, 'PNG')
    img_io.seek(0)
    return b64encode(img_io.read()).decode('utf-8')


def send_image(image):
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename


def process_file(files, name, extension):
    if name not in files:
        return {"error": "No file was received"}
    file = request.files[name]
    if file.filename == '':
        return {"error": "Wrong filename"}
    if file and file.filename.lower().endswith(extension):
        filename = save_file(file)
        return {"filename": filename}


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/upload_files", methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        image = process_file(request.files, 'image', '.dcm')
        dictionary = process_file(request.files, 'dictionary', '.json')
        return jsonify({"image": image, "dictionary": dictionary})
    return jsonify({"error": "Unknown error"})


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.fromarray(do.get_image_from_dicom(file_path))
    return send_image(image)


@app.route('/show_image')
def show_image():
    image = request.args.get("image", type=str)
    dictionary = request.args.get("dictionary", type=str)
    return render_template("show_image.html", uploaded_image=image, uploaded_dictionary=dictionary)


# @app.route('/show_result/<filename>')
# def show_result(filename):
#     return render_template("show_result.html", uploaded_image=filename)


@app.route("/process_files", methods=['GET', 'POST'])
def process_files():
    image = request.json["image"]
    dictionary = request.json["dictionary"]
    try:
        original_image = do.get_image_from_dicom(image)
        model_objects = do.load_objects(dictionary)
        graphs_processed = do.process_image(original_image)
        graphs_of_objects = do.GraphOfFoundObjects.find_objects_in_graphs(graphs_processed, model_objects)

        color_per_object = print_objects_on_graphs(
            graphs_of_objects,
            original_image,
            fill=False,
            method='color_per_object'
        )
        color_per_type = print_objects_on_graphs(
            graphs_of_objects,
            original_image,
            fill=False,
            method='color_per_type'
        )

        networkx_json_graph_list = do.GraphOfFoundObjects.to_networkx_json_graph_list(graphs_of_objects)

        return jsonify({
            "color_per_object": serialize_image(color_per_object),
            "color_per_type": serialize_image(color_per_type),
            "networkx_json_graph_list": do.GraphOfFoundObjects.serialize(networkx_json_graph_list)
        })
    except IOError:
        return jsonify({"error": "Unknown error"})


if __name__ == "__main__":
    app.run(debug=True)
