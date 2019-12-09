#!/usr/bin/python

import os

from PIL import Image
from flask import *
from werkzeug.utils import secure_filename

import detect_object as do
from print_objects import print_objects_on_graphs
from report import ReportGenerator

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + "/uploads"
FILES_FOLDER = os.path.abspath(os.path.dirname(__file__)) + "/files"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FILES_FOLDER"] = FILES_FOLDER


def process_uploaded_file(files, name, extension):
    if name not in files:
        return {"error": "No file was received"}
    file = request.files[name]
    if file.filename == '':
        return {"error": "Wrong filename"}
    if file and file.filename.lower().endswith(extension):
        ff = filepath_filename(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(ff[0])
        return {"filename": ff[1]}


def get_pil_image(array):
    return Image.fromarray(array)


def get_dicom_image(filepath):
    dataset = do.get_dicom_dataset(filepath)
    return do.get_image(dataset)


def filepath_filename(folder, filename):
    filepath = os.path.join(folder, filename)
    return filepath, filename


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/upload_files", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        image = process_uploaded_file(request.files, "image", ".dcm")
        dictionary = process_uploaded_file(request.files, "dictionary", ".json")
        try:
            dicom_image_ff = filepath_filename(app.config["UPLOAD_FOLDER"], image["filename"])
            png_image_ff = filepath_filename(app.config["FILES_FOLDER"], dicom_image_ff[1] + ".png")
            get_pil_image(get_dicom_image(dicom_image_ff[0])).save(png_image_ff[0])
            image["png_filename"] = png_image_ff[1]
        except KeyError:
            pass
        return jsonify({"image": image, "dictionary": dictionary})
    return jsonify({"error": "Unknown error"})


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_file(filepath_filename(app.config["UPLOAD_FOLDER"], filename)[0])


@app.route("/files/<filename>")
def get_processed_file(filename):
    return send_file(filepath_filename(app.config["FILES_FOLDER"], filename)[0])


@app.route("/show_image")
def show_image():
    image = request.args.get("image", type=str)
    dictionary = request.args.get("dictionary", type=str)
    return render_template("show_image.html", image=image, dictionary=dictionary)


@app.route("/process_files", methods=["GET", "POST"])
def process_files():
    upload_folder = app.config["UPLOAD_FOLDER"]
    files_folder = app.config["FILES_FOLDER"]

    try:
        image_ff = filepath_filename(upload_folder, request.json["image"])
        dictionary_ff = filepath_filename(upload_folder, request.json["dictionary"])

        dataset = do.get_dicom_dataset(image_ff[0])
        original_image = do.get_image(dataset)
        mm_per_px = do.get_mm_per_px_ratio(dataset)
        model_objects = do.load_objects(dictionary_ff[0])
        graphs_processed = do.process_image(original_image)
        graphs_of_objects = do.GraphOfFoundObjects.find_objects_in_graphs(graphs_processed, model_objects)

        color_per_type = print_objects_on_graphs(graphs_of_objects, original_image, fill=False,
                                                 method="color_per_type")

        color_per_object = print_objects_on_graphs(graphs_of_objects, original_image, fill=False,
                                                   method="color_per_object")

        networkx_graphs = do.GraphOfFoundObjects.parse_networkx_graphs(graphs_of_objects)
        networkx_json_graph_list = do.GraphOfFoundObjects.to_networkx_json_graph_list(networkx_graphs)

        rg = ReportGenerator(networkx_graphs, color_per_type, color_per_object, original_image, image_ff[1],
                             dictionary_ff[1], mm_per_px)

        color_per_type_ff = filepath_filename(files_folder, "color_per_type.png")
        get_pil_image(color_per_type).save(color_per_type_ff[0])

        color_per_object_ff = filepath_filename(files_folder, "color_per_object.png")
        get_pil_image(color_per_object).save(color_per_object_ff[0])

        networkx_json_graph_list_ff = filepath_filename(files_folder, "graphs.json")
        do.GraphOfFoundObjects.serialize(networkx_json_graph_list, networkx_json_graph_list_ff[0])

        pdf_report_ff = filepath_filename(files_folder, "report.pdf")
        rg.to_pdf(pdf_report_ff[0])

        xlsx_report_ff = filepath_filename(files_folder, "report.xlsx")
        rg.to_xlsx(xlsx_report_ff[0])

        return jsonify({
            "color_per_type": color_per_type_ff[1],
            "color_per_object": color_per_object_ff[1],
            "networkx_json_graph_list": networkx_json_graph_list_ff[1],
            "pdf_report": pdf_report_ff[1],
            "xlsx_report": xlsx_report_ff[1]
        })
    except IOError:
        return jsonify({"error": "Unknown error"})


if __name__ == "__main__":
    app.run(debug=True)
