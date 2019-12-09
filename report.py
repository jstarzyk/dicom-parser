import datetime
from io import BytesIO

import networkx as nx
import cv2 as cv
import numpy as np

import fpdf
import xlsxwriter
from PIL import Image


class TextDrawer:
    def __init__(self, font_face, font_scale, font_thickness):
        self.font_face = font_face
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_texts(self, image, texts, center_coords):
        assert len(texts) == len(center_coords)
        rectangle_size = self._get_max_text_size(texts)
        for text, center_coord in zip(texts, center_coords):
            self._draw_text(image, text, center_coord, rectangle_size)

    def _get_max_text_size(self, texts):
        return max(max(cv.getTextSize(text, self.font_face, self.font_scale, self.font_thickness)[0]) for text in texts)

    def _draw_text(self, image, text, center_coord, rectangle_size, fill_color=(255, 255, 255), text_color=(0, 0, 0),
                   line_color=(255, 0, 0), line_thickness=0):
        coord1 = center_coord - rectangle_size / 2
        coord2 = center_coord + rectangle_size / 2

        text_size, _ = cv.getTextSize(text, self.font_face, self.font_scale, self.font_thickness)
        offset = (0, rectangle_size) + ((np.repeat(rectangle_size, 2) - text_size) / 2) * (1, -1)

        pt1 = self._parse_point(coord1)
        pt2 = self._parse_point(coord2)

        if line_thickness > 0:
            cv.rectangle(image, pt1, pt2, line_color, line_thickness)

        cv.rectangle(image, pt1, pt2, fill_color, -1)
        cv.putText(image, text, self._parse_point(coord1 + offset), self.font_face, self.font_scale, text_color,
                   self.font_thickness, cv.LINE_AA)

    @staticmethod
    def _parse_point(coord):
        return tuple(coord.astype("int"))


class ReportGenerator:
    class PDFReport(fpdf.FPDF, fpdf.HTMLMixin):
        pass

    def __init__(self, networkx_graphs, original_image_filepath, color_per_type_filepath, color_per_object_data,
                 mm_per_px=None):
        self.mm_per_px = mm_per_px
        self.networkx_graphs = networkx_graphs
        self.found_objects = self._get_found_objects()
        self.color_per_type_filepath = color_per_type_filepath
        self.original_image_filepath = original_image_filepath
        self.object_numbers_filename = "object_numbers.png"
        Image.fromarray(self._draw_object_numbers(color_per_object_data)).save(self.object_numbers_filename)
        self.image_height, self.image_width, _ = color_per_object_data.shape

    def _get_found_objects(self):
        return dict(enumerate(self._flatten_list(
            self._flatten_list(nx.get_node_attributes(graph, "found_objects").values()) for graph in
            self.networkx_graphs), start=1))

    @staticmethod
    def _flatten_list(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    def _draw_object_numbers(self, image_data):
        f = 4
        resized_data = cv.resize(image_data, None, fx=f, fy=f)
        text_drawer = TextDrawer(cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 1)
        text_drawer.draw_texts(
            resized_data,
            [str(number) for number in self.found_objects.keys()],
            [np.flip(found_object["center_coords"]) * f for found_object in self.found_objects.values()]
        )
        return resized_data

    def _generate_pdf_report(self, pdf):
        pdf.add_page()
        pdf.set_font("Arial")

        pdf.set_font_size(20)
        pdf.cell(w=0, h=15, ln=1, txt="Found Objects", align="L")

        self._write_object_count(pdf)

        width_mm = pdf.fw - pdf.l_margin - pdf.r_margin
        height_mm = self.image_height * (width_mm / self.image_width)

        pdf.set_font_size(8)

        pdf.add_page()
        pdf.image(self.original_image_filepath, w=width_mm, h=height_mm)
        pdf.cell(w=0, h=6, ln=1, txt="Original Image", align="C")

        pdf.add_page()
        pdf.image(self.color_per_type_filepath, w=width_mm, h=height_mm)
        pdf.cell(w=0, h=6, ln=1, txt="Objects by Type", align="C")

        pdf.add_page()
        pdf.image(self.object_numbers_filename, w=width_mm, h=height_mm)
        pdf.cell(w=0, h=6, ln=1, txt="Objects by Number", align="C")

        pdf.add_page()
        self._write_object_features(pdf, "px")
        pdf.cell(w=0, h=9, ln=1,
                 txt="1 mm per px = {}".format(self.mm_per_px if self.mm_per_px is not None else "(Not available)"),
                 align="L")

        if self.mm_per_px is None:
            return

        pdf.add_page()
        self._write_object_features(pdf, "mm")

    def _write_object_count(self, pdf):
        pdf.set_font_size(16)
        pdf.cell(w=0, h=12, ln=1, txt="Object Count", align="L")
        pdf.set_font_size(12)
        count = {object_type: self._count_found_objects(object_type) for object_type in self._get_found_object_types()}
        for k, v in count.items():
            pdf.cell(w=0, h=9, ln=1, txt="Objects of type '{}': {}".format(k, v), align="L")
        pdf.cell(w=0, h=9, ln=1, txt="Total objects found: {}".format(sum(count.values())), align="L")

    def _write_object_features(self, pdf, length_unit):
        x = pdf.get_x()
        pdf.set_font_size(16)
        pdf.cell(w=0, h=12, ln=1, txt="Object Features ({})".format(length_unit), align="L")
        pdf.set_font_size(12)
        pdf.set_draw_color(0)
        pdf.write_html(self._repr_object_features(length_unit))
        pdf.set_font("Arial")
        pdf.set_x(x)

    def _count_found_objects(self, object_type=None):
        if object_type:
            types = [found_object["type"] for found_object in self.found_objects.values()]
            return types.count(object_type)
        else:
            return sum([g.number_of_nodes() for g in self.networkx_graphs])

    def _get_found_object_types(self):
        return set(found_object["type"] for found_object in self.found_objects.values())

    def _repr_object_features(self, length_unit):
        length_ratio = self.mm_per_px if length_unit == "mm" else 1
        return """
        <table border="1" align="center" width="100%">
        <thead>
            <tr>
                <th width="5%"> </th>
                <th width="15%">Type</th>
                <th width="20%">Length [{}]</th>
                <th width="20%">Min width [{}]</th>
                <th width="20%">Max width [{}]</th>
                <th width="20%">Max angle [°]</th>
            </tr>
        </thead>
        <tbody>
            {}
        </tbody>
        </table>
        """.format(length_unit, length_unit, length_unit, "\n".join(
            self._repr_found_object(number, found_object, length_ratio) for number, found_object in
            self.found_objects.items()))

    @staticmethod
    def _repr_found_object(number, found_object, length_ratio):
        return """
        <tr bgcolor="{}">
        <td>{}</td>
        <td>{}</td>
        <td align="right">{}</td>
        <td align="right">{}</td>
        <td align="right">{}</td>
        <td align="right">{}</td>
        </tr>
        """.format(
            "#FFFFFF" if number % 2 == 0 else "#F0FFF0",
            number,
            found_object["type"],
            round(found_object["length"] * length_ratio, 1),
            round(found_object["min_width"] * length_ratio, 2),
            round(found_object["max_width"] * length_ratio, 2),
            round(found_object["max_angle"], 1),
        )

    def to_pdf(self, filename=None):
        pdf = ReportGenerator.PDFReport()
        self._generate_pdf_report(pdf)
        return pdf.output(dest="S").encode("latin-1") if filename is None else pdf.output(name=filename, dest="F")

    def _generate_xlsx_report(self, xlsx):
        formats = {
            "bold": xlsx.add_format({"bold": True}),
            "round1": xlsx.add_format({"num_format": "0.0"}),
            "round2": xlsx.add_format({"num_format": "0.00"})
        }

        start_row = 0
        start_col = 0

        object_count = xlsx.add_worksheet("Object Count")
        count = {object_type: self._count_found_objects(object_type) for object_type in self._get_found_object_types()}
        data_row = self._write_table(
            object_count, start_row, start_col, ("Type", "Count"), count.items(), formats["bold"]
        )
        number_range = xlsxwriter.utility.xl_range(start_row + 1, start_col + 1, data_row - 1, start_col + 1)
        object_count.write_string(data_row, start_col, "Total", formats["bold"])
        object_count.write_formula(data_row, start_col + 1, "=SUM({})".format(number_range), value=sum(count.values()))

        original_image = xlsx.add_worksheet("Original Image")
        original_image.insert_image(start_row, start_col, self.original_image_filepath)

        color_per_type = xlsx.add_worksheet("Objects by Type")
        color_per_type.insert_image(start_row, start_col, self.color_per_type_filepath)

        object_numbers = xlsx.add_worksheet("Objects by Number")
        object_numbers.insert_image(start_row, start_col, self.object_numbers_filename)

        object_features_columns = ("Number", "Type", "Length [{}]", "Min width [{}]", "Max width [{}]", "Max angle [°]")
        object_features_name = "Object Features ({})"
        object_features_column_formats = (
            None, None, formats["round1"], formats["round2"], formats["round2"], formats["round1"]
        )

        def _get_object_features(length_ratio):
            return ((number,
                     found_object["type"],
                     found_object["length"] * length_ratio,
                     found_object["min_width"] * length_ratio,
                     found_object["max_width"] * length_ratio,
                     found_object["max_angle"]) for number, found_object in self.found_objects.items())

        object_features_px = xlsx.add_worksheet(object_features_name.format("px"))
        self._write_table(object_features_px, start_row, start_col,
                          (column.format("px") for column in object_features_columns),
                          _get_object_features(1), formats["bold"], object_features_column_formats)

        object_size_ratio = xlsx.add_worksheet("Object Size Ratio")
        object_size_ratio.write_string(start_row, start_col, "1 mm per px", formats["bold"])
        object_size_ratio.write(
            start_row + 1, start_col, self.mm_per_px if self.mm_per_px is not None else "(Not available)"
        )

        if self.mm_per_px is None:
            return

        object_features_mm = xlsx.add_worksheet(object_features_name.format("mm"))
        self._write_table(object_features_mm, start_row, start_col,
                          (column.format("mm") for column in object_features_columns),
                          _get_object_features(self.mm_per_px), formats["bold"], object_features_column_formats)

    @staticmethod
    def _write_table(worksheet, start_row, start_col, column_names, table_values, column_names_format,
                     column_formats=None):
        for col, column_name in enumerate(column_names):
            worksheet.write_string(start_row, start_col + col, column_name, column_names_format)
        data_row = start_row + 1
        for row_values in table_values:
            for col, cell in enumerate(row_values):
                worksheet.write(
                    data_row, start_col + col, cell, None if column_formats is None else column_formats[col]
                )
            data_row += 1
        return data_row

    def to_xlsx(self, filename=None):
        output = BytesIO() if filename is None else filename
        xlsx = xlsxwriter.Workbook(output)
        self._generate_xlsx_report(xlsx)
        xlsx.close()
        if filename is None:
            output.seek(0)
            return output.read()
