import datetime
import networkx as nx

import fpdf
# import reportlab
# from reportlab.pdfgen import canvas
from PIL import Image


class Report(fpdf.FPDF, fpdf.HTMLMixin):
    pass


class ReportGenerator:
    def __init__(self, model_objects, networkx_graphs, image_with_objects=None, mm_per_px=None):
        if image_with_objects is not None:
            self.image_filename = "img_{}.png".format(datetime.datetime.utcnow().timestamp())
            self.image = Image.fromarray(image_with_objects)
            self.image.save(self.image_filename)
        self.model_objects = model_objects
        self.networkx_graphs = networkx_graphs
        self.found_objects = self._get_found_objects()
        self.mm_per_px = mm_per_px

    @staticmethod
    def _flatten_list(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

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

    def _repr_found_objects(self, length_unit):
        if length_unit == "mm":
            length_ratio = self.mm_per_px
        elif length_unit == "px":
            length_ratio = 1
        else:
            length_ratio = 0
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

    def _get_found_object_types(self):
        return set(found_object["type"] for found_object in self.found_objects.values())

    def _get_found_objects(self):
        return dict(enumerate(self._flatten_list(
            self._flatten_list(nx.get_node_attributes(graph, "found_objects").values()) for graph in
            self.networkx_graphs), start=1))

    def _count_found_objects(self, object_type=None):
        if object_type:
            types = [found_object["type"] for found_object in self.found_objects.values()]
            return types.count(object_type)
        else:
            return sum([g.number_of_nodes() for g in self.networkx_graphs])

    @staticmethod
    def _print_found_object_number(pdf, number, center_x, center_y, width=1, height=1):
        x = center_x - width / 2
        y = center_y - height / 2
        # pdf.ellipse(x, y, width, height, "DF")
        pdf.set_xy(x, y)
        # pdf.set_font_size(height * 2.83464567)
        pdf.set_font_size(height * 2)
        pdf.cell(width, height, str(number), align="C", border=1, fill=True)

    def generate_pdf_report(self, image):
        pdf = Report()

        pdf.add_page()
        pdf.set_font("Arial")

        pdf.set_font_size(16)
        pdf.cell(w=0, h=10, ln=1, txt="Found Objects", align="L")

        pdf.set_font_size(12)
        found_object_types = self._get_found_object_types()
        count = {object_type: self._count_found_objects(object_type) for object_type in found_object_types}
        pdf.cell(w=0, h=10, ln=1, txt="Total number of found objects: {}".format(sum(count.values())), align="L")
        for k, v in count.items():
            pdf.cell(w=0, h=10, ln=1, txt="Number of found objects ({}): {}".format(k, v), align="L")

        width_px, height_px = self.image.size
        # r = 5
        # width_mm = width_px / r
        # height_mm = height_px / r
        width_mm = pdf.fw - pdf.l_margin - pdf.r_margin
        height_mm = height_px * width_mm / width_px
        start_x = pdf.get_x()
        start_y = pdf.get_y()
        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(255, 0, 0)
        pdf.set_line_width(0.2)
        pdf.image(image, w=width_mm, h=height_mm)
        # x = pdf.get_x()
        # y = pdf.get_y()
        for number, found_object in self.found_objects.items():
            center_coords = found_object["center_coords"]
            center_x = start_x + center_coords[1] / width_px * width_mm
            center_y = start_y + center_coords[0] / height_px * height_mm
            self._print_found_object_number(pdf, number, center_x, center_y, width_mm / 50, height_mm / 50)

        pdf.add_page()
        pdf.set_font_size(12)
        pdf.set_draw_color(0)
        # pdf.set_xy(x, y)
        pdf.cell(w=0, h=10, ln=1,
                 txt="1 mm per px = {}".format(self.mm_per_px if self.mm_per_px is not None else "(Not available)"),
                 align="L")
        pdf.write_html(self._repr_found_objects("px"))

        if self.mm_per_px is not None:
            pdf.add_page()
            pdf.write_html(self._repr_found_objects("mm"))

        return pdf

    # def generate_pdf_report2(self, image):
    #     pdf = fpdf.FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial")
    #     pdf.set_font_size(16)
    #     pdf.cell(w=0, h=10, ln=1, txt="Found Objects", align="L")
    #     pdf.image(image, w=100)
    #     print(pdf.get_x())
    #     print(pdf.get_y())
    #     pdf.ellipse(0, 0, 10, 10)
    #     return pdf

    # def generate_pdf_report3(self, image):
    #     c = canvas.Canvas("hello.pdf")
    #     c.drawString(100,750,"Welcome to Reportlab!")
    #     c.save()

    def to_pdf(self, pdf_filename=None, image_filename=None):
        if image_filename:
            image = image_filename
        else:
            image = self.image_filename

        pdf = self.generate_pdf_report(image)

        if pdf_filename:
            pdf.output(name=pdf_filename, dest="F")
        else:
            return pdf.output(dest="S").encode("latin-1")

    def to_excel(self):
        pass


if __name__ == "__main__":
    rg = ReportGenerator(None, None)
    rg.to_pdf("report.pdf", "result.png")
    # p = fpdf.FPDF()
    # p.add_page()
    # p.set_line_width(2)
    # p.set_fill_color(255, 0, 0)
    # p.ellipse(1, 1, 2, 2, "DF")
    # p.output("report_test.pdf")
