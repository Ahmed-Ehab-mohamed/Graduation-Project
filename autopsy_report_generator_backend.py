from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas

app = Flask(__name__)

class AutopsyReportGeneratorBackend:
    def __init__(self):
        self.styles = {
            'Title': ParagraphStyle(name='Title', fontName='Helvetica', fontSize=24, leading=28),
            'Heading': ParagraphStyle(name='Heading', fontName='Helvetica', fontSize=18, leading=22),
            'Normal': ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=12, leading=14)
        }

    def generate_autopsy_report(self, report_path, case_info):
        c = canvas.Canvas(report_path, pagesize=letter)
        c.setFont('Helvetica', 24)
        c.drawString(100, 750, "Autopsy Report")
        y = 650
        for key, value in case_info.items():
            c.setFont('Helvetica', 12)
            c.drawString(100, y, f"{key}: {value}")
            y -= 20
        c.save()

autopsy_backend = AutopsyReportGeneratorBackend()

class SketchGenerator:
    def __init__(self):
        self.generator = load_model('face_sketch_model.h5')

    def preprocess_sketch(self, sketch):
        try:
            if isinstance(sketch, str):
                sketch = Image.open(sketch)
            else:
                sketch = Image.open(sketch.stream)
        except Exception as e:
            raise ValueError("Failed to open the image file: " + str(e))

        sketch = sketch.resize((100, 100))
        sketch = np.array(sketch)
        sketch = (sketch - 127.5) / 127.5
        return sketch

    def generate_image(self, preprocessed_sketch):
        noise = np.random.normal(size=(1, 100))
        generated_photo = self.generator.predict(noise)[0]
        generated_photo = (generated_photo * 0.5 + 0.5) * 255
        generated_photo = generated_photo.astype(np.uint8)
        generated_photo = Image.fromarray(generated_photo)
        return generated_photo

sketch_generator = SketchGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        preprocessed_sketch = sketch_generator.preprocess_sketch(file)
        generated_image = sketch_generator.generate_image(preprocessed_sketch)
        # Save generated image
        generated_image_path = 'static/generated_image.jpg'
        generated_image.save(generated_image_path)
        return jsonify({'result': 'success', 'generated_image_path': generated_image_path})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/autopsy')
def autopsy():
    return render_template('autopsy_report_gen.html')


@app.route('/generate_autopsy_report', methods=['POST'])
def generate_autopsy_report():
    if request.method == 'POST':
        case_info = {
            'Case Number': request.form['case_number'],
            'Date of Death': request.form['date_of_death'],
            'Date of Incident': request.form['date_of_incident'],
            'Age': request.form['age'],
            'Gender': request.form['gender'],
            'Race': request.form['race'],
            'Latina': request.form['latina'],
            'Manner of Death': request.form['manner_of_death'],
            'Primary Cause': request.form['primary_cause'],
            'Secondary Cause': request.form['secondary_cause'],
            'Incident Address': request.form['incident_address']
        }
        report_folder = 'autopsy_reports'
        if not os.path.exists(report_folder):
            os.makedirs(report_folder)
        report_path = os.path.join(report_folder, f"autopsy_report_{case_info['Case Number']}.pdf")
        autopsy_backend.generate_autopsy_report(report_path, case_info)
        return f'Report generated successfully! <a href="{report_path}">Download Report</a>'

if __name__ == '__main__':
    app.run(debug=True,port=5001)
