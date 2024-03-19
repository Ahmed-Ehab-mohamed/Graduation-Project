from flask import Flask, render_template, request
from autopsy_report_generator_backend import AutopsyReportGeneratorBackend
import os

app = Flask(__name__)
autopsy_backend = AutopsyReportGeneratorBackend()

@app.route('/')
def index():
    return render_template('autopsy_report_gen.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
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
