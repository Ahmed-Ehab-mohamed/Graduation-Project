from flask import Flask, render_template, request, redirect, url_for, session
import csv
import random
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
from keras.models import load_model
import joblib
import cv2
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_folder='static')

app = Flask(__name__)


def generate_security_code(title):
    if title == "Admin":
        return "015" + str(random.randint(1000, 9999))
    elif title == "Forensic Scientist":
        return "012" + str(random.randint(1000, 9999))
    elif title == "Lawyer":
        return "010" + str(random.randint(1000, 9999))
    elif title == "Investigator":
        return "010" + str(random.randint(1000, 9999))


@app.route('/')
def index():
    return render_template('homepage.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/homepage')
def redirect_homepage():
    return render_template('home.html')


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        title = request.form['title']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        security_code = generate_security_code(title)
        approved = False

        # Write data to CSV file
        with open('users.csv', 'a', newline='') as csvfile:
            fieldnames = ['Name', 'Title', 'Email', 'Username', 'Password', 'Approved', 'Security Code']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write user data to CSV
            writer.writerow({'Name': name, 'Title': title, 'Email': email, 'Username': username, 'Password': password,
                             'Approved': approved, 'Security Code': security_code})

        return redirect(url_for('success'))


@app.route('/success')
def success():
    return 'Account created successfully!'


app.secret_key = 'your_secret_key_here'


def check_credentials(name, security_id, title, email, password):
    with open('users.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row['Full Name'] == name and
                    row['Security Code'] == security_id and
                    row['Type'] == title and
                    row['Email'] == email and
                    row['Password'] == password and
                    row['Approved'] == 'TRUE'):
                return True
    return False


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        security_id = request.form['security_id']
        title = request.form['title']
        email = request.form['email']
        password = request.form['password']

        if check_credentials(name, security_id, title, email, password):
            session['name'] = name
            session['title'] = title
            return redirect(url_for(f'{title.lower().replace(" ", "_")}_homepage'))
        else:
            return render_template('login.html', error="Invalid credentials or account not approved.")
    else:
        return render_template('login.html', error="")


class GlassTypePredictor:
    def __init__(self):
        self.glass_df = pd.read_csv("glass (1).csv")
        self.features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        self.target = 'Type'
        self.slider_ranges = {'RI': (1.50, 1.54, 0.0001),
                              'Na': (10.7, 17.4, 0.1),
                              'Mg': (0, 4.5, 0.1),
                              'Al': (0, 3.5, 0.1),
                              'Si': (69, 76, 1),
                              'K': (0, 6, 0.1),
                              'Ca': (5, 17, 0.1),
                              'Ba': (0, 4, 0.1),
                              'Fe': (0, 1.5, 0.1)}
        self.scaler = StandardScaler()
        self.X = self.glass_df[self.features]
        self.y = self.glass_df[self.target]
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.svm_mod = SVC(probability=True)
        self.svm_mod.fit(self.X_scaled, self.y)

    def predict_glass_type(self, feature_values):
        feature_values_scaled = self.scaler.transform([feature_values])
        prediction = self.svm_mod.predict(feature_values_scaled)
        probability = np.max(self.svm_mod.predict_proba(feature_values_scaled)) * 100
        return prediction[0], probability


@app.route('/glass')
def glass_page():
    glass_predictor = GlassTypePredictor()  # Instantiate the predictor
    return render_template('glass.html', slider_ranges=glass_predictor.slider_ranges)


@app.route('/predict_glass_type', methods=['POST'])
def predict_glass_type():
    data = request.form
    features = [float(data[feature]) for feature in GlassTypePredictor().features]
    glass_predictor = GlassTypePredictor()
    prediction, probability = glass_predictor.predict_glass_type(features)

    # Append prediction details to the file
    case_id = request.form.get('case_id')
    if case_id:
        with open(f"cases/{case_id}_crimescene.txt", 'a') as f:
            f.write("Glass found with the following values:\n")
            f.write(f"RI: {data['RI']}\n")
            f.write(f"Na: {data['Na']}\n")
            f.write(f"Mg: {data['Mg']}\n")
            f.write(f"Al: {data['Al']}\n")
            f.write(f"Si: {data['Si']}\n")
            f.write(f"K: {data['K']}\n")
            f.write(f"Ca: {data['Ca']}\n")
            f.write(f"Ba: {data['Ba']}\n")
            f.write(f"Fe: {data['Fe']}\n")
            f.write(f"Predicted Type: {prediction}\n")
            f.write(f"Prediction Probability: {round(probability, 2)}%\n\n")

    response = {
        'predicted_glass_type': int(prediction),
        'prediction_probability': round(probability, 2)
    }
    return jsonify(response)


@app.route('/create_glass_case', methods=['POST'])
def create_glass_case():
    case_id = request.form.get('new_case_id')
    if case_id:
        try:
            # Create the file with the provided case ID
            with open(f"cases/{case_id}_crimescene.txt", 'w') as f:
                # Add some initial content if needed
                f.write(f"CaseID: {case_id}\n")
                f.write(f"Crime Scene Description:\n")
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'No case ID provided.'})


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


# Autopsy Report Predictor
class AutopsyReportPredictor:
    def __init__(self):
        # Load the dataset
        data = pd.read_csv("autopsy_report_new.csv")
        data['Text'] = data['Secondary Cause'] + ' ' + data['Manner of Death'] + ' ' + data['Primary Cause']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Primary Cause'], test_size=0.2,
                                                            random_state=42)

        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer()

        # Fit and transform on training data
        self.X_train_tfidf = self.tfidf.fit_transform(X_train)

        # Train SVM model
        self.svm_model = SVC(kernel='linear')
        self.svm_model.fit(self.X_train_tfidf, y_train)

        # Predictions on the testing set
        X_test_tfidf = self.tfidf.transform(X_test)
        self.svm_predictions = self.svm_model.predict(X_test_tfidf)

        # Calculate accuracy
        self.accuracy = accuracy_score(y_test, self.svm_predictions)

    def predict_primary_cause(self, secondary_cause, manner_of_death):
        input_text = f"{secondary_cause} {manner_of_death}"
        input_vector = self.tfidf.transform([input_text])
        predicted_cause = self.svm_model.predict(input_vector)[0]
        return predicted_cause, self.accuracy


autopsy_predictor = AutopsyReportPredictor()


def get_case_files():
    case_files = []
    for file in os.listdir('cases'):
        if file.endswith("_crimescene.txt"):
            case_files.append(file)
    return case_files


@app.route('/generate_report', methods=['POST'])
def generate_report():
    if request.method == 'POST':
        case_number = request.form['case_number']
        selected_case = request.form['selected_case']
        case_info = {
            'Case Number': case_number,
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
        report_path = os.path.join(report_folder, f"autopsy_report_{case_number}.pdf")
        autopsy_backend.generate_autopsy_report(report_path, case_info)

        # Write the autopsy report path to the selected case file
        with open(f'cases/{selected_case}', 'a') as f:
            f.write(
                f"Autopsy Report for the body found in the crime scene associated with Report Number: {case_number}\n")
            f.write(f"Autopsy Report Path: {report_path}\n\n")

        return redirect(url_for('autopsy_report', case_number=case_number))


@app.route('/download_report/<case_number>')
def download_report(case_number):
    report_folder = 'autopsy_reports'
    report_path = os.path.join(report_folder, f"autopsy_report_{case_number}.pdf")
    return send_from_directory(directory=report_folder, filename=f"autopsy_report_{case_number}.pdf",
                               as_attachment=True)


@app.route('/predict_primary_cause', methods=['POST'])
def predict_primary_cause():
    secondary_cause = request.form['secondary_cause']
    manner_of_death = request.form['manner_of_death']
    predicted_cause, accuracy = autopsy_predictor.predict_primary_cause(secondary_cause, manner_of_death)
    return jsonify({'predicted_primary_cause': predicted_cause, 'accuracy': accuracy})


@app.route('/autopsy_create_case', methods=['POST'])
def create_case():
    if request.method == 'POST':
        case_id = request.form['case_id']
        if case_id:
            file_path = f"cases/{case_id}_crimescene.txt"
            with open(file_path, 'w') as file:
                file.write(f"Case ID: {case_id}\n")
            return f"Case created successfully! Case ID: {case_id}"
        else:
            return "Please provide a Case ID."


@app.route('/autopsy_report/<case_number>')
def autopsy_report(case_number):
    return render_template('autopsy_report.html', case_number=case_number)


# ___ Face Sketch ___#
class SketchGenerator:
    def __init__(self):
        self.generator = load_model('face_sketch_model.h5')
        self.cases_folder = 'cases'

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

    def create_case_files(self, case_id):
        if not os.path.exists(self.cases_folder):
            os.makedirs(self.cases_folder)
        case_file = os.path.join(self.cases_folder, f'{case_id}.txt')
        killer_file = os.path.join(self.cases_folder, f'{case_id}_killers.txt')

        with open(case_file, 'w'):
            pass

        with open(killer_file, 'w'):
            pass

    def update_rank(self, case_id, prisoner_name):
        case_file = os.path.join(self.cases_folder, f'{case_id}.txt')
        killer_file = os.path.join(self.cases_folder, f'{case_id}_killers.txt')
        rank = 500
        if os.path.exists(case_file):
            with open(case_file, 'r+') as f:
                lines = f.readlines()
                found = False
                for i, line in enumerate(lines):
                    if prisoner_name in line:
                        rank = int(line.split(':')[1].strip()) + 500
                        lines[i] = f'{prisoner_name}: {rank}\n'
                        found = True
                        break
                if not found:
                    lines.append(f'{prisoner_name}: {rank}\n')
                lines.sort(reverse=True, key=lambda x: int(x.split(':')[1]))
                f.seek(0)
                f.truncate()
                f.writelines(lines)
            # Append to killer file
            with open(killer_file, 'a') as f:
                f.write(f'Face Sketch Found for {prisoner_name}\n')

        else:
            with open(case_file, 'w') as f:
                f.write(f'{prisoner_name}: {rank}\n')
            # Create and append to killer file
            with open(killer_file, 'w') as f:
                f.write(f'Face Sketch Found for {prisoner_name}\n')

        return rank

    def get_case_list(self):
        case_files = [f for f in os.listdir(self.cases_folder) if f.endswith('.txt')]
        case_list = [f.split('.')[0] for f in case_files]
        return case_list

sketch_generator = SketchGenerator()

@app.route('/generate_face_sketch', methods=['GET', 'POST'])
def generate_face_sketch():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            preprocessed_sketch = sketch_generator.preprocess_sketch(file)
            generated_image = sketch_generator.generate_image(preprocessed_sketch)
            prisoner_name = request.form['prisoner_name']
            case_id = request.form['case_id']
            generated_image_path = f'static/{prisoner_name}_generated_image.jpg'
            generated_image.save(generated_image_path)
            rank = sketch_generator.update_rank(case_id, prisoner_name)
            case_list = sketch_generator.get_case_list()
            return render_template('facesketch.html', case_list=case_list, rank=rank, generated_image_path=generated_image_path)
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        # Handle GET requests if needed
        return render_template('facesketch.html')

@app.route('/facesketch_create_case', methods=['GET', 'POST'])
def facesketch_create_case():
    if request.method == 'POST':
        case_id = request.form.get('case_id')
        sketch_generator.create_case_files(case_id)
        return jsonify({'success': 'Case created successfully'})
    elif request.method == 'GET':
        # Handle GET requests if needed
        return render_template('facesketch.html')


# __blood stain___#
# Load the trained model for bloodstain
model = joblib.load('random_forest_model.joblib')


# Bloodstain route

@app.route('/bloodstain', methods=['GET', 'POST'])
def blood_stain():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            image_np = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 32))
            img = img / 255.0
            img = img.reshape(1, -1)

            predicted_age = int(model.predict(img)[0])
            accuracy = np.max(model.predict_proba(img)) * 100

            case_id = request.form.get('case_id')
            if case_id:
                with open(f'cases/{case_id}_crimescene.txt', 'a') as f:
                    f.write(f"BloodStain found with age {predicted_age} with accuracy of {accuracy}%\n")

            return jsonify({
                'predicted_age': predicted_age,
                'accuracy': accuracy
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return render_template('bloodstain.html')


# Create case route
@app.route('/blood_create_case', methods=['POST'])
def blood_create_case():
    case_id = request.form.get('case_id')
    if case_id:
        with open(f'cases/{case_id}_crimescene.txt', 'w') as f:
            f.write(f"CaseID: {case_id}\n")
        return jsonify({'success': 'Case created successfully'})
    else:
        return jsonify({'error': 'No case ID provided'})


class FingerprintRecognitionBackend:
    def __init__(self):
        self.model_subject_id = load_model('fingerprint_model0.h5')
        self.model_finger_num = load_model('fingerprint_model1.h5')

    def predict_fingerprint(self, img):
        predictions_subject_id = self.model_subject_id.predict(img)
        predictions_finger_num = self.model_finger_num.predict(img)

        subject_id_predicted = np.argmax(predictions_subject_id) + 1
        finger_num_predicted = np.argmax(predictions_finger_num) + 1

        accuracy_subject_id = np.max(predictions_subject_id)
        accuracy_finger_num = np.max(predictions_finger_num)

        return subject_id_predicted, finger_num_predicted, accuracy_subject_id, accuracy_finger_num

    def finger_update_rank(self, case_id, name, rank):
        if not os.path.exists('cases'):
            os.makedirs('cases')

        case_file = f'cases/{case_id}.txt'
        killer_file = f'cases/{case_id}_killers.txt'

        with open(case_file, 'a') as f:
            f.write(f"{name}: {int(rank)}\n")

        with open(killer_file, 'a') as f:
            f.write(f"Fingerprint Found for Prisoner: {name} (Prisoner ID: {case_id})\n")
            f.write(f"Predictions: Subject ID: {self.subject_id_predicted}, Finger: {self.finger_num_predicted}\n")
            f.write(f"Accuracies: Subject ID: {self.accuracy_subject_id}, Finger: {self.accuracy_finger_num}\n\n")

    def finger_create_case(self, case_id):
        if request.method == 'POST':
            case_id = request.form['case_id']
            success = backend.create_new_case(case_id)
            if success:
                return render_template('fingerprint.html')
            else:
                return jsonify({'error': 'Failed to create case'})
        elif request.method == 'GET':
            return render_template('fingerprint.html')
    def sort_by_rank(self, case_id):
        case_file = f'cases/{case_id}.txt'

        with open(case_file, 'r') as f:
            lines = f.readlines()

        lines.sort(reverse=True, key=lambda x: int(x.split(":")[1].strip()))

        with open(case_file, 'w') as f:
            f.writelines(lines)


Fingerbackend = FingerprintRecognitionBackend()


@app.route('/fingerprint', methods=['GET', 'POST'])
def fingerprint():
    if request.method == 'POST':
        img_file = request.files.get('image')
        case_id = request.form.get('case')

        if not img_file:
            return "Error: No image uploaded"

        if not case_id:
            return "Error: No case selected"

        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        subject_id_predicted, finger_num_predicted, accuracy_subject_id, accuracy_finger_num = Fingerbackend.predict_fingerprint(
            img)

        rank = (accuracy_subject_id + accuracy_finger_num) * 1000

        prisoner_names = pd.read_csv('prisoner_names.csv', encoding='ISO-8859-1')
        prisoner_name = prisoner_names.loc[prisoner_names['SubjectID'] == subject_id_predicted, 'Name'].values
        if len(prisoner_name) > 0:
            prisoner_name = prisoner_name[0]
        else:
            prisoner_name = 'Unknown'

        Fingerbackend.subject_id_predicted = subject_id_predicted
        Fingerbackend.finger_num_predicted = finger_num_predicted
        Fingerbackend.accuracy_subject_id = accuracy_subject_id
        Fingerbackend.accuracy_finger_num = accuracy_finger_num

        Fingerbackend.finger_update_rank(case_id, prisoner_name, rank)
        Fingerbackend.sort_by_rank(case_id)

        return jsonify({
            'subject_id': int(subject_id_predicted),
            'finger': int(finger_num_predicted),
            'accuracy_subject': float(accuracy_subject_id),
            'accuracy_finger': float(accuracy_finger_num),
            'prisoner_name': prisoner_name,
            'rank': int(rank),
            'case_id': case_id
        })
    return render_template('fingerprint.html')


class VictimPredictionBackend:
    def __init__(self):
        # Initialize variables and models
        self.rf_model = RandomForestClassifier()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.case_files_dir = 'cases/'

        # Load and preprocess data for number of victims prediction
        file_path = "C:\\Users\\hp\\Desktop\\University\\4th Year\\Graduation Project\\Serial Killer\\Serial_Killer_Pattern.csv"
        data = pd.read_csv(file_path)
        data = data[['# Male', '# Female', '# White', '# Black', '# Hisp', '#Asian', 'NumVics']]
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        X = data_imputed.drop(columns=['NumVics'])
        y = np.round(data_imputed['NumVics'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)

        # Train SVM model for number of victims prediction
        self.svm_model = SVC(C=1, gamma='scale')
        self.svm_model.fit(self.X_train_scaled, y_train)
        self.y_pred_svm = self.svm_model.predict(self.X_test_scaled)
        self.accuracy_svm = accuracy_score(y_test, self.y_pred_svm)

    def fit_tfidf_vectorizer(self, csv_file_path):
        # Fit TF-IDF vectorizer for victim prediction
        df = pd.read_csv(csv_file_path)
        df.dropna(subset=['Aamodt Type'], inplace=True)
        text_data = df['Aamodt Type'].tolist()
        self.tfidf_vectorizer.fit(text_data)
        targets = df['Victim Description']
        self.rf_model.fit(self.tfidf_vectorizer.transform(text_data), targets)

    def predict_victim_description(self, aamodt_type_description):
        # Predict victim description
        new_aamodt_type = [aamodt_type_description]
        new_aamodt_type_tfidf = self.tfidf_vectorizer.transform(new_aamodt_type)
        prediction = self.rf_model.predict(new_aamodt_type_tfidf)
        accuracy = np.max(self.rf_model.predict_proba(new_aamodt_type_tfidf))
        return prediction[0], accuracy

    def predict_number_of_victims(self, input_values):
        # Predict number of victims
        input_scaled = StandardScaler().fit_transform([input_values])
        prediction = self.svm_model.predict(input_scaled)[0]

        # Calculate accuracy
        predicted_values = self.svm_model.predict(input_scaled)
        accuracy = accuracy_score([prediction], predicted_values)

        return prediction, accuracy

    def create_new_case(self, case_id):
        try:
            # Create a new case file with the provided case ID
            case_file_path = os.path.join(self.case_files_dir, f"{case_id}.txt")
            with open(case_file_path, 'w') as f:
                pass
            # Create a new killers file with the provided case ID
            killers_file_path = os.path.join(self.case_files_dir, f"{case_id}_killers.txt")
            with open(killers_file_path, 'w') as f:
                pass
            return True
        except Exception as e:
            print(e)
            return False

    def update_case_file(self, case_file, name, rank, victim_type, prediction, accuracy):
        try:
            case_file_path = os.path.join(self.case_files_dir, case_file)
            killers_file_path = os.path.join(self.case_files_dir, f"{case_file.split('.')[0]}_killers.txt")

            # Check if the name already exists in the case file
            name_found = False
            existing_ranks = []
            if os.path.exists(case_file_path):
                with open(case_file_path, 'r') as f:
                    for line in f:
                        if name in line:
                            name_found = True
                            existing_ranks.append(int(line.split(': ')[1]))

            # Update case file
            with open(case_file_path, 'a') as f:
                if name_found:
                    # Add rank to existing ranks
                    rank += sum(existing_ranks)
                    f.write(f"{name}: {rank}\n")
                else:
                    f.write(f"{name}: {rank}\n")

            # Update killers file
            with open(killers_file_path, 'a') as f:
                f.write(
                    f"Killer: {name}, Killed: {victim_type}, Predicted Victim Type: {prediction}, Accuracy: {accuracy}\n")

            # Sort ranks in descending order and rewrite the case file
            with open(case_file_path, 'r') as f:
                lines = f.readlines()
            ranks = [int(line.split(': ')[1]) for line in lines]
            ranks.sort(reverse=True)
            with open(case_file_path, 'w') as f:
                for r in ranks:
                    f.write(f"Rank: {r}\n")

            return True
        except Exception as e:
            print(e)
            return False


backend = VictimPredictionBackend()
backend.fit_tfidf_vectorizer('reports_with_description_and_sentence.csv')


@app.route('/killer')
def killer():
    return render_template('killer.html')


@app.route('/victim', methods=['GET', 'POST'])
def victim_prediction():
    if request.method == 'POST':
        aamodt_type_description = request.form['aamodt_type']
        case_file = request.form['case_file']
        name = request.form['name']
        victim_type = request.form['victim_type']

        prediction, accuracy = backend.predict_victim_description(aamodt_type_description)

        if prediction == victim_type:
            success = backend.update_case_file(case_file, name, 250, victim_type, prediction, accuracy)
            if not success:
                return jsonify({'error': 'Failed to update case file'})

        return render_template('victim.html', prediction=prediction, accuracy=accuracy)
    elif request.method == 'GET':
        case_files = os.listdir(backend.case_files_dir)
        return render_template('victim.html', case_files=case_files)


from flask import session, redirect, url_for

@app.route('/victim_create_case', methods=['POST', 'GET'])
def victim_create_case():
    # Store the referrer URL in the session
    referrer_url = request.referrer
    if referrer_url:
        session['previous_page'] = referrer_url

    if request.method == 'POST':
        case_id = request.form['case_id']
        success = backend.create_new_case(case_id)
        if success:
            # Redirect back to the previous page stored in the session
            return redirect(session.pop('previous_page', '/'))
        else:
            return jsonify({'error': 'Failed to create case'})
    elif request.method == 'GET':
        return render_template('victim.html')



@app.route('/numberofvic', methods=['POST'])
def number_of_victims_prediction():
    if request.method == 'POST':
        input_values = [float(request.form[f'feature{i}']) for i in range(1, 7)]
        prediction, accuracy = backend.predict_number_of_victims(input_values)
        print('Prediction:', prediction)  # Check prediction in Flask console
        print('Accuracy:', accuracy)  # Check accuracy in Flask console
        return jsonify({'prediction': prediction, 'accuracy': accuracy})

@app.route('/numberofvic', methods=['GET'])
def number_of_victims_form():
    return render_template('numberofvic.html')


# Enter Case ID page
# Enter Case ID page
@app.route('/case')
def enter_case_id():
    return render_template('enter_case_id.html')


# View Rank page
@app.route('/rank/<case_id>', methods=['GET', 'POST'])
def view_rank(case_id):
    if request.method == 'GET':
        if check_files_exist(case_id):
            case_data = read_case_data(case_id)
            return render_template('rank.html', case_id=case_id, case_data=case_data)
        else:
            return "Files for this case do not exist."
    else:
        return redirect(url_for('rank', case_id=case_id))


@app.route('/rank', methods=['POST', 'GET'])
def rank():
    if request.method == 'POST':
        case_id = request.form['case_id']
        if check_files_exist(case_id):
            crime_scene_details = read_crime_scene_evidence(case_id)
            killer_details = read_killer_evidence(case_id)
            case_data = read_case_data(case_id)
            sorted_suspects = sort_suspects_by_rank(case_data)
            return render_template('view_rank.html', case_id=case_id, crime_scene_details=crime_scene_details,
                                   killer_details=killer_details, suspects=sorted_suspects)
        else:
            return "Files for this case do not exist."
    else:
        return render_template('rank.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/contactus', methods=['GET', 'POST'])
def contact_us():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']

        # Write the data to a CSV file
        with open('contact_us.csv', 'a', newline='') as csvfile:
            fieldnames = ['Name', 'Email', 'Phone', 'Message']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({'Name': name, 'Email': email, 'Phone': phone, 'Message': message})

        return render_template('contact_us.html', success=True)
    return render_template('contact_us.html', success=False)


# Utility functions


def read_crime_scene_evidence(case_id):
    crimescene_file_path = os.path.join('cases', f"{case_id}_crimescene.txt")
    with open(crimescene_file_path, 'r') as f:
        crime_scene_details = f.read()
    return crime_scene_details


def read_killer_evidence(case_id):
    killers_file_path = os.path.join('cases', f"{case_id}_killers.txt")
    with open(killers_file_path, 'r') as f:
        killer_details = f.read()
    return killer_details


def check_files_exist(case_id):
    # Check if any of the specified files exist for the given case ID
    file_names = [f'cases/{case_id}.txt', f'cases/{case_id}_killer.txt', f'cases/{case_id}_crimescene.txt']
    for file_name in file_names:
        try:
            with open(file_name, 'r') as f:
                pass  # File exists, return True
        except FileNotFoundError:
            continue  # File not found, continue checking
        else:
            return True  # Found a file, return True
    return False  # None of the files found


def read_case_data(case_id):
    case_data = []
    case_file_path = os.path.join('cases', f"{case_id}.txt")
    with open(case_file_path, 'r') as f:
        for line in f:
            case_data.append(line.strip().split(':'))
    return case_data


# Function to sort suspects by rank
def sort_suspects_by_rank(suspects):
    return sorted(suspects, key=lambda x: int(x[1]), reverse=True)


# Check if any of the specified files exist for the given case ID
def check_files_exist(case_id):
    file_names = [f'cases/{case_id}.txt', f'cases/{case_id}_killer.txt', f'cases/{case_id}_crimescene.txt']
    for file_name in file_names:
        if os.path.exists(file_name):
            return True
    return False


# Function to get feedback from the CSV file
def get_feedback(case_id):
    feedback_list = []
    try:
        with open('feedback.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Case ID') == case_id:
                    feedback_list.append(row.get('Feedback'))
    except FileNotFoundError:
        print("Feedback file not found.")
    return feedback_list


# Route for viewing feedback
@app.route('/view_feedback', methods=['GET', 'POST'])
def view_feedback():
    if request.method == 'POST':
        case_id = request.form['case_id']

        # Check if the case ID exists in any of the files
        if check_files_exist(case_id):
            # Read feedback from CSV file
            feedback = get_feedback(case_id)
            return render_template('view_feedback.html', case_id=case_id, feedback=feedback)
        else:
            return "Error: Case ID not found."

    return render_template('view_feedback.html')  # Render the form HTML


@app.route('/submit_feedback', methods=['GET', 'POST'])
def submit_feedback():
    if request.method == 'POST':
        case_number = request.form['case_number']
        feedback_text = request.form['feedback_text']

        # Save the feedback to a file
        save_feedback(case_number, feedback_text)

        # Return the same template with a success message
        return render_template('submit_feedback.html', success_message='Case submitted successfully!')

    return render_template('submit_feedback.html')


def save_feedback(case_number, feedback_text):
    # Define the directory to save the feedback files
    feedback_dir = 'case'

    # Check if the directory exists, if not, create it
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)

    # Create the file path
    file_path = os.path.join(feedback_dir, f'{case_number}.txt')

    # Write the feedback to the file without modification
    with open(file_path, 'w') as f:
        f.write(feedback_text)


def sort_suspects_by_rank(suspects):
    return sorted(suspects, key=lambda x: int(x[1]), reverse=True)


app.jinja_env.globals['enumerate'] = enumerate


@app.route('/view_case', methods=['GET', 'POST'])
def view_case():
    case_content = None
    if request.method == 'POST':
        case_number = request.form['case_number']
        case_content = get_case_content(case_number)
    return render_template('case_display.html', case_content=case_content)


def get_case_content(case_number):
    case_dir = 'case'
    file_path = os.path.join(case_dir, f'{case_number}.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    else:
        return None


# Route for the investigator home page
@app.route('/investigator_homepage', methods=['GET', 'POST'])
def investigator_homepage():
    if request.method == 'POST':
        # If the button is clicked, redirect to the homepage
        return redirect(url_for('home'))
    else:
        return render_template('investigator_homepage.html')


# Route for the homepage
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # If the button is clicked, redirect to the investigator home page
        return redirect(url_for('investigator_homepage'))
    else:
        return render_template('home.html')


@app.route('/lawyer_homepage')
def lawyer_homepage():
    if 'name' in session and 'title' in session and session['title'] == 'Lawyer':
        return render_template('lawyer_homepage.html')
    else:
        return redirect(url_for('login'))
def read_csv():
    with open('users.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        users = [row for row in reader]
    return users

def write_csv(users):
    fieldnames = ['Full Name', 'Type', 'Email', 'UserName', 'Password', 'Approved', 'Security Code']
    with open('users.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)

@app.route('/accept_accounts')
def accept_accounts():
    users = read_csv()
    pending_users = [user for user in users if user['Approved'].strip().lower() == 'false']
    return render_template('accept_accounts.html', users=pending_users)

@app.route('/approve/<full_name>', methods=['POST'])
def approve_account(full_name):
    users = read_csv()
    for user in users:
        if user['Full Name'] == full_name:
            user['Approved'] = 'TRUE'
    write_csv(users)
    return redirect(url_for('accept_accounts'))

@app.route('/reject/<full_name>', methods=['POST'])
def reject_account(full_name):
    users = read_csv()
    users = [user for user in users if user['Full Name'] != full_name]
    write_csv(users)
    return redirect(url_for('accept_accounts'))


@app.route('/admin_homepage')
def admin_homepage():
    if 'name' in session and 'title' in session and session['title'] == 'Admin':
        return render_template('admin_homepage.html')
    else:
        return redirect(url_for('login'))


@app.route('/casefeedback', methods=['GET', 'POST'])
def case_feedback():
    if request.method == 'POST':
        case_id = request.form['case_id']
        feedback = request.form['feedback']

        # Check if any of the specified files exist for the given case ID
        if check_files_exist(case_id):
            # Write the feedback to feedback.csv
            with open('feedback.csv', 'a', newline='') as csvfile:
                fieldnames = ['Case ID', 'Feedback']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write headers if file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                writer.writerow({'Case ID': case_id, 'Feedback': feedback})

            return "Feedback submitted successfully!"
        else:
            return "Case ID not found!"

    return render_template('case_feedback.html')


@app.route('/forensic_scientist_homepage')
def forensic_scientist_homepage():
    if 'name' in session and 'title' in session and session['title'] == 'Forensic Scientist':
        return render_template('forensic_scientist_homepage.html')
    else:
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, port=5004)
