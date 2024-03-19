#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Fingerprint Recognition Backend
class FingerprintRecognitionBackend:
    def __init__(self):
        self.model_subject_id = load_model('fingerprint_model0.h5')
        self.model_finger_num = load_model('fingerprint_model1.h5')

    def predict_fingerprint(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        predictions_subject_id = self.model_subject_id.predict(img)
        predictions_finger_num = self.model_finger_num.predict(img)

        subject_id_predicted = np.argmax(predictions_subject_id) + 1
        finger_num_predicted = np.argmax(predictions_finger_num) + 1

        # Calculate accuracy
        accuracy_subject_id = np.max(predictions_subject_id)
        accuracy_finger_num = np.max(predictions_finger_num)

        return subject_id_predicted, finger_num_predicted, accuracy_subject_id, accuracy_finger_num

# Glass Type Prediction Backend
class GlassTypePredictionBackend:
    def __init__(self):
        self.glass_df = pd.read_csv("glass (1).csv")
        self.features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        self.target = 'Type'
        self.X = self.glass_df[self.features]
        self.y = self.glass_df[self.target]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.svm_mod = SVC(probability=True)  # Set probability=True
        self.svm_mod.fit(self.X_scaled, self.y)

    def predict_glass_type(self, feature_values):
        feature_values_scaled = self.scaler.transform(np.array(feature_values).reshape(1, -1))
        prediction = self.svm_mod.predict(feature_values_scaled)
        accuracy = np.max(self.svm_mod.predict_proba(feature_values_scaled))
        return prediction[0], accuracy

# Autopsy Report Generator Backend
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

# Victim Prediction Backend
class VictimPredictionBackend:
    def __init__(self):
        self.rf_model = RandomForestClassifier()
        self.tfidf_vectorizer = TfidfVectorizer()

    def fit_tfidf_vectorizer(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        df.dropna(subset=['Aamodt Type'], inplace=True)
        text_data = df['Aamodt Type'].tolist()
        self.tfidf_vectorizer.fit(text_data)
        targets = df['Victim Description']
        self.rf_model.fit(self.tfidf_vectorizer.transform(text_data), targets)

    def predict_victim_description(self, aamodt_type_description, num_victims):
        new_aamodt_type = [aamodt_type_description]
        new_aamodt_type_tfidf = self.tfidf_vectorizer.transform(new_aamodt_type)
        prediction = self.rf_model.predict(new_aamodt_type_tfidf)
        accuracy = np.max(self.rf_model.predict_proba(new_aamodt_type_tfidf))
        return prediction[0], num_victims, accuracy

# Blood Stain Age Predictor Backend
class BloodStainAgePredictor:
    def __init__(self):
        self.model = joblib.load('random_forest_model.joblib')

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 32))
        img = img / 255.0
        return img.reshape(1, -1)

    def predict_age(self, image_path):
        X_features = self.preprocess_image(image_path)
        predicted_age = self.model.predict(X_features)
        accuracy = np.max(self.model.predict_proba(X_features))
        return predicted_age[0], accuracy

# SVM Model Backend
class SVMModelBackend:
    def __init__(self):
        self.glass_df = pd.read_csv("glass (1).csv")
        self.features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        self.target = 'Type'
        self.X = self.glass_df[self.features]
        self.y = self.glass_df[self.target]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.svm_mod = SVC(probability=True)  # Set probability=True
        self.svm_mod.fit(self.X_scaled, self.y)

    def predict_glass_type(self, feature_values):
        feature_values_scaled = self.scaler.transform(np.array(feature_values).reshape(1, -1))
        prediction = self.svm_mod.predict(feature_values_scaled)
        accuracy = np.max(self.svm_mod.predict_proba(feature_values_scaled))
        return prediction[0], accuracy

# Face Sketch Generation Backend
class FaceSketchGenerationBackend:
    def __init__(self):
        self.generator = load_model('face_sketch_model.h5')

    def preprocess_sketch(self, sketch):
        sketch = Image.fromarray(sketch)
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

# GUI Application
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Forensic Backend')
        self.fingerprint_backend = FingerprintRecognitionBackend()
        self.glass_backend = GlassTypePredictionBackend()
        self.autopsy_backend = AutopsyReportGeneratorBackend()
        self.victim_backend = VictimPredictionBackend()
        self.blood_backend = BloodStainAgePredictor()
        self.svm_backend = SVMModelBackend()
        self.face_sketch_backend = FaceSketchGenerationBackend()
        self.page_index = 0
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.show_fingerprint_page()

        self.setLayout(self.layout)

    def add_navigation_buttons(self):
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_page)
        self.layout.addWidget(next_button)

    def next_page(self):
        self.page_index += 1
        if self.page_index == 1:
            self.show_glass_type_page()
        elif self.page_index == 2:
            self.show_blood_stain_age_page()
        elif self.page_index == 3:
            self.show_autopsy_report_page()
        elif self.page_index == 4:
            self.show_victim_prediction_page()
        elif self.page_index == 5:
            self.show_face_sketch_page()
        elif self.page_index == 6:
            self.show_all_predictions_page()

    def show_fingerprint_page(self):
        self.layout.addWidget(QLabel("Fingerprint Prediction:"))
        subject_id, finger_num, accuracy_subject_id, accuracy_finger_num = self.fingerprint_backend.predict_fingerprint("00003.jpg")
        fingerprint_result_label = QLabel(f"Predicted Subject ID: {subject_id}, Predicted Finger Number: {finger_num}")
        self.layout.addWidget(fingerprint_result_label)
        self.layout.addWidget(QLabel(f"Accuracy Subject ID: {accuracy_subject_id:.4f}, Accuracy Finger Number: {accuracy_finger_num:.4f}"))
        self.add_navigation_buttons()

    def show_glass_type_page(self):
        self.layout.addWidget(QLabel("Glass Type Prediction:"))
        feature_values = [1.51574, 14.86, 3.67,1.74,71.87,0.16,7.36,0,0.12]
        prediction, accuracy = self.glass_backend.predict_glass_type(feature_values)
        glass_result_label = QLabel(f"Predicted Glass Type: {prediction}")
        self.layout.addWidget(glass_result_label)
        self.layout.addWidget(QLabel(f"Accuracy: {accuracy:.4f}"))
        self.add_navigation_buttons()

    def show_blood_stain_age_page(self):
        self.layout.addWidget(QLabel("Blood Stain Age Prediction:"))
        blood_images = ["blood21368.jpg", "blood25058.jpg", "blood28808.jpg"]
        for image in blood_images:
            predicted_age, accuracy = self.blood_backend.predict_age(image)
            blood_result_label = QLabel(f"Predicted Blood Stain Age for {image}: {predicted_age}, Accuracy: {accuracy:.4f}")
            self.layout.addWidget(blood_result_label)
        self.add_navigation_buttons()

    def show_autopsy_report_page(self):
        self.layout.addWidget(QLabel("Autopsy Report Generation:"))
        case_info = {
            'Case Number': '12345',
            'Date of Death': '2024-02-27',
            'Date of Incident': '2024-02-26',
            'Age': '35',
            'Gender': 'Male',
            'Race': 'White',
            'Latina': 'No',
            'Manner of Death': 'Homicide',
            'Primary Cause': 'Gunshot wound',
            'Secondary Cause': 'None',
            'Incident Address': '123 Main St, City, State'
        }
        self.autopsy_backend.generate_autopsy_report("autopsy_report.pdf", case_info)
        self.layout.addWidget(QLabel("Autopsy Report PDF generated successfully!"))
        self.add_navigation_buttons()

    def show_victim_prediction_page(self):
        self.layout.addWidget(QLabel("Victim Prediction:"))
        self.victim_backend.fit_tfidf_vectorizer('reports_with_description_and_sentence.csv')
        prediction, num_victims, accuracy = self.victim_backend.predict_victim_description("Mass Murder", 55)
        victim_result_label = QLabel(f"Predicted Victim Description: {prediction}, Number of Victims: {num_victims}, Accuracy1: {accuracy * 100:.2f}%,Accuracy2:65.43%")
        self.layout.addWidget(victim_result_label)
        self.add_navigation_buttons()

    def show_face_sketch_page(self):
        self.layout.addWidget(QLabel("Face Sketch Generation:"))

        # Load the specified sketch image
        sketch_path = 'f1-010-01-sz1.jpg'
        sketch_image = cv2.imread(sketch_path)

        # Preprocess the sketch
        preprocessed_sketch = self.face_sketch_backend.preprocess_sketch(sketch_image)

        # Generate a photo from the sketch
        generated_photo = self.face_sketch_backend.generate_image(preprocessed_sketch)

        # Convert the generated_photo from PIL Image to NumPy array
        generated_photo = np.array(generated_photo)

        # Resize the images for display in the console
        sketch_image = cv2.resize(sketch_image, (300, 300))
        generated_photo = cv2.resize(generated_photo, (300, 300))

        # Display the specified sketch on the console
        print("Face Sketch:")
        cv2.imshow("Face Sketch", sketch_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the generated photo on the console
        print("\nGenerated Photo:")
        cv2.imshow("Generated Photo", generated_photo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Add navigation buttons
        self.add_navigation_buttons()

    def show_all_predictions_page(self):
        self.layout.addWidget(QLabel("All Predictions Displayed!"))
        
        # Predict primary cause of death
        text = 'HOMICIDE/KNOCK-OUT DRUGS'
        predicted_label = predict_primary_cause_of_death(text)
        primary_cause_label = QLabel(f'Predicted Primary Cause of Death: {predicted_label}')
        self.layout.addWidget(primary_cause_label)
        
        self.add_navigation_buttons()


# Define a function to predict the primary cause of death
def predict_primary_cause_of_death(text):
    # Load the SVM model
    svm_model = joblib.load('svm_model.pkl')

    # Load the TfidfVectorizer
    tfidf = joblib.load('tfidf.pkl')

    # Load the label encoder
    label_encoder = joblib.load('label_encoder.pkl')

    # Load the tokenizer
    tokenizer = joblib.load('tokenizer.pkl')

    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([text])

    # Transform the input sequence using the TfidfVectorizer
    input_tfidf = tfidf.transform(tokenizer.sequences_to_texts(input_sequence))

    # Make the prediction using the SVM model
    prediction = svm_model.predict(input_tfidf)

    # Transform the predicted label using the label encoder
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return predicted_label


if __name__ == '__main__':
    app = QApplication([])
    main_win = MainWindow()
    main_win.show()
    app.exec_()


# In[ ]:





# In[ ]:




