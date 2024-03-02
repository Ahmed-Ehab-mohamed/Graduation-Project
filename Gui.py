#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home Page")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        glass_button = QPushButton("Glass Type Predictor")
        glass_button.clicked.connect(self.open_glass_predictor)
        layout.addWidget(glass_button)

        fingerprint_button = QPushButton("Fingerprint Recognition")
        fingerprint_button.clicked.connect(self.open_fingerprint_recognition)
        layout.addWidget(fingerprint_button)

        self.setLayout(layout)

    def open_glass_predictor(self):
        self.glass_predictor = GlassTypePredictor()
        self.glass_predictor.show()

    def open_fingerprint_recognition(self):
        self.fingerprint_recognition = FingerprintApp()
        self.fingerprint_recognition.show()

class GlassTypePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Glass Type Predictor')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.setStyleSheet("background-color: lightgray;")

        self.sliders = {}
        for feature_name in features:
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(slider_ranges[feature_name][0] * 10000))
            slider.setMaximum(int(slider_ranges[feature_name][1] * 10000))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            slider.setSingleStep(int(slider_ranges[feature_name][2] * 10000))
            slider.valueChanged.connect(self.on_slider_value_changed)
            layout.addWidget(slider)
            label = QLabel(f"{feature_name}: {slider.value() / 10000:.4f}")
            layout.addWidget(label)
            self.sliders[feature_name] = (slider, label)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.on_predict_button_clicked)
        layout.addWidget(predict_button)

        self.prediction_label = QLabel()
        layout.addWidget(self.prediction_label)
        self.accuracy_label = QLabel()
        layout.addWidget(self.accuracy_label)

        self.setLayout(layout)

        self.update_labels()

    def on_slider_value_changed(self):
        self.update_labels()

    def update_labels(self):
        for feature_name, (slider, label) in self.sliders.items():
            label.setText(f"{feature_name}: {slider.value() / 10000:.4f}")

    def on_predict_button_clicked(self):
        feature_values = [slider.value() / 10000 for slider, _ in self.sliders.values()]
        feature_values_scaled = scaler.transform(np.array(feature_values).reshape(1, -1))
        prediction = svm_mod.predict(feature_values_scaled)
        probability = np.max(svm_mod.predict_proba(feature_values_scaled)) * 100
        self.prediction_label.setText(f'Predicted Glass Type: {prediction[0]}')
        self.accuracy_label.setText(f'Prediction Probability: {probability:.2f}%')

class FingerprintApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fingerprint Recognition")
        self.setGeometry(100, 100, 500, 300)

        self.filename = None

        layout = QVBoxLayout()

        self.image_label = QLabel("No Image Selected")
        self.image_label.setStyleSheet("QLabel { border: 2px solid gray; padding: 10px; background-color: lightgray; }")
        layout.addWidget(self.image_label)

        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setStyleSheet("QPushButton:hover { background-color: lightblue } QPushButton:pressed { background-color: lightgreen }")
        layout.addWidget(self.select_button)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_fingerprint)
        self.predict_button.setStyleSheet("QPushButton:hover { background-color: lightblue } QPushButton:pressed { background-color: lightgreen }")
        layout.addWidget(self.predict_button)

        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(self.go_home)
        self.home_button.setStyleSheet("QPushButton:hover { background-color: lightblue } QPushButton:pressed { background-color: lightgreen }")
        self.home_button.setFixedSize(50, 30)
        self.home_button.move(self.width() - self.home_button.width() - 10, 10)
        layout.addWidget(self.home_button)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("QLabel { border: 2px solid gray; padding: 10px; background-color: lightyellow; }")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.model_subject_id = load_model('fingerprint_model0.h5')
        self.model_finger_num = load_model('fingerprint_model1.h5')

        self.prisoner_names = pd.read_csv('prisoner_names.csv')

        self.true_subject_ids = np.random.randint(1, high=101, size=100)

    def select_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.bmp *.jpg *.png)")
        if filename:
            self.image_label.setPixmap(QPixmap(filename))
            self.filename = filename

    def predict_fingerprint(self):
        if self.filename is not None:
            img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (96, 96))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            subject_id_predicted, finger_num_predicted, accuracy_subject_id, accuracy_finger_num = \
                self.get_backend_predictions(img)

            prisoner_name = self.prisoner_names[self.prisoner_names['SubjectID'] == subject_id_predicted]['Name'].values
            if len(prisoner_name) > 0:
                prisoner_name = prisoner_name[0]
            else:
                prisoner_name = 'Unknown'

            finger_names = ["little", "ring", "middle", "index", "thumb"]
            finger = finger_names[(finger_num_predicted - 1) % 5]

            result_text = f"Predicted: Person {subject_id_predicted}, Finger {finger}, Name: {prisoner_name}\n"
            result_text += f"Accuracy - Subject ID: {accuracy_subject_id:.2f}%, Finger Num: {accuracy_finger_num:.2f}%"
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("Please select an image first")

    def go_home(self):
        self.result_label.setText("")
        self.image_label.setText("No Image Selected")

    def get_backend_predictions(self, img):
        predictions_subject_id = self.model_subject_id.predict(img)
        predictions_finger_num = self.model_finger_num.predict(img)

        subject_id_predicted = np.argmax(predictions_subject_id) + 1
        finger_num_predicted = np.argmax(predictions_finger_num) + 1

        accuracy_subject_id = np.max(predictions_subject_id)
        accuracy_finger_num = np.max(predictions_finger_num)

        return subject_id_predicted, finger_num_predicted, accuracy_subject_id, accuracy_finger_num

# Home Page UI
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

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == np.argmax(y_pred, axis=1))

# Load dataset
glass_df = pd.read_csv("glass (1).csv")

# Features and target
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'

# Slider ranges
slider_ranges = {'RI': (1.50, 1.54, 0.0001),
                 'Na': (10.7, 17.4, 0.1),
                 'Mg': (0, 4.5, 0.1),
                 'Al': (0, 3.5, 0.1),
                 'Si': (69, 76, 1),
                 'K': (0, 6, 0.1),
                 'Ca': (5, 17, 0.1),
                 'Ba': (0, 4, 0.1),
                 'Fe': (0, 1.5, 0.1)}

# Split data into features and target
X = glass_df[features]
y = glass_df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
svm_mod = SVC(probability=True)
svm_mod.fit(X_scaled, y)

# Run the application
if __name__ == '__main__':
    app = QApplication([])
    home_page = HomePage()
    home_page.show()
    sys.exit(app.exec_())


# In[ ]:





# In[ ]:




