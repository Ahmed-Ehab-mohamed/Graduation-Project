from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("autopsy report new.csv")
data['Text'] = data['Secondary Cause'] + ' ' + data['Manner of Death'] + ' ' + data['Primary Cause']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Primary Cause'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit and transform on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Predictions on the testing set
X_test_tfidf = tfidf.transform(X_test)
svm_predictions = svm_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, svm_predictions)

@app.route('/')
def index():
    return render_template('primary.html')

@app.route('/predict_primary_cause', methods=['POST'])
def predict_primary_cause():
    secondary_cause = request.form['secondary_cause']
    manner_of_death = request.form['manner_of_death']
    input_text = f"{secondary_cause} {manner_of_death}"
    input_vector = tfidf.transform([input_text])
    predicted_cause = svm_model.predict(input_vector)[0]
    return jsonify({'predicted_primary_cause': predicted_cause, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
