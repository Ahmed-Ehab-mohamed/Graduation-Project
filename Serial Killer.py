#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Load the data
file_path = "C:\\Users\\hp\\Desktop\\University\\4th Year\\Graduation Project\\Serial Killer\\Serial_Killer_Pattern.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data[['# Male', '# Female', '# White', '# Black', '# Hisp', '#Asian', 'NumVics']]

# Replace missing values with the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features (X) and target (y)
X = data_imputed.drop(columns=['NumVics'])
y = data_imputed['NumVics']

# Convert target to integer class labels
y_class = np.round(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(max_iter=1000)
svm_model = SVC()
lstm_model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# Train KNN model
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Train Logistic Regression model with hyperparameter tuning
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=5)
lr_grid.fit(X_train_scaled, y_train)
best_lr_model = lr_grid.best_estimator_
y_pred_lr = best_lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Train SVM model with hyperparameter tuning
svm_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=5)
svm_grid.fit(X_train_scaled, y_train)
best_svm_model = svm_grid.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Prepare LSTM data
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Compile and train LSTM model with 100 epochs
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)
y_pred_lstm = np.round(lstm_model.predict(X_test_lstm))
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
print("LSTM Accuracy:", accuracy_lstm)


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('reports_with_description_and_sentence.csv')

# Handle missing values
data.fillna(value={'Aamodt Type': 'Unknown', 'Victim Description': 'Unknown'}, inplace=True)

# Split data into features (X) and target (y)
X = data['Aamodt Type']  # Feature: Aamodt Type
y = data['Victim Description']  # Target: Victim Description

# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier with hyperparameter tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=3)
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions with Random Forest
y_pred_rf = rf_classifier.predict(X_test_tfidf)

# Calculate accuracy for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Best Parameters:", rf_classifier.best_params_)

# Train the SVM classifier with hyperparameter tuning
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
svm_classifier = GridSearchCV(SVC(), param_grid=svm_param_grid, cv=3)
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions with SVM
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("SVM Best Parameters:", svm_classifier.best_params_)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('reports_with_description_and_sentence.csv')

# Handle missing values
data.fillna(value={'Aamodt Type': 'Unknown', 'Victim Description': 'Unknown'}, inplace=True)

# Split data into features (X) and target (y)
X = data['Aamodt Type']  # Feature: Aamodt Type
y = data['Victim Description']  # Target: Victim Description

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in X_tokenized])
X_padded = pad_sequences(X_tokenized, maxlen=max_sequence_length, padding='post')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define the LSTM model
embedding_dim = 100  # Dimension of word embeddings
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (+1 for padding token)
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("LSTM Accuracy:", accuracy)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('reports_with_description_and_sentence.csv')

# Handle missing values
data.fillna(value={'Aamodt Type': 'Unknown', 'Victim Description': 'Unknown'}, inplace=True)

# Split data into features (X) and target (y)
X = data['Aamodt Type']  # Feature: Aamodt Type
y = data['Victim Description']  # Target: Victim Description

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
history = model.fit(X_train.toarray(), y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
_, accuracy = model.evaluate(X_test.toarray(), y_test)
print("Neural Network Accuracy:", accuracy)


# In[ ]:


import pandas as pd

def find_killer_by_number(file_path, case_number):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Search for the row with the specified case number
    killer_row = data[data['Number'] == case_number]
    
    # Check if the row is found
    if not killer_row.empty:
        # Retrieve the attributes of the killer
        name = killer_row.iloc[0]['Name']
        type_of_killer = killer_row.iloc[0]['Type of Killer']
        aamodt_type = killer_row.iloc[0]['Aamodt Type']
        method_description = killer_row.iloc[0]['Method Description']
        current_status = killer_row.iloc[0]['Current Status']
        victim_description = killer_row.iloc[0]['Victim Description']
        sentence_description = killer_row.iloc[0]['Sentence Description']
        
        # Return the attributes as a dictionary
        return {
            'Name': name,
            'Type of Killer': type_of_killer,
            'Aamodt Type': aamodt_type,
            'Method Description': method_description,
            'Current Status': current_status,
            'Victim Description': victim_description,
            'Sentence Description': sentence_description
        }
    else:
        # If the row is not found, return None
        return None

# Ask the user to input the case number
case_number = input("Enter the case number: ")

# Search for the killer attributes
killer_attributes = find_killer_by_number("reports_with_description_and_sentence.csv", case_number)
if killer_attributes:
    print("Killer found:")
    for key, value in killer_attributes.items():
        print(f"{key}: {value}")
else:
    print("Killer not found.")


# In[ ]:


import pandas as pd
from fpdf import FPDF
import os

# Load the data
data = pd.read_csv('Serial_Killer_Profiling.csv')

# Replace NaN values with "Unknown"
data.fillna("Unknown", inplace=True)

# Create a directory for the reports
report_directory = 'Serial_killer_profile_reports'
if not os.path.exists(report_directory):
    os.makedirs(report_directory)
else:
    # Remove existing files from the directory
    files_in_directory = os.listdir(report_directory)
    filtered_files = [file for file in files_in_directory if file.endswith(".pdf")]
    for file in filtered_files:
        os.remove(os.path.join(report_directory, file))

# Define the mapping of Victim Codes to their descriptions
victim_code_mapping = {
    1.0: 'Street People',
    1.1: 'Prostitute',
    1.10: 'Female',
    1.11: 'Male',
    1.12: 'Male and Female',
    1.2: 'Homeless',
    1.3: 'Junkies',
    1.4: 'Exotic dancers',
    1.5: 'Refugees/Immigrants',
    1.6: 'Migrant workers',
    1.7: 'Cult followers',
    2.0: 'Hitchhikers',
    3.0: 'Johns/Sexual encounters',
    3.1: 'Johns',
    3.2: 'Sexual encounters',
    3.21: 'Straight',
    3.22: 'LGBT',
    3.3: 'Lonely hearts',
    3.4: 'Lovers',
    3.5: 'Former lovers',
    3.6: 'Met at a bar/party',
    4.0: 'Patients/Wards',
    4.1: 'Hospital patients',
    4.2: 'Wards',
    4.3: 'Child care',
    4.4: 'Nursing homes',
    4.5: 'Home care patients',
    4.6: 'Late term abortion',
    4.7: 'Boarders',
    5.0: 'Family',
    5.1: 'Spouse',
    5.2: 'Child',
    5.21: 'Newborn children',
    5.22: 'Step-child',
    5.3: 'Parents',
    5.4: 'Siblings',
    5.5: 'Grandparents',
    5.6: 'Other relatives',
    5.7: 'Girl/Boy friends',
    5.8: 'In-laws',
    5.9: 'Friends',
    5.91: 'Roommates',
    6.0: 'Employees/Customers',
    6.1: 'Employees',
    6.11: 'Maids',
    6.12: 'Slaves/Peons/Serfs',
    6.13: 'Coworkers',
    6.14: 'Job applicants',
    6.2: 'Customers',
    6.21: 'Taxi passengers',
    6.22: 'Train passengers',
    6.3: 'Taxi drivers',
    6.31: 'Truckers',
    6.4: 'Police officers',
    6.41: 'Prison guards',
    6.45: 'Security Guards',
    6.50: 'Lodgers',
    6.51: 'Hotel guests',
    6.6: 'Employer',
    6.65: 'Business Partner',
    6.7: 'Models',
    6.8: 'Judges',
    6.9: 'Military',
    7.0: 'Home invasion',
    7.1: 'Men & Women',
    7.2: 'Women',
    7.3: 'Elderly',
    7.31: 'Men & Women',
    7.32: 'Women',
    7.33: 'Men',
    7.11: 'Men',
    7.4: 'Acquaintances',
    7.5: 'Children',
    7.6: 'Realtors/people selling their home',
    8.0: 'Street',
    8.1: 'Neighbors',
    8.2: 'Tourists',
    8.3: 'General public',
    8.31: 'Women',
    8.32: 'Men',
    8.33: 'Adults - men & women',
    8.34: 'Children - girls',
    8.35: 'Children - boys',
    8.36: 'Children - boys & girls',
    8.37: 'Women - Elderly',
    8.38: 'Men & Women elderly',
    8.4: 'Couples',
    8.5: 'Acquaintances',
    8.52: 'Met at a party/bar',
    8.51: 'Witnesses',
    8.6: 'Gay men',
    8.7: 'Hikers',
    8.71: 'Campers',
    8.81: 'Picked up hitchhiker',
    8.82: 'Motorist with car trouble',
    8.83: 'Taxi passengers',
    8.84: 'Truck drivers',
    8.9: 'Mentally retarded',
    9.0: 'Convenience',
    10.0: 'Criminals',
    10.1: 'Drug dealers',
    10.11: 'Drug customers',
    10.2: 'Pedophiles',
    10.3: 'Gang members',
    10.4: 'Inmates',
    10.41: 'Escaped inmates',
    10.5: 'Looters',
    10.6: 'Rival criminal gangs',
    10.7: 'Informants/witnesses',
    10.8: 'Prison inmates',
    10.9: 'Accomplices',
    11.0: 'Multiple victim types',
    12.0: 'Slaves',
}
# Define the race mapping
race_mapping = {
    1: 'White',
    2: 'Black',
    3: 'Hisp',
    4: 'Asian',
    5: 'Native American',
    6: 'Aboriginal',
    8: 'NHPI'
}
sentence_mapping = {
    '1': 'NGRI (Not Guilty by Reason of Insanity)',
    '2': 'Life Imprisonment',
    '3': 'Death Penalty',
    '3.1': 'Death Sentence Commuted to Life Imprisonment',
    '4': 'Acquitted',
    '5': 'Died Prior to Trial',
    '6': 'GBMI (Guilty But Mentally Ill)',
    '7': 'Guilty, Sent to Forensic Hospital',
    '8': 'Not Competent for Trial',
    '9': 'Hospitalized Prior to Trial',
    '10': 'Not Prosecuted',
    '11': 'Escaped Prior to Trial',
    '12': 'Lynched Prior to Trial',
    '14': 'Hung Jury',
    'Unknown': 'Unknown'
}


# Define a function to create a PDF report for each serial killer
def generate_report(killer):
    # Create a new PDF document
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Set the font
    pdf.set_font('Arial', 'B', 16)

    # Add the title of the report
    pdf.cell(0, 10, 'Serial Killer Profile Report', 0, 1, align='C')
    pdf.ln(20)

    # Define the attributes of the serial killer
    attributes = [
        ('Number', killer['Number']),
        ('Name', killer['Name']),
        ('Prison ID', killer['Prison ID']),
        ('Alias or AKA', killer['Alias or AKA']),
        ('Type of Killer', killer['Type of Killer']),
        ('Sex', 'Female' if killer['Sex'] == 1 else 'Male' if killer['Sex'] == 2 else 'Unknown'),
        ('Race', race_mapping.get(killer['Race'], 'Unknown')),
       ('WhiteMale', 'Yes' if killer['WhiteMale'] == 1 else 'No'),
        ('WhiteMale20s', 'Yes' if killer['WhiteMale20s'] == 1 else 'No'),
        ('NumVics', killer['NumVics']),
        ('Country', killer['Country']),
        ('State', killer['State']),
        ('birthyear', killer['birthyear']),
        ('BirthDate', killer['BirthDate']),
        ('Birth Place', killer['Birth Place']),
        ('Birth State', killer['Birth State']),
        ('Age1stKill', killer['Age1stKill']),
        ('AgeLastKill', killer['AgeLastKill']),
        ('Age Group', killer['Age Group']),
        ('DateFirst', killer['DateFirst']),
        ('DateFinal', killer['DateFinal']),
        ('Victim Code', victim_code_mapping.get(killer['Victim Code'], 'Unknown')),
        ('Aamodt Type', killer['Aamodt Type']),
        ('Height in Inches', killer['Height in Inches']),
        ('SchoolProb', 'Yes' if killer['SchoolProb'] == 1 else 'No'),
        ('Teased', 'Yes' if killer['Teased'] == 1 else 'No'),
        ('Degree', killer['Degree']),
        ('IQ1', killer['IQ1']),
        ('Physical Disability', 'Yes' if killer['Physical Disability'] == 1 else 'No'),
        ('Speech Disability', 'Yes' if killer['Speech Disability'] == 1 else 'No'),
        ('Occupation', killer['Occupation']),
        ('Previous Arrests', killer['Previous Arrests']),
        ('Previous jail or prison time', killer['Previous jail or prison time']),
        ('AgeKill', killer['AgeKill']),
        ('AgeSeries', killer['AgeSeries']),
        ('Killed Prior to Series', 'Yes' if killer['Killed Prior to Series'] == 1 else 'No'),
        ('Kill Method', killer['Kill Method']),
        ('Method Description', killer['Method Description']),
        ('# Male', killer['# Male']),
        ('# Female', killer['# Female']),
        ('Race of Victim', killer['Race of Victim']),
        ('# White', killer['# White']),
        ('# Black', killer['# Black']),
        ('# Hisp', killer['# Hisp']),
        ('#Asian', killer['#Asian']),
        ('Same race, sex, age', 'Yes' if killer['Same race, sex, age'] == 1 else 'No'),
        ('Primary Victim Age', killer['Primary Victim Age']),
        ('Date Arrested', killer['Date Arrested']),
        ('Sentence', sentence_mapping.get(killer['Sentence'], 'Unknown')),
        ('Exec', 'Yes' if killer['Exec'] == 1 else 'No'),
        ('Current Status', killer['Current Status']),
        ('Years Between First and Last', killer['Years Between First and Last']),
        ('YearFirst', killer['YearFirst']),
        ('YearFinal', killer['YearFinal'])
    ]

    # Add the attributes to the report in a table format
    col_width = 70  # Initial column width
    for attr, value in attributes:
        # Adjust column width based on the length of attribute name and value
        width = max(len(attr), len(str(value))) * 4
        if width > col_width:
            col_width = width  # Update column width if necessary
        pdf.cell(col_width, 10, f'{attr}:', 0)
        pdf.cell(col_width, 10, str(value), 0, 1)

    # Save the PDF document with a sanitized filename
    killer_name_sanitized = ''.join(c if c.isalnum() else '_' for c in killer['Name'])
    pdf.output(f'{report_directory}/{killer_name_sanitized}.pdf')


# Generate reports for each serial killer
for _, killer in data.iterrows():
    generate_report(killer)

