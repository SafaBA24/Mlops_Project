import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import mlflow
import mlflow.sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')

# Charger le dataset
df = pd.read_csv(r'C:\Users\safab\Desktop\Mlops_Projects\train (1).csv')

# Fonction de prétraitement du texte
def preprocess_text(sms):
    words = word_tokenize(sms)  # Tokenisation
    words = [word.lower() for word in words if word.isalnum()]  # Mise en minuscule et filtre des symboles
    words = [word for word in words if word not in stopwords.words("english")]  # Suppression des mots vides (stopwords)
    return " ".join(words)  # Concatenation des mots filtrés

# Appliquer le prétraitement à la colonne 'sms'
df['sms'] = df['sms'].apply(preprocess_text)

# Vectorisation du texte avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['sms']).toarray()
y = df['label']

# Séparer le dataset en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Démarrer une expérience avec MLflow
mlflow.start_run()

# Paramètres du modèle SVM
kernel_value = 'linear'  # Exemple de paramètre pour SVM

# Entraîner le modèle SVM
svm_classifier = SVC(kernel=kernel_value)
svm_classifier.fit(X_train, y_train)

# Inférer la signature du modèle
signature = infer_signature(X_train, svm_classifier.predict(X_train))

# Faire des prédictions
y_pred_svm = svm_classifier.predict(X_test)

# Calculer différentes métriques d'évaluation
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Enregistrer les métriques avec MLflow
accuracy_svm = accuracy_score(y_test, y_pred_svm)
mlflow.log_param("kernel", kernel_value)
mlflow.log_metric("accuracy", accuracy_svm)
mlflow.log_metric("precision", precision_svm)
mlflow.log_metric("recall", recall_svm)
mlflow.log_metric("f1_score", f1_svm)

# Enregistrer le modèle avec sa signature
mlflow.sklearn.log_model(svm_classifier, "svm_model", signature=signature)