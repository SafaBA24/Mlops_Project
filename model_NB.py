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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from mlflow.models.signature import infer_signature
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

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

# Paramètres du modèle Naive Bayes
alpha_value = 0.1

# Entraîner le classificateur Naive Bayes
sklearn_classifier = MultinomialNB(alpha=alpha_value)  # alpha=0.1 pour une meilleure précision
sklearn_classifier.fit(X_train, y_train)

# Inférer la signature du modèle
signature = infer_signature(X_train, sklearn_classifier.predict(X_train))

# Faire des prédictions sur le jeu de test
y_pred = sklearn_classifier.predict(X_test)

# Calculer la précision et afficher le rapport de classification
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc = f"Accuracy is : {accuracy:.2f}"
# Calcul des différentes métriques d'évaluation
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Enregistrer les paramètres et le modèle avec MLflow
mlflow.log_param("alpha", alpha_value)  # Enregistrement du paramètre alpha
mlflow.log_metric("accuracy", accuracy)  # Enregistrement de la précision
# Enregistrer les métriques avec MLflow
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)


# Enregistrer le modèle avec sa signature
mlflow.sklearn.log_model(sklearn_classifier, "NB_model", signature=signature)


