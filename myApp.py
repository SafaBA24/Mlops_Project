import tkinter as tk
from tkinter import messagebox
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle depuis MLflow (assurez-vous que l'URI est correct)
model_uri = "models:/BestModel_RandomForest/Production"  # Modèle en stage Production
model = mlflow.sklearn.load_model(model_uri)

# Charger le TfidfVectorizer
tfidf_vectorizer_uri = "models:/TFIDF_Vectorizer/1"  # URI correct du TfidfVectorizer
tfidf_vectorizer = mlflow.sklearn.load_model(tfidf_vectorizer_uri)

# Fonction pour faire la prédiction
def make_prediction():
    try:
        # Récupérer le texte d'entrée
        message = entry_message.get()

        # Vectoriser le message
        message_vectorized = tfidf_vectorizer.transform([message])

        # Faire la prédiction
        prediction = model.predict(message_vectorized)

        # Afficher le résultat (Spam ou Non-Spam)
        result_label.config(text=f"Prediction: {'Spam' if prediction[0] == 1 else 'Non-Spam'}")

    except Exception as e:
        messagebox.showerror("Input Error", f"Une erreur est survenue: {str(e)}")

# Créer la fenêtre principale
root = tk.Tk()
root.title("Classification Spam/Non-Spam")

# Créer les widgets
label_message = tk.Label(root, text="Entrez le message à classifier:")
label_message.pack()

entry_message = tk.Entry(root, width=50)
entry_message.pack()

# Bouton pour effectuer la prédiction
predict_button = tk.Button(root, text="Classer le message", command=make_prediction)
predict_button.pack()

# Label pour afficher le résultat
result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

# Lancer l'application
root.mainloop()
