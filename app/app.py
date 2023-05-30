import streamlit as st
import pandas as pd
import pickle

def charger_donnees():
    """Cette fonction charge les données du modèle et du fichier csv."""
    model_rf = pickle.load(open('../models/model_carprice.pkl', 'rb'))
    df = pd.read_csv('../data/carprice_clean.csv')
    return model_rf, df

def interface_utilisateur(df):
    """Cette fonction crée l'interface utilisateur pour entrer les informations de la voiture."""
    st.write("# Prédiction du prix d'une voiture")

    numeric_cols = ['longueur_voiture(cm)', 'largeur_voiture(cm)', 'poids_vehicule(kg)', 
                    'chevaux', 'consommation_ville(L/100km)', 'consommation_autoroute(L/100km)', 
                    'taille_moteur']
    categorial_cols = ['carburant', 'turbo', 'type_vehicule', 'roues_motrices', 
                       'emplacement_moteur', 'type_moteur', 'nombre_cylindres', 
                       'systeme_carburant', 'marque', 'modele']

    data = {}

    for col in numeric_cols:
        data[col] = st.slider(f'{col}', float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        
    for col in categorial_cols:
        data[col] = st.selectbox(f'{col}', options=list(df[col].unique()))

    return data

def faire_prediction(model, input_data):
    """Cette fonction utilise le modèle pour faire une prédiction basée sur les données d'entrée."""
    prediction = model.predict(input_data)
    st.write(f"Le prix de la voiture est estimé à {prediction[0]:.2f} Dollars.")

# Exécution principale
if __name__ == "__main__":
    model_rf, df = charger_donnees()
    user_input = interface_utilisateur(df)
    
    if st.button('Prédire le prix de la voiture'):
        input_data = pd.DataFrame(user_input, index=[0])
        faire_prediction(model_rf, input_data)
