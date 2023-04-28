import streamlit as st
import pandas as pd
import pickle

# Charger le modèle depuis le fichier pickle
model_rf = pickle.load(open('../models/model_carprice.pkl', 'rb'))

# Charger les données du fichier CSV
df = pd.read_csv('../data/carprice_clean.csv')

# Créer une liste des colonnes numériques et catégorielles
numeric_cols = ['longueur_voiture(cm)', 'largeur_voiture(cm)', 'poids_vehicule(kg)', 'chevaux', 'consommation_ville(L/100km)', 'consommation_autoroute(L/100km)', 'taille_moteur']
categorial_cols = ['carburant', 'turbo', 'type_vehicule', 'roues_motrices', 'emplacement_moteur', 'type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque', 'modele']

# Créer une interface utilisateur pour saisir les informations de la voiture
st.write("# Prédiction du prix d'une voiture")

longueur = st.slider('Longueur de la voiture (cm)', float(df['longueur_voiture(cm)'].min()), float(df['longueur_voiture(cm)'].max()), float(df['longueur_voiture(cm)'].mean()))
largeur = st.slider('Largeur de la voiture (cm)', float(df['largeur_voiture(cm)'].min()), float(df['largeur_voiture(cm)'].max()), float(df['largeur_voiture(cm)'].mean()))
poids = st.slider('Poids de la voiture (kg)', float(df['poids_vehicule(kg)'].min()), float(df['poids_vehicule(kg)'].max()), float(df['poids_vehicule(kg)'].mean()))
chevaux = st.slider('Puissance du moteur (chevaux)', int(df['chevaux'].min()), int(df['chevaux'].max()), int(df['chevaux'].mean()))
taille_moteur = st.slider('taille moteur', int(df['taille_moteur'].min()), int(df['taille_moteur'].max()), int(df['taille_moteur'].mean()))
consommation_ville = st.slider('Consommation en ville (L/100km)', float(df['consommation_ville(L/100km)'].min()), float(df['consommation_ville(L/100km)'].max()), float(df['consommation_ville(L/100km)'].mean()))
consommation_autoroute = st.slider('Consommation sur autoroute (L/100km)', float(df['consommation_autoroute(L/100km)'].min()), float(df['consommation_autoroute(L/100km)'].max()), float(df['consommation_autoroute(L/100km)'].mean()))

marque = st.selectbox('marque', options=list(df['marque'].unique()))
modele = st.selectbox('modele', options=list(df['modele'].unique()))
roues_motrices = st.selectbox('roues motrices', options=list(df['roues_motrices'].unique()))
emplacement_moteur = st.selectbox('emplacement moteur', options=list(df['emplacement_moteur'].unique()))
type_moteur = st.selectbox('Type de moteur', options=list(df['type_moteur'].unique()))
nombre_cylindres = st.selectbox('nombre cylindres', options=list(df['nombre_cylindres'].unique()))
systeme_carburant = st.selectbox('systeme carburant', options=list(df['systeme_carburant'].unique()))
carburant = st.selectbox('Type de carburant', options=list(df['carburant'].unique()))
turbo = st.selectbox('Turbo', options=list(df['turbo'].unique()))
type_vehicule = st.selectbox('Type de véhicule', options=list(df['type_vehicule'].unique()))

# Ajouter un bouton pour prédire le prix de la voiture
if st.button('Prédire le prix de la voiture'):
    # Créer un dictionnaire avec les données de la voiture
    data = {
        'longueur_voiture(cm)': longueur,
        'largeur_voiture(cm)': largeur,
        'poids_vehicule(kg)': poids,
        'chevaux': chevaux,
        'taille_moteur' : taille_moteur,
        'consommation_ville(L/100km)': consommation_ville,
        'consommation_autoroute(L/100km)': consommation_autoroute,
        'carburant': carburant,
        'turbo': turbo,
        'type_vehicule': type_vehicule,
        'roues_motrices': roues_motrices,
        'emplacement_moteur': emplacement_moteur,
        'type_moteur': type_moteur,
        'nombre_cylindres': nombre_cylindres,
        'systeme_carburant': systeme_carburant,
        'marque': marque,
        'modele': modele
    }

    # Créer un dataframe à partir du dictionnaire de données
    input_data = pd.DataFrame(data, index=[0])

    # Faire la prédiction avec le modèle
    prediction = model_rf.predict(input_data)

    # Afficher le résultat de la prédiction
    st.write(f"Le prix de la voiture est estimé à {prediction[0]:.2f} Dollars.")
