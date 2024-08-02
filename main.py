import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('CNN_model.h5')

# Charger les données
(_, _), (x_test, y_test) = mnist.load_data()

# Fonction pour dessiner et prédire un chiffre
def draw_and_predict(model):
    st.title("Reconnaissance de chiffres dessinés à la main")
    st.write("Dessinez un chiffre dans la case ci-dessous.")

    # Ajouter une zone de dessin
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = np.expand_dims(img, axis=-1)
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        st.write(f"Le chiffre prédit est : {np.argmax(prediction)}")

# Fonction pour afficher une image aléatoire et prédire
def show_random_image(model, x_test, y_test):
    st.title('Prédiction de Chiffres')

    if st.button('Afficher une image aléatoire'):
        # Choisir une image aléatoire
        index = np.random.randint(0, len(x_test))
        img = x_test[index]
        label = y_test[index]

        # Afficher l'image
        st.image(img, channels='GRAY', width=300)
        st.write(f'Label réel : {label}')

        # Prédire avec le modèle
        img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour le batch
        img = img / 255.0  # Normaliser si nécessaire
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        # Afficher la prédiction
        st.write(f'Prédiction : {predicted_label}')

        # Bouton pour valider la prédiction
        if st.button('Valider la Prédiction'):
            if predicted_label == label:
                st.success('La prédiction est correcte !')
            else:
                st.error('La prédiction est incorrecte.')

# Options du menu
options = ["Dessiner un chiffre", "Afficher une image aléatoire"]

# Créer le menu de navigation
choice = st.selectbox("Choisissez une application", options)

# Afficher le contenu de la fonction sélectionnée
if choice == "Dessiner un chiffre":
    draw_and_predict(model)
elif choice == "Afficher une image aléatoire":
    show_random_image(model, x_test, y_test)