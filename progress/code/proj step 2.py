# importation des bibliothèque
import cv2  # bibliothèque pour le traitement d'images et de vidéos
import numpy as np  # bibliothèque pour le calcul numérique
from sklearn.cluster import KMeans  # pour l'algorithme de KMeans (clustering)
import speech_recognition as sr  # pour la reconnaissance vocale
import librosa  # pour le traitement audio
import librosa.display  # pour l'affichage des données audio
import matplotlib.pyplot as plt  # pour la visualisation de graphiques
import scipy.io.wavfile as wav  # pour lire des fichiers audio WAV
from vosk import Model, KaldiRecognizer  # pour la reconnaissance vocale avec Vosk
import wave  # pour lire des fichiers WAV
import os  # pour interagir avec le système de fichiers

# étape 2: extraction des caractéristiques

# 2.1 : extraction des caractéristiques visuelles

# *****couleur
def extract_dominant_colors(image_path, k=3):
    # Lire l'image
    image = cv2.imread(image_path)
    # Convertir l'image de BGR (OpenCV) à RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshaper l'image en une seule colonne de pixels
    image = image.reshape((-1, 3))

    # Appliquer KMeans pour extraire les couleurs dominantes
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(image)

    # Obtenir les couleurs dominantes
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Extraire les couleurs dominantes d'une image donnée
dominant_colors = extract_dominant_colors('keyframes/keyframe_953.jpg')
# Afficher les couleurs dominantes extraites
print("Dominant Colors:", dominant_colors)

# *****mouvement
# Ouvrir la vidéo pour analyser le mouvement
cap = cv2.VideoCapture('howl_scene.mp4')
ret, prev_frame = cap.read()  # Lire la première image de la vidéo
# Convertir la première image en niveaux de gris
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# Paramètres pour l'optical flow (flux optique)
lk_params = dict(winSize=(15, 15), maxLevel=2)

# Parcourir chaque image de la vidéo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image courante en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les contoures dans l'image précédente
    corners = cv2.goodFeaturesToTrack(prev_gray, 100, 0.3, 7)
    
    if corners is not None:
        # Calculer le flux optique entre l'image précédente et l'image actuelle
        new_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, **lk_params)

        # Dessiner des lignes pour représenter le mouvement des coins détectés
        for i in range(len(corners)):
            x, y = corners[i].ravel()
            x2, y2 = new_corners[i].ravel()
            cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

    prev_gray = gray  # Mettre à jour l'image précédente
    cv2.imshow('Optical Flow', frame)  # Afficher l'image avec le flux optique

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()  # Libérer la capture vidéo
cv2.destroyAllWindows()  # Fermer toutes les fenêtres ouvertes

# 2.2 : extraction des caractéristiques audio

# *****caractéristiques audio
# Charger le fichier audio
sample_rate, audio_data = wav.read("howl_scene_audio.wav")
# Convertir un fichier stéréo en mono si nécessaire
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

print(f"Sample Rate: {sample_rate} Hz")  # Afficher la fréquence d'échantillonnage
print(f"Audio Data Shape: {audio_data.shape}")  # Afficher la forme des données audio

# *****intensité sonore
# Taille de la fenêtre pour l'analyse (par exemple, 1024 échantillons par fenêtre)
frame_size = 1024
# Calculer l'énergie RMS (loudness) pour chaque fenêtre
rms_energy = [np.sqrt(np.mean(audio_data[i:i+frame_size]**2)) for i in range(0, len(audio_data), frame_size)]

# Tracer l'énergie RMS au fil du temps
plt.plot(rms_energy)
plt.title("RMS Energy (Loudness)")  # Titre du graphique
plt.xlabel("Frame")  # Étiquette de l'axe x
plt.ylabel("Energy")  # Étiquette de l'axe y
plt.show()  # Afficher le graphique

# 2.3 : extraction des caractéristiques textuelles
# Initialiser le reconnaisseur vocal
recognizer = sr.Recognizer()

# Charger le fichier audio pour la reconnaissance vocale
with sr.AudioFile("howl_scene_audio.wav") as source:
    audio = recognizer.record(source)

# Essayer de reconnaître la parole avec la méthode offline PocketSphinx
try:
    print("Recognizing with PocketSphinx...")
    text = recognizer.recognize_sphinx(audio)  # Reconnaître la parole dans l'audio
    print("Extracted Speech:", text)  # Afficher le texte extrait
except sr.UnknownValueError:
    print("Sphinx could not understand the audio")  # Erreur si l'audio n'est pas compris
except sr.RequestError:
    print("Error in request to Sphinx")  # Erreur de demande si le service est inaccessible
'''    
Color Extraction K-Means Clustering OpenCV, Scikit-learn
Motion Analysis Optical Flow (Lucas-Kanade) OpenCV
Audio Processing WAV File Handling Scipy, NumPy
Loudness Estimation RMS Energy Calculation NumPy, Matplotlib
Speech Recognition PocketSphinx (Offline) SpeechRecognition
'''
