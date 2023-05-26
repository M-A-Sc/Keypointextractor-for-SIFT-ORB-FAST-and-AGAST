
# M-A-Sc
# Keypointextractor for SIFT, ORB, FAST und AGAST


import cv2
import csv
import time
import os

def detect_keypoints(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray, None)
    return keypoints

def save_features_csv(features, label, csv_path):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(features + [label])

# Picturelabel(0 = nichtMorph, 1 = Morph)
label = input("Geben Sie das Label für die Bilder ein (0 = nichtMorph, 1 = Morph): ")

# Ordnerpfad abfragen
folder_path = input("Geben Sie den Ordnerpfad der Bilder ein: ")

# Output-Ordner erstellen
output_folder = os.path.join(folder_path, "output")
os.makedirs(output_folder, exist_ok=True)

# Detektoren werden Initialisiert
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
fast = cv2.FastFeatureDetector_create()
agast = cv2.AgastFeatureDetector_create()

# Zähler Bilder und Keypoints
total_images = 0
total_sift_keypoints = 0
total_orb_keypoints = 0
total_fast_keypoints = 0
total_agast_keypoints = 0

# Runtime
start_time = time.time()

# Loop durch die Bilder im angegebenen Ordner
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Nur Bilddateien berücksichtigen
        # Bildpfad erstellen
        image_path = os.path.join(folder_path, filename)

        # Bild laden
        image = cv2.imread(image_path)

        # Keypoints für alle Detektor finden
        sift_keypoints = detect_keypoints(image, sift)
        orb_keypoints = detect_keypoints(image, orb)
        fast_keypoints = detect_keypoints(image, fast)
        agast_keypoints = detect_keypoints(image, agast)

        # Anzahl der gefundenen Keypoints je Detektor
        sift_num_keypoints = len(sift_keypoints)
        orb_num_keypoints = len(orb_keypoints)
        fast_num_keypoints = len(fast_keypoints)
        agast_num_keypoints = len(agast_keypoints)

        # Speichern der Keypoints in separaten Vektoren
        sift_vector = [kp.pt for kp in sift_keypoints]
        orb_vector = [kp.pt for kp in orb_keypoints]
        fast_vector = [kp.pt for kp in fast_keypoints]
        agast_vector = [kp.pt for kp in agast_keypoints]

        # Extrahieren der Menge der gefundenen Keypoints für jeden Detektor
        sift_length = len(sift_vector)
        orb_length = len(orb_vector)
        fast_length = len(fast_vector)
        agast_length = len(agast_vector)

        # Speichern der extrahierten Merkmale in einer .csv Datei
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_filename)
        save_features_csv([sift_length, orb_length, fast_length, agast_length], label, csv_path)

        # Aktualisierung der Zähler
        total_images += 1
        total_sift_keypoints += sift_num_keypoints
        total_orb_keypoints += orb_num_keypoints
        total_fast_keypoints += fast_num_keypoints
        total_agast_keypoints += agast_num_keypoints

# Ausgabe der Gesamtzahl der verarbeiteten Bilder und Keypoints pro Detektor
print("Gesamtzahl der Bilder:", total_images)
print("Gesamtzahl der SIFT-Keypoints:", total_sift_keypoints)
print("Gesamtzahl der ORB-Keypoints:", total_orb_keypoints)
print("Gesamtzahl der FAST-Keypoints:", total_fast_keypoints)
print("Gesamtzahl der AGAST-Keypoints:", total_agast_keypoints)

# Berechnung Runtime
execution_time = time.time() - start_time
print("Ausführungszeit: %.2f Sekunden" % execution_time)
