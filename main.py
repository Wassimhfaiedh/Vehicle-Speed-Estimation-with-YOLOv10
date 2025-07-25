import cv2
import torch
from ultralytics import YOLO
from speed import SpeedEstimator

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Charger le modèle YOLOv10 sur le GPU (si disponible)
model = YOLO("models\yolov10n.pt").to(device)

# Configuration de l'estimateur de vitesse
line_pts = [(101, 371), (1035, 367)]
names = model.model.names  
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# Fonction pour afficher les coordonnées de la souris (optionnel)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse coordinates: ({x}, {y})")

# Chargement de la vidéo
cap = cv2.VideoCapture('speed.mp4')
frame_width, frame_height = 1280, 720  # résolution cible
fps = cap.get(cv2.CAP_PROP_FPS)

# Configuration de la sortie vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou 'avc1' pour H.264
out = cv2.VideoWriter('output_annotated.mp4', fourcc, fps, (frame_width, frame_height))

# Fenêtre pour affichage
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin du flux vidéo ou échec de lecture.")
        break

    frame_count += 1
    if frame_count % 3 != 0:  # Option : traiter une image sur 3
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Inférence avec suivi (classes 2 = voiture, 7 = camion)
    result = model.track(frame, persist=True, classes=[2, 7], verbose=False, device=device)

    if result and hasattr(result[0], "boxes") and result[0].boxes.id is not None:
        processed = speed_obj.estimate_speed(frame.copy(), result)
        out.write(processed)  # Écriture de l'image annotée
        cv2.imshow("RGB", processed)
    else:
        out.write(frame)  # Image originale si rien détecté
        cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
