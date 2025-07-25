from collections import defaultdict 
from time import time
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow

class SpeedEstimator:
    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10):
        self.reg_pts = reg_pts if reg_pts is not None else [(0, 288), (1019, 288)]
        self.names = names
        self.trk_history = defaultdict(list)
        self.view_img = view_img
        self.tf = line_thickness
        self.spd = {}
        self.trkd_ids = []
        self.spdl = spdl_dist_thresh
        self.trk_pt = {}
        self.trk_pp = {}
        self.env_check = check_imshow(warn=True)
        self.speeding_count = 0
        self.counted_ids = set()
        self.speed_limit = 60
        self.panel_width = 400  # Largeur fixe pour le panneau
        self.panel_margin = 25  # Marge interne
        self.panel_height = 120  # Hauteur du panneau

    def draw_dashed_line(self, img, pt1, pt2, color=(0, 0, 0), thickness=2, dash_len=15, space_len=10):
        """Dessine une ligne pointillée sur l'image"""
        dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        direction = ((pt2[0] - pt1[0]) / dist, (pt2[1] - pt1[1]) / dist)

        for i in range(0, dist, dash_len + space_len):
            start = (int(pt1[0] + direction[0] * i), int(pt1[1] + direction[1] * i))
            end = (int(pt1[0] + direction[0] * min(i + dash_len, dist)), 
                   int(pt1[1] + direction[1] * min(i + dash_len, dist)))
            cv2.line(img, start, end, color, thickness, lineType=cv2.LINE_AA)

    def draw_speed_panel(self, img):
        """Affiche le panneau d'information de vitesse"""
        # Position et dimensions
        panel_x = img.shape[1] - self.panel_width - self.panel_margin
        panel_y = self.panel_margin
        
        # Fond semi-transparent
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (panel_x, panel_y), 
                     (panel_x + self.panel_width, panel_y + self.panel_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Bordure
        cv2.rectangle(img, 
                     (panel_x, panel_y),
                     (panel_x + self.panel_width, panel_y + self.panel_height),
                     (0, 0, 200), 2)
        
        # Configuration du texte
        font_title = cv2.FONT_HERSHEY_SIMPLEX
        font_text = cv2.FONT_HERSHEY_DUPLEX
        
        # Titre principal (centré)
        title = "SPEED MONITORING"
        title_size = cv2.getTextSize(title, font_title, 0.8, 2)[0]
        title_x = panel_x + (self.panel_width - title_size[0]) // 2
        title_y = panel_y + 35
        cv2.putText(img, title, (title_x, title_y), 
                   font_title, 0.8, (0, 100, 255), 2)
        
        # Ligne séparatrice
        sep_y = title_y + 10
        cv2.line(img, 
                (panel_x + 20, sep_y),
                (panel_x + self.panel_width - 20, sep_y),
                (0, 0, 200), 1)
        
        # Limite de vitesse
        limit_text = f"Speed limit: {self.speed_limit} km/h"
        limit_y = sep_y + 30
        cv2.putText(img, limit_text, (panel_x + 20, limit_y),
                   font_text, 0.9, (200, 200, 200), 1)
        
        # Compteur d'infractions (centré)
        count_text = f"Speeding: {self.speeding_count}"
        count_size = cv2.getTextSize(count_text, font_text, 0.7, 2)[0]
        count_x = panel_x + (self.panel_width - count_size[0]) // 2
        count_y = limit_y + 30
        cv2.putText(img, count_text, (count_x, count_y),
                   font_text, 0.6, (0, 0, 255), 2)
        

    def draw_vehicle_box(self, img, box, color, label, speed=None):
        """Dessine le cadre du véhicule avec son étiquette"""
        x1, y1, x2, y2 = map(int, box)
        
        # Déterminer la couleur du label
        label_color = (0, 0, 255) if speed and speed > self.speed_limit else (0, 255, 255)
        label_text = f"{speed} km/h (!)" if speed and speed > self.speed_limit else f"{speed} km/h" if speed else label
        
        # Fond semi-transparent
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Coins stylisés
        corner_len = 12
        thickness = 2
        corners = [
            ((x1, y1), (x1 + corner_len, y1)),  # Haut gauche
            ((x1, y1), (x1, y1 + corner_len)),
            ((x2, y1), (x2 - corner_len, y1)),  # Haut droit
            ((x2, y1), (x2, y1 + corner_len)),
            ((x1, y2), (x1 + corner_len, y2)),  # Bas gauche
            ((x1, y2), (x1, y2 - corner_len)),
            ((x2, y2), (x2 - corner_len, y2)),  # Bas droit
            ((x2, y2), (x2, y2 - corner_len))
        ]
        for pt1, pt2 in corners:
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Étiquette de texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
        
        # Position de l'étiquette (au-dessus si possible)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + text_size[1] + 10
        
        # Fond de l'étiquette
        cv2.rectangle(img,
                     (label_x, label_y - text_size[1] - 4),
                     (label_x + text_size[0] + 6, label_y + 4),
                     label_color, -1)
        
        # Texte
        cv2.putText(img, label_text, (label_x + 3, label_y),
                   font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def estimate_speed(self, im0, tracks):
        """Fonction principale pour estimer et afficher la vitesse"""
        # Afficher le panneau même sans détections
        if tracks[0].boxes.id is None:
            self.draw_speed_panel(im0)
            return im0

        # Ligne de référence
        self.draw_dashed_line(im0, self.reg_pts[0], self.reg_pts[1], color=(0, 0, 0), thickness=self.tf)

        # Récupérer les données des pistes
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        t_ids = tracks[0].boxes.id.int().cpu().tolist()

        current_speeding_ids = set()

        for box, t_id, cls in zip(boxes, t_ids, clss):
            # Mettre à jour l'historique de la piste
            track = self.trk_history[t_id]
            bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track.append(bbox_center)
            if len(track) > 30:
                track.pop(0)

            # Dessiner la trajectoire
            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            name = self.names[int(cls)].lower()
            color = (255, 0, 0) if name == "car" else \
                   (255, 128, 0) if name == "bus" else \
                   (0, 140, 255) if name == "truck" else \
                   (255, 255, 255)
            
            cv2.polylines(im0, [trk_pts], False, color, self.tf)
            cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), self.tf * 2, color, -1)

            # Calculer et afficher la vitesse
            current_speed = int(self.spd[t_id]) if t_id in self.spd else 0
            label = self.names[int(cls)].capitalize()
            self.draw_vehicle_box(im0, box, color, label, current_speed)

            # Détection des excès de vitesse
            if t_id in self.spd and self.spd[t_id] > self.speed_limit:
                current_speeding_ids.add(t_id)

            # Calcul de vitesse (logique inchangée)
            if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
                continue
                
            if self.reg_pts[1][1] - self.spdl < track[-1][1] < self.reg_pts[1][1] + self.spdl:
                direction = "known"
            elif self.reg_pts[0][1] - self.spdl < track[-1][1] < self.reg_pts[0][1] + self.spdl:
                direction = "known"
            else:
                direction = "unknown"

            if self.trk_pt.get(t_id) != 0 and direction != "unknown" and t_id not in self.trkd_ids:
                self.trkd_ids.append(t_id)
                time_diff = time() - self.trk_pt[t_id]
                if time_diff > 0:
                    self.spd[t_id] = np.abs(track[-1][1] - self.trk_pp[t_id][1]) / time_diff

            self.trk_pt[t_id] = time()
            self.trk_pp[t_id] = track[-1]

        # Mettre à jour le compteur d'infractions
        new_speeding = current_speeding_ids - self.counted_ids
        self.speeding_count += len(new_speeding)
        self.counted_ids.update(current_speeding_ids)

        # Afficher le panneau
        self.draw_speed_panel(im0)

        if self.view_img and self.env_check:
            cv2.imshow("Speed Monitoring", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    vehicle_classes = {0: "car", 1: "bus", 2: "truck"}
    estimator = SpeedEstimator(vehicle_classes, view_img=True)