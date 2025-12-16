#A face detection

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier('/home/qulorenzo/Cours/Traitement Images/TD7/haarcascades/haarcascades/haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('/home/qulorenzo/Cours/Traitement Images/TD7/haarcascades/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('/home/qulorenzo/Cours/Traitement Images/TD7/haarcascades/haarcascades/haarcascade_smile.xml')

# cap = cv2.VideoCapture('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/video1.mp4')
cap = cv2.VideoCapture(0)
sunglasses = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/sunglasses.png')
sunglassesAlpha = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/alpha.png')
hat = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/image 1.jpeg')
hatAlpha = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/image 2.jpg')

# Charger le personnage animé
character = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/02_personage.png', cv2.IMREAD_UNCHANGED)
character_alpha = cv2.imread('/home/qulorenzo/Cours/Traitement Images/TD7/images/images/02_personage_alpha.png')

# Paramètres d'animation du personnage
character_y = -200  # Position initiale (au-dessus de l'écran)
character_speed = 3  # Vitesse de déplacement (pixels par frame)
character_width = 150  # Largeur du personnage
character_height = int(character.shape[0] * character_width / character.shape[1]) if character is not None else 200

# Liste pour stocker les étoiles de sourire
smile_stars = []  # Chaque étoile: {'x': x, 'y': y, 'life': life, 'size': size}


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
     # Appliquer le filtre noir et blanc à toute la frame
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
    
    # Convertir en niveaux de gris pour la détection
    gray = gray_full
    
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (fx, fy, fw, fh) in faces:
        # center = (fx + fw//2, fy + fh//2)
        # axes = (fw//2, fh//2)
        # frame = cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 255), 4)
        
        # Ajouter le chapeau au-dessus du visage
        hat_width = int(fw * 1.8)  # Le chapeau est plus large que le visage
        hat_height = int(hat.shape[0] * hat_width / hat.shape[1])
        
        hat_resized = cv2.resize(hat, (hat_width, hat_height))
        hatAlpha_resized = cv2.resize(hatAlpha, (hat_width, hat_height))
        
        # Position du chapeau (au-dessus du visage)
        hat_x = fx + fw//2 - hat_width//2
        hat_y = fy - int(hat_height * 0.8)  # Position au-dessus du front
        
        # Ajuster si hors limites
        start_y = max(0, hat_y)
        start_x = max(0, hat_x)
        end_y = min(hat_y + hat_height, frame.shape[0])
        end_x = min(hat_x + hat_width, frame.shape[1])
        
        # Calculer les offsets dans l'image du chapeau
        offset_y = start_y - hat_y
        offset_x = start_x - hat_x
        
        actual_height = end_y - start_y
        actual_width = end_x - start_x
        
        if actual_height > 0 and actual_width > 0:
            # Extraire la région correspondante du chapeau
            hat_crop = hat_resized[offset_y:offset_y+actual_height, offset_x:offset_x+actual_width]
            alpha_crop = hatAlpha_resized[offset_y:offset_y+actual_height, offset_x:offset_x+actual_width]
            
            # Normaliser le masque alpha
            maskNormalized = cv2.normalize(
                alpha_crop, None, alpha=0, beta=1, 
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            
            if len(maskNormalized.shape) == 2:
                maskNormalized = np.expand_dims(maskNormalized, axis=2)
            
            # Conversion en float32
            hatFloat = np.float32(hat_crop)
            
            # Région du background
            bgRegion = frame[start_y:end_y, start_x:end_x].astype(np.float32)
            
            # Blending alpha
            blended = hatFloat * maskNormalized + bgRegion * (1 - maskNormalized)
            
            # Insertion
            frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        # Détecter les yeux dans la région du visage
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
        # center = (fx + fw//2, fy + fh//2)
        # axes = (fw//2, fh//2)
        # frame = cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 255), 4)
        # On a besoin d'au moins 2 yeux
        if len(eyes) >= 2:
            # Trier les yeux de gauche à droite
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Prendre les 2 premiers yeux
            eye1 = eyes[0]
            eye2 = eyes[1]
            
            # Calculer la position et la taille des lunettes
            # Position du premier œil (relatif au visage)
            x1 = eye1[0] + eye1[2]//2
            y1 = eye1[1] + eye1[3]//2
            
            # Position du deuxième œil
            x2 = eye2[0] + eye2[2]//2
            y2 = eye2[1] + eye2[3]//2
            
            # Distance entre les yeux
            eye_distance = abs(x2 - x1)
            
            # Redimensionner les lunettes proportionnellement
            glasses_width = int(eye_distance * 2.5)  # Ajustez ce facteur
            glasses_height = int(sunglasses.shape[0] * glasses_width / sunglasses.shape[1])
            
            # Vérifier que les dimensions sont valides
            if glasses_width <= 0 or glasses_height <= 0:
                continue
            
            sunglasses_resized = cv2.resize(sunglasses, (glasses_width, glasses_height))
            sunglassesAlpha_resized = cv2.resize(sunglassesAlpha, (glasses_width, glasses_height))
            
            # Position centrale entre les deux yeux (en coordonnées absolues)
            center_x = fx + (x1 + x2) // 2
            center_y = fy + (y1 + y2) // 2
            
            # Position de départ pour les lunettes
            start_x = center_x - glasses_width // 2
            start_y = center_y - glasses_height // 2
            
            # Vérifier les limites
            if start_y >= 0 and start_x >= 0:
                end_y = min(start_y + glasses_height, frame.shape[0])
                end_x = min(start_x + glasses_width, frame.shape[1])
                
                # Ajuster la taille si nécessaire
                actual_height = end_y - start_y
                actual_width = end_x - start_x
                
                if actual_height > 0 and actual_width > 0:
                    # Préparer le masque alpha normalisé
                    alpha_resized = sunglassesAlpha_resized[:actual_height, :actual_width]
                    glasses_crop = sunglasses_resized[:actual_height, :actual_width]
                    maskNormalized = cv2.normalize(
                        alpha_resized, None, alpha=0, beta=1, 
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
                    )
                    
                    if len(maskNormalized.shape) == 2:
                        maskNormalized = np.expand_dims(maskNormalized, axis=2)
                    
                    # Conversion en float32
                    imgFloat = np.float32(glasses_crop)
                    
                    # Région du background
                    bgRegion = frame[start_y:end_y, start_x:end_x].astype(np.float32)
                    
                    # Blending alpha
                    blended = imgFloat * maskNormalized + bgRegion * (1 - maskNormalized)
                    
                    # Insertion
                    frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        # Détecter les sourires dans la région du visage
        roi_gray_smile = gray[fy:fy+fh, fx:fx+fw]
        smiles = smile_cascade.detectMultiScale(roi_gray_smile, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        
        if len(smiles) > 0:
            # Un sourire est détecté ! Ajouter des étoiles
            for _ in range(3):  # Ajouter 3 étoiles
                star = {
                    'x': fx + fw//2 + np.random.randint(-50, 50),
                    'y': fy + fh//2 + np.random.randint(-30, 30),
                    'life': 30,  # Durée de vie en frames
                    'size': np.random.randint(20, 40),
                    'vx': np.random.uniform(-2, 2),  # Vélocité horizontale
                    'vy': np.random.uniform(-3, -1)  # Vélocité verticale (vers le haut)
                }
                smile_stars.append(star)
    
    # Ajouter le personnage animé qui descend
    if character is not None and character_alpha is not None:
        # Convertir en BGR si nécessaire (enlever le canal alpha)
        if character.shape[2] == 4:
            character_bgr = character[:, :, :3]
        else:
            character_bgr = character
        
        # Redimensionner le personnage
        character_resized = cv2.resize(character_bgr, (character_width, character_height))
        character_alpha_resized = cv2.resize(character_alpha, (character_width, character_height))
        
        # Position horizontale (centrée)
        character_x = frame.shape[1] - character_width - 50  # À droite de l'écran
        
        # Vérifier l'intersection avec les visages détectés
        intersection_detected = False
        for (fx, fy, fw, fh) in faces:
            # Rectangle du visage
            face_x1, face_y1 = fx, fy
            face_x2, face_y2 = fx + fw, fy + fh
            
            # Rectangle du personnage
            char_x1, char_y1 = character_x, int(character_y)
            char_x2, char_y2 = character_x + character_width, int(character_y) + character_height
            
            # Détecter l'intersection
            if not (char_x2 < face_x1 or char_x1 > face_x2 or char_y2 < face_y1 or char_y1 > face_y2):
                intersection_detected = True
                break
        
        # Appliquer une teinte différente si intersection
        if intersection_detected:
            # Changer la couleur en rouge (teinte rouge)
            character_resized = character_resized.astype(np.float32)
            character_resized[:, :, 0] = character_resized[:, :, 0] * 0.3  # Réduire le bleu
            character_resized[:, :, 1] = character_resized[:, :, 1] * 0.3  # Réduire le vert
            character_resized[:, :, 2] = np.minimum(character_resized[:, :, 2] * 1.5, 255)  # Augmenter le rouge
            character_resized = character_resized.astype(np.uint8)
        
        # Calculer les limites pour l'overlay
        start_y_char = max(0, int(character_y))
        end_y_char = min(int(character_y) + character_height, frame.shape[0])
        start_x_char = max(0, character_x)
        end_x_char = min(character_x + character_width, frame.shape[1])
        
        # Calculer les offsets dans l'image du personnage
        offset_y_char = start_y_char - int(character_y)
        offset_x_char = start_x_char - character_x
        
        actual_height_char = end_y_char - start_y_char
        actual_width_char = end_x_char - start_x_char
        
        if actual_height_char > 0 and actual_width_char > 0:
            # Extraire la région correspondante
            char_crop = character_resized[offset_y_char:offset_y_char+actual_height_char, 
                                         offset_x_char:offset_x_char+actual_width_char]
            alpha_crop_char = character_alpha_resized[offset_y_char:offset_y_char+actual_height_char, 
                                                      offset_x_char:offset_x_char+actual_width_char]
            
            # Normaliser le masque alpha
            maskNormalized_char = cv2.normalize(
                alpha_crop_char, None, alpha=0, beta=1, 
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            
            if len(maskNormalized_char.shape) == 2:
                maskNormalized_char = np.expand_dims(maskNormalized_char, axis=2)
            
            # Conversion en float32
            charFloat = np.float32(char_crop)
            
            # Région du background
            bgRegion_char = frame[start_y_char:end_y_char, start_x_char:end_x_char].astype(np.float32)
            
            # Blending alpha
            blended_char = charFloat * maskNormalized_char + bgRegion_char * (1 - maskNormalized_char)
            
            # Insertion
            frame[start_y_char:end_y_char, start_x_char:end_x_char] = blended_char.astype(np.uint8)
        
        # Mettre à jour la position (mouvement vers le bas)
        character_y += character_speed
        
        # Réinitialiser la position quand le personnage sort de l'écran
        if character_y > frame.shape[0]:
            character_y = -character_height
    
    # Dessiner et animer les étoiles de sourire
    stars_to_remove = []
    for i, star in enumerate(smile_stars):
        # Mettre à jour la position
        star['x'] += star['vx']
        star['y'] += star['vy']
        star['life'] -= 1
        
        # Calculer l'opacité basée sur la durée de vie
        opacity = star['life'] / 30.0
        
        if star['life'] <= 0:
            stars_to_remove.append(i)
        else:
            # Dessiner une étoile à 5 branches
            center = (int(star['x']), int(star['y']))
            size = star['size']
            
            # Créer les points de l'étoile
            points = []
            for j in range(10):
                angle = np.pi / 2 + j * np.pi / 5
                if j % 2 == 0:
                    r = size
                else:
                    r = size * 0.4
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] - r * np.sin(angle))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            
            # Dessiner l'étoile avec de la transparence
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], (0, 215, 255))  # Couleur dorée (BGR)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
            
            # Contour de l'étoile
            cv2.polylines(frame, [points], True, (0, 165, 255), 2)
    
    # Retirer les étoiles expirées
    for i in reversed(stars_to_remove):
        smile_stars.pop(i)

    # Afficher le résultat
    cv2.imshow('Real-Time Face Detection with Sunglasses', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

