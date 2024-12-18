
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os
import json
import platform

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ultralytics import YOLO

# Charger le modèle YOLOv8 pré-entraîné pour l'estimation de pose
model = YOLO('yolov8n-pose.pt')


#import config



def calculer_liste_webcams_dispo():
    '''
        Fonction permettant de détecter quelles wecams sont disponible pour la captation live
        Sortie: Une liste contenant les numéro des webcam disponibles
    '''
    index = 0
    arr = []
    while True:
        if not platform.system() == "Windows":
            cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        else:
            cap = cv.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return(arr)



def recuperation_points_table(json_path):
    """
    Fonction permettant de ressortir les points enregistrés pour l'homographie (généralement les coins de la table)
    Entrée: Le chemin du json contenant les points pour l'homographie
    Sortie: un éléments contenant les points
    """

    with open(json_path) as json_file:
        json_course = json.load(json_file)

    # we convert to a flat array [[20,10],[80,10],[95,90],[5,90]]
    scr_pct1 = json_course['calibration']['srcPct1']
    src_pts1 = np.float32(list(map(lambda x: list(x.values()), scr_pct1)))

    src_pts1 = np.float32(json_course['homography']['srcPts'])

    return(src_pts1)


def calcul_nb_cluster_optimal(self,liste_deuxiemes_rebonds):
    """
        Fonction permettant d'appliquer la méthode du coude pour calculer le nombre de clusters optimal pour la méthode du Kmeans
    """
    inertia = list()
    for i in range(2, min(13,len(liste_deuxiemes_rebonds))):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(liste_deuxiemes_rebonds)
        inertia.append(kmeans.inertia_)


    diff = np.diff(inertia)
    diff2 = np.diff(diff)
    elbow_point = np.argmax(diff2) + 2 + 1
    return(elbow_point)


def tracer_table_vide(dim):
    """
        Fonction permettant de creer une image de la table vide avec des dimensions données
        Entrée:
                - La dimension de l'image souhaitée
        Sortie:
                - L'image
    """
    frame_blanc = np.ones(dim, np.uint8) * 255
    height_blanc,width_blanc,_ = frame_blanc.shape
    centre_w = width_blanc//2
    centre_h = height_blanc//2
    cv.line(frame_blanc, (-152//2+centre_w, -274//2+centre_h), (152//2+centre_w, -274//2+centre_h), (0, 0, 0), 1)
    cv.line(frame_blanc, (-152//2+centre_w, -274//2+centre_h), (-152//2+centre_w, 274//2+centre_h), (0, 0, 0), 1)
    cv.line(frame_blanc, (152//2+centre_w, -274//2+centre_h), (152//2+centre_w, 274//2+centre_h), (0, 0, 0), 1)
    cv.line(frame_blanc, (-152//2+centre_w, 274//2+centre_h), (152//2+centre_w, 274//2+centre_h), (0, 0, 0), 1)
    cv.line(frame_blanc, (-152//2-10+centre_w, centre_h), (152//2+10+centre_w, centre_h), (0, 0, 0), 1)
    cv.line(frame_blanc, (centre_w, -274//2+centre_h), (centre_w, 274//2+centre_h), (128, 128, 128), 1)

    return(frame_blanc)

def tracer_image_niveau_gris_uniforme(dim,couleur):
    """
        Fonction permettant de créer une image de couleur en niveau de gris uniforme (ex: np.ones((150,180,3), np.uint8) * 168)
        Entrée:
                - La dimension de l'image souhaitée
                - La couleur souhaitée (entre 0 et 255)
        Sortie:
                - L'image
    """
    if couleur == 0:
        frame_deuxieme_rebond = np.zeros(dim, np.uint8)
    elif couleur == 1:
        frame_deuxieme_rebond = np.ones(dim, np.uint8)
    else:
        frame_deuxieme_rebond = np.ones(dim, np.uint8) * couleur
    return(frame_deuxieme_rebond)


def tracer_rebonds_fleche_sur_table(frame_blanc,liste_premiers_rebonds,liste_deuxiemes_rebonds,couleur):
    """
        Fonction permettant de tracer les rebonds et les flèches d'un rebond à l'autre
        Entrée:
                - L'image de la table
                - La liste des premiers rebonds
                - La liste des deuxiemes rebonds
                - La couleur (ex: (0,0,0))
        Sortie: 
                - L'image avec le tracé
    """
    for p in liste_premiers_rebonds:
        cv.circle(frame_blanc, p, 4, (0, 0, 0), 1)
    for i in range(len(liste_deuxiemes_rebonds)):
        cv.circle(frame_blanc, (liste_deuxiemes_rebonds[i][0],liste_deuxiemes_rebonds[i][1]), 4, couleur, -1)
        if len(liste_premiers_rebonds) == len(liste_deuxiemes_rebonds):
            cv.arrowedLine(frame_blanc, (liste_premiers_rebonds[i][0],liste_premiers_rebonds[i][1]), 
                            (liste_deuxiemes_rebonds[i][0],liste_deuxiemes_rebonds[i][1]), couleur, 1, tipLength=0.1)

def afficher_detection_orange():
    # Open the first available camera (index 0)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print('Error: Could not read frame.')
            break

        # Convert the frame from BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Define the lower and upper bounds for yellow color in HSV
        lower_yellow = np.array([0, 150, 190])
        upper_yellow = np.array([50, 255, 255])

        # Create a mask for yellow color
        mask = cv.inRange(hsv, lower_yellow, upper_yellow)
        # Bitwise-AND the mask with the original frame to show only the yellow parts
        yellow_objects = cv.bitwise_and(frame, frame, mask=mask)

        # Display the original frame and the yellow objects
        cv.imshow('Original', frame)
        cv.imshow('Yellow Objects', yellow_objects)
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()


def detection_ellipse_couleur(frame,lower_color,upper_color,first_blurred = []):
    """
        Fonction permettant de détecter les ellipse d'une certaine couleur sur une image
        Entrée:
                - L'image
                - L'échelle de couleur minimum en hsv (ex: lower_orange = np.array([0, 150, 190])) 
                - L'échelle de couleur maximum en hsv (ex: upper_orange = np.array([25, 255, 255])) 
                - Une image nuance de gris pour soustraction de fond 
        Sortie:
                - Trace sur l'image
    """
    
    # Convert the frame from BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Create a mask for yellow color
    mask = cv.inRange(hsv, lower_color, upper_color)
    # Bitwise-AND the mask with the original frame to show only the yellow parts
    yellow_objects = cv.bitwise_and(frame, frame, mask=mask)
    gray = cv.cvtColor(yellow_objects, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv.threshold(blurred, 30, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 5:  # La fonction cv2.fitEllipse() nécessite au moins 5 points
            ellipse = cv.fitEllipse(contour)
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                cv.ellipse(yellow_objects, ellipse, (0, 255, 0), 2)


    """gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    if len(first_blurred) > 0:
        difference = cv.absdiff(first_blurred, blurred)
        _, thresh = cv.threshold(difference, 30, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        _, thresh = cv.threshold(blurred, 30, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 5:  # La fonction cv2.fitEllipse() nécessite au moins 5 points
            ellipse = cv.fitEllipse(contour)
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv.ellipse(mask, ellipse, (255, 255, 255), thickness=-1)
                masked_region = cv.bitwise_and(frame, mask)
                hsv_image = cv.cvtColor(masked_region, cv.COLOR_BGR2HSV)
                orange_mask = cv.inRange(hsv_image, lower_color, upper_color)
                if cv.countNonZero(orange_mask) > 10: #modifier pour comparer un pourcentage par rapport à l'ellipse totale
                    cv.ellipse(frame, ellipse, (0, 255, 0), 2)"""
    return(yellow_objects)



def appliquer_pose_estimation(frame):
    """
    Fonction qui applique l'estimation de pose de YOLOv8 à une frame et renvoie une image annotée.
    
    :param frame: Image (sous forme de tableau numpy) sur laquelle appliquer l'estimation de pose.
    #:return: Image annotée avec les poses détectées.
    :return: le resultat.
    """
    # Effectuer une prédiction d'estimation de pose sur la frame
    results = model(frame)
    
    # Annoter la frame avec les poses détectées
    #annotated_frame = results[0].plot()  # pour afficher sur l'image

    return results[0]


def check_sequence(sequence):
    taille_sequence = len(sequence)
    sequence_ref = "ARRB"
    if sequence == sequence_ref:
        print("sequence valide")
    elif sequence == sequence_ref[0:taille_sequence]:
        print("Sequence incomplète")
    else:
        print("Sequence incorrect")
    

def coup_droit_rever(image):
    """
        Fonction permettant de dire sur une image si c'est un coup droit ou un revers en utilisant l'estimation de pose de YOLO
        Entrée:
                - image
        Sortie:
                - "R" ou "C" (revers ou coup droit)
    """

    results = model(image)


    annotated_frame = results[0].plot()
    print(results)
    plt.imshow(annotated_frame)
    plt.axis('off')  # Masquer les axes
    plt.show()


if __name__ == "__main__":
    coup_droit_rever("C:/Users/ReViVD/Documents/GitHub/table-tennis-trainer/test_coup_droit.jpg")
    #liste des rebonds: [18,50,82]
    afficher_detection_orange()
    im = tracer_image_niveau_gris_uniforme((200,200,3),255)
    cv.imwrite('avant.jpg', im)
    tracer_rebonds_fleche_sur_table(im, [[0,0],[50,50],[100,100]], [[100,100],[150,150],[190,190]], (0,0,0))
    cv.imwrite('apres.jpg', im)
    print("rien")