from tkinter import * 
from tkinter.messagebox import *
from tkinter import ttk
from PIL import Image, ImageTk
import urllib.request
import numpy as np
from tkinter import filedialog as fd
from tkinter.font import BOLD, Font
import sys
import cv2 as cv
import os
import time
import json
import csv
import asyncio
import threading
import pygame
import platform
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from bleak import BleakClient, BleakScanner

sys.path.append('./')
from utils_interface import *

# Initialisation de pygame pour la lecture du son
pygame.mixer.init()
sound = pygame.mixer.Sound("sound/rebond_beep-07a.wav")

# Adresse du capteur BLE et caractéristiques
address = "D4:22:CD:00:9B:E6" #"D4:22:CD:00:A0:FD"#"D4:22:CD:00:9B:E6"  # Remplacez par l'adresse de votre capteur
short_payload_characteristic_uuid = "15172004-4947-11e9-8646-d663bd873d93"
measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"

payload_modes = {
    "Free acceleration": [6, b"\x06"],  # Récupérer les mesures d'accélération libre
}

version = "v1_2024.10.11"

class App:
    def __init__(self, window, window_title, video_source=[0], resolution=(640,480), fps=25, nom_match='', video=[os.path.join('samples/balle_aruco.mp4')]): 
        self.window = window
        self.h = 480
        self.w = 640
        self.fps = fps
        self.play = False
        self.police = Font(self.window, size=10, weight=BOLD)
        self.video_source = video_source
        self.vid = []
        self.num_webcam = video_source[0]
        self.resolution = resolution
        self.sauver = 0
        self.num_frame = 0
        self.num_frame_general = 0
        self.liste_evenements = [['Frame','Evenement']]
        self.police = Font(self.window, size=10, weight=BOLD)
        self.horaire_video = time.strftime("%d-%m-%Y-%H-%M-%S")
        self.liste_frame_premier_rebond = []
        self.index_frame_premier_rebond = 0
        self.liste_frame_deuxieme_rebond = []
        self.index_frame_deuxieme_rebond = 0
        self.liste_frame_ref = []
        self.seuil_blanc = 200
        self.backSub = cv.createBackgroundSubtractorMOG2()
        self.faire_from_above = 0
        self.liste_annotation_clic = []
        self.liste_premiers_rebonds = []#[[80,330],[60,340],[70,320],[80+160,350],[60+160,360],[70+160,370],[80,330],[60,340],[70,320],[80+160,350],[60+160,360],[70+160,370],[80,330],[60,340],[70,320],[80+160,350],[60+160,360],[70+160,370]]
        self.liste_deuxiemes_rebonds = []#[(1, 2), (2, 3), (3, 1), (8, 8), (9, 10), (10, 8), (10, 9), (25, 30), (24, 29), (26, 28)]

        self.lower_orange = np.array([0, 150, 190])
        self.upper_orange = np.array([25, 255, 255])
        
        
        self.kernel_erode = np.ones((1, 1), np.uint8)
        self.kernel_dilate = np.ones((5, 5), np.uint8)

        self.val_premier_rebond = []
        self.val_deuxieme_rebond = []

        self.faire_yolo_pose = 0
        self.afficher_from_above = 0
        self.faire_detection_orange = 0
        self.faire_detection_orange_et_ellipse = 0
        self.soustraction = 0

        self.ecart_rebonds = 0
        self.distance_entre_rebonds = 0

        self.num_frame_premier_rebond = 0
        self.num_frame_deuxieme_rebond = 0

        self.val_thresh = 30
        ########################################
        ###### FIn initialisation valeurs ######
        ########################################



        # Initialisation de la vidéo
        if len(video) != 2:
            for i in range(len(video_source)):
                self.vid.append(MyVideoCapture(self.video_source[i]))
        else:
            self.vid.append(MyVideoCapture_fichier_video(video[0]))
            self.vid.append(MyVideoCapture_fichier_video(video[1]))
        
        ret = True
        frame = np.zeros((self.w, self.h, 3), np.uint8)
        self.height, self.width, _ = frame.shape
        if ret:
            self.photo2 = cv.resize(frame, dsize=(self.w, self.h), interpolation=cv.INTER_CUBIC)
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2, cv.COLOR_BGR2RGB)))
            self.label_image = Label(self.window, image=self.photo)
            self.label_image.grid(row=0, column=0, columnspan=40, rowspan=40, sticky=NW)
            self.label_image.bind('<Button-1>',self.clic_gauche_choix_couleur)
            self.label_image.bind('<Button-3>',self.clic_droit_annotation_table)
        

        frame_blanc = tracer_table_vide((276,154,3))

        self.photo2_blanc = cv.resize(frame_blanc, dsize=(self.w//2,self.h), interpolation=cv.INTER_CUBIC)
        self.photo_blanc = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_blanc, cv.COLOR_BGR2RGB)))
        self.label_image_blanc = Label(self.window, image = self.photo_blanc)
        self.label_image_blanc.grid(row = 0, column = 40, columnspan = 40, rowspan = 40, sticky=NW)


        # Photos rebonds
        frame_deuxieme_rebond = tracer_image_niveau_gris_uniforme((150,180,3),168)
        self.photo2_deuxieme_rebond = cv.resize(frame_deuxieme_rebond, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_deuxieme_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond = Label(self.window, image = self.photo_deuxieme_rebond)
        self.label_image_deuxieme_rebond.grid(row = 0, column = 80, columnspan = 40, rowspan = 20, sticky=NW)



        frame_premier_rebond = tracer_image_niveau_gris_uniforme((150,180,3),1)
        self.photo2_premier_rebond = cv.resize(frame_premier_rebond, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_premier_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond = Label(self.window, image = self.photo_premier_rebond)
        self.label_image_premier_rebond.grid(row = 20, column = 80, columnspan = 40, rowspan = 20, sticky=NW)


        # Photos soustractions
        frame_deuxieme_rebond_sous = tracer_image_niveau_gris_uniforme((150,180,3),0)
        self.photo2_deuxieme_rebond_sous = cv.resize(frame_deuxieme_rebond_sous, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_deuxieme_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond_sous = Label(self.window, image = self.photo_deuxieme_rebond_sous)
        self.label_image_deuxieme_rebond_sous.grid(row = 0, column = 120, columnspan = 40, rowspan = 20, sticky=NW)

        frame_premier_rebond_sous = tracer_image_niveau_gris_uniforme((150,180,3),0)
        self.photo2_premier_rebond_sous = cv.resize(frame_premier_rebond_sous, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_premier_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond_sous = Label(self.window, image = self.photo_premier_rebond_sous)
        self.label_image_premier_rebond_sous.grid(row = 20, column = 120, columnspan = 40, rowspan = 20, sticky=NW)



        #########################
        ####### Positions #######
        #########################

        row_choix_webcam = 41
        column_choix_webcam = 0
        row_vue = 47
        column_vue = 0
        row_seuils = 41
        column_seuils = 1
        row_options = 41
        column_options = 20
        row_ajustement_rebonds = 41
        column_ajustement_rebonds = 41
        row_simulation = 46
        column_simulation = 41
        row_statistique = 41
        column_statistique = 81




        # 1. Choix webcams
        self.label_choix_webcam = Label(self.window, text = "1. Choix webcam", font=self.police)
        self.label_choix_webcam.grid(row = row_choix_webcam, column = column_choix_webcam, sticky = W, pady = 0,columnspan = 10)

        # Liste des webcams
        self.liste_webcams = []
        for i in self.video_source:
            self.liste_webcams.append('webcam '+str(i))
        self.combobox = ttk.Combobox(window, values = self.liste_webcams)
        self.combobox.bind('<<ComboboxSelected>>', self.change_webcam_liste_box)
        self.combobox.current(0)
        self.combobox.grid(row = row_choix_webcam+1, column = column_choix_webcam, sticky = NW)
        
        # Perspective
        self.label_perspective = Label(self.window, text = "Perspective:")
        self.label_perspective.grid(row = row_choix_webcam+2, column = column_choix_webcam,columnspan = 15, sticky = W, pady = 0)

        self.liste_perspective = []
        for pers in os.listdir(os.path.join("fds","perspective")):
            self.liste_perspective.append(pers)
        self.combobox_perspective = ttk.Combobox(window, values = self.liste_perspective)
        self.combobox_perspective.bind('<<ComboboxSelected>>', self.change_perspective_liste_box)
        self.combobox_perspective.current(0)
        self.combobox_perspective.grid(row = row_choix_webcam+3, column = column_choix_webcam, sticky = NW)


        
        self.label_capteur = Label(self.window, text = "Perspective:")
        self.label_capteur.grid(row = row_choix_webcam+4, column = column_choix_webcam,columnspan = 15, sticky = W, pady = 0)

        self.liste_capteur = ["D4:22:CD:00:9B:E6","D4:22:CD:00:A0:FD","D4:22:CD:00:9E:2F","D4:22:CD:00:A0:66","D4:22:CD:00:A0:D2"]
        self.combobox_capteur = ttk.Combobox(window, values = self.liste_capteur)
        self.combobox_capteur.bind('<<ComboboxSelected>>', self.change_capteur_liste_box)
        self.combobox_capteur.current(0)
        self.combobox_capteur.grid(row = row_choix_webcam+5, column = column_choix_webcam, sticky = NW)

        
        
        # 2. Choix de la vue
        self.label_choix_vue = Label(self.window, text = "2. Choix de la vue", font=self.police)
        self.label_choix_vue.grid(row = row_vue, column = column_vue, sticky = W, pady = 0,columnspan = 10)
        
        self.checkbox_perspective = Checkbutton(window, text="Faire vue dessus", command=self.changer_valeur_faire_from_above)
        self.checkbox_perspective.grid(row = row_vue+1, column = column_vue,columnspan = 10, sticky = W)
        self.checkbox_yolo_pose = Checkbutton(window, text="Faire Yolo pose", command=self.changer_valeur_faire_yolo_pose)
        self.checkbox_yolo_pose.grid(row = row_vue+2, column = column_vue,columnspan = 10, sticky = W)
        self.checkbox_afficher_from_above = Checkbutton(window, text="Afficher vue dessus", command=self.changer_valeur_afficher_from_above)
        self.checkbox_afficher_from_above.grid(row = row_vue+3, column = column_vue,columnspan = 10, sticky = W)
        self.checkbox_detection_orange = Checkbutton(window, text="Faire détection orange", command=self.changer_valeur_faire_detection_orange)
        self.checkbox_detection_orange.grid(row = row_vue+4, column = column_vue,columnspan = 10, sticky = W)
        #self.checkbox_detection_orange_et_ellipse = Checkbutton(window, text="Faire détection orange et ellipse", command=self.changer_valeur_faire_detection_orange_et_ellipse)
        #self.checkbox_detection_orange_et_ellipse.grid(row = 46, column = 1,columnspan = 10, sticky = W)
        self.checkbox_detection_orange_et_ellipse = Checkbutton(window, text="Faire soustraction", command=self.changer_valeur_faire_soustraction)
        self.checkbox_detection_orange_et_ellipse.grid(row = row_vue+5, column = column_vue,columnspan = 10, sticky = W)
        


        
        # 3. Choix seuils
        self.label_choix_seuil = Label(self.window, text = "2. Choix seuils", font=self.police)
        self.label_choix_seuil.grid(row = row_seuils, column = column_seuils, sticky = W, pady = 0,columnspan = 10)

        # Seuil pour les capteurs
        self.label_seuil = Label(self.window, text = "Seuil:")
        self.label_seuil.grid(row = row_seuils+1, column = column_seuils,columnspan = 15, sticky = W, pady = 0)
        self.sliding_seuil = Scale(self.window, from_=0.1, to=5, resolution=0.1, showvalue=1, length=150, orient=HORIZONTAL, command=self.changer_seuil)
        self.sliding_seuil.set(1)
        self.sliding_seuil.grid(row = row_seuils+2, column = column_seuils,columnspan = 10, sticky = W)
        
        self.label_min_temps = Label(self.window, text = "Min temps:")
        self.label_min_temps.grid(row = row_seuils+3, column = column_seuils,columnspan = 15, sticky = W, pady = 0)
        self.sliding_min_temps = Scale(self.window, from_=0, to=2, resolution=0.1, showvalue=1, length=150, orient=HORIZONTAL, command=self.changer_min_temps)
        self.sliding_min_temps.set(0.1)
        self.sliding_min_temps.grid(row = row_seuils+4, column = column_seuils,columnspan = 10, sticky = W)

        self.label_color_lower = Label(self.window, text = "Couleur lower:")
        self.label_color_lower.grid(row = row_seuils+5, column = column_seuils,columnspan = 15, sticky = W, pady = 0)
        self.sliding_color_lower_h = Scale(self.window, from_=0, to=179, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_lower)
        self.sliding_color_lower_h.set(0)
        self.sliding_color_lower_h.grid(row = row_seuils+6, column = column_seuils,columnspan = 10, sticky = W)
        self.sliding_color_lower_s = Scale(self.window, from_=0, to=255, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_lower)
        self.sliding_color_lower_s.set(150)
        self.sliding_color_lower_s.grid(row = row_seuils+7, column = column_seuils,columnspan = 10, sticky = W)
        self.sliding_color_lower_v = Scale(self.window, from_=0, to=255, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_lower)
        self.sliding_color_lower_v.set(190)
        self.sliding_color_lower_v.grid(row = row_seuils+8, column = column_seuils,columnspan = 10, sticky = W)

        
        self.label_color_upper = Label(self.window, text = "Couleur upper:")
        self.label_color_upper.grid(row = row_seuils+9, column = column_seuils,columnspan = 15, sticky = W, pady = 0)
        self.sliding_color_upper_h = Scale(self.window, from_=0, to=179, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_upper)
        self.sliding_color_upper_h.set(25)
        self.sliding_color_upper_h.grid(row = row_seuils+10, column = column_seuils,columnspan = 10, sticky = W)
        self.sliding_color_upper_s = Scale(self.window, from_=0, to=255, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_upper)
        self.sliding_color_upper_s.set(255)
        self.sliding_color_upper_s.grid(row = row_seuils+11, column = column_seuils,columnspan = 10, sticky = W)
        self.sliding_color_upper_v = Scale(self.window, from_=0, to=255, resolution=1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_color_upper)
        self.sliding_color_upper_v.set(255)
        self.sliding_color_upper_v.grid(row = row_seuils+12, column = column_seuils,columnspan = 10, sticky = W)

        
        self.label_seuil = Label(self.window, text = "Seuil tresh:")
        self.label_seuil.grid(row = row_seuils+13, column = column_seuils,columnspan = 15, sticky = W, pady = 0)
        self.sliding_seuil = Scale(self.window, from_=0, to=255, resolution=0.1, showvalue=0, length=150, orient=HORIZONTAL, command=self.changer_seuil_tresh)
        self.sliding_seuil.set(30)
        self.sliding_seuil.grid(row = row_seuils+14, column = column_seuils,columnspan = 10, sticky = W)

        
        #self.label_seuil_blanc = Label(self.window, text = "Seuil blanc:")
        #self.label_seuil_blanc.grid(row = 50, column = 0,columnspan = 15, sticky = W, pady = 0)
        #self.sliding_seuil_blanc = Scale(self.window, from_=0, to=255, resolution=1, showvalue=1, length=150, orient=HORIZONTAL, command=self.changer_seuil_blanc)
        #self.sliding_seuil_blanc.set(self.seuil_blanc)
        #self.sliding_seuil_blanc.grid(row = 46, column = 1,columnspan = 10, sticky = W)


        # 4. Options
        self.label_options = Label(self.window, text = "4. Options", font=self.police)
        self.label_options.grid(row = row_options, column = column_options, sticky = W, pady = 0,columnspan = 10)
    
        self.bouton_reinitialiser_table = Button(self.window, text="Réinitialiser", width=30, command=self.reinitialiser_table)
        self.bouton_reinitialiser_table.grid(row=row_options+1, column=column_options, sticky=NW)
        self.btn_save = Button(self.window, text='Réinitialiser soustraction', width=30, relief=RAISED, command=self.reinitialiser_soustraction)
        self.btn_save.grid(row=row_options+2, column=column_options, sticky=NW)
        self.btn_save = Button(self.window, text='Enregistrement', width=30, relief=RAISED, command=self.save)
        self.btn_save.grid(row=row_options+3, column=column_options, sticky=NW)
        
        # 5. Ajustement rebonds
        self.label_ajustement_rebonds = Label(self.window, text = "5. Ajustement rebonds", font=self.police)
        self.label_ajustement_rebonds.grid(row = row_ajustement_rebonds, column = column_ajustement_rebonds, sticky = W, pady = 0,columnspan = 10)
        
        # Déplacement dans images rebonds
        self.btn_plus_1_premier_rebond = Button(self.window, text='+1 Rebond 1', width=15, relief=RAISED, command=self.plus_1_premier_rebond)
        self.btn_plus_1_premier_rebond.grid(row=row_ajustement_rebonds+1, column=column_ajustement_rebonds, sticky=NW)
        self.btn_plus_1_deuxieme_rebond = Button(self.window, text='+1 Rebond 2', width=15, relief=RAISED, command=self.plus_1_deuxieme_rebond)
        self.btn_plus_1_deuxieme_rebond.grid(row=row_ajustement_rebonds+1, column=column_ajustement_rebonds+1, sticky=NW)
        
        self.btn_moins_1_premier_rebond = Button(self.window, text='-1 Rebond 1', width=15, relief=RAISED, command=self.moins_1_premier_rebond)
        self.btn_moins_1_premier_rebond.grid(row=row_ajustement_rebonds+2, column=column_ajustement_rebonds, sticky=NW)
        self.btn_moins_1_deuxieme_rebond = Button(self.window, text='-1 Rebond 2', width=15, relief=RAISED, command=self.moins_1_deuxieme_rebond)
        self.btn_moins_1_deuxieme_rebond.grid(row=row_ajustement_rebonds+2, column=column_ajustement_rebonds+1, sticky=NW)
        
        self.btn_enregistrer_valeurs = Button(self.window, text='Sauv rebonds', width=30, relief=RAISED, command=self.save_rebonds)
        self.btn_enregistrer_valeurs.grid(row=row_ajustement_rebonds+3, column=column_ajustement_rebonds, columnspan = 20, sticky=NW)
        self.btn_faire_kmeans = Button(self.window, text='Kmeans', width=30, relief=RAISED, command=self.faire_kmeans)
        self.btn_faire_kmeans.grid(row=row_ajustement_rebonds+4, column=column_ajustement_rebonds, columnspan = 20, sticky=NW)


        # 6. Simulation
        self.label_simulation = Label(self.window, text = "6. Simulation", font=self.police)
        self.label_simulation.grid(row = row_simulation, column = column_simulation, sticky = W, pady = 0,columnspan = 10)

        # Interface graphique
        self.bouton_bouton_1 = Button(self.window, text="PremierR", width=30, command=self.bouton_1)
        self.bouton_bouton_1.grid(row=row_simulation+1, column=column_simulation, columnspan = 20, sticky=NW)
        self.bouton_bouton_2 = Button(self.window, text="DeuxiemeR", width=30, command=self.bouton_2)
        self.bouton_bouton_2.grid(row=row_simulation+2, column=column_simulation, columnspan = 20, sticky=NW)

        
        # 7. Statistiques
        self.label_statistiques = Label(self.window, text = "7. Statistiques", font=self.police)
        self.label_statistiques.grid(row = row_statistique, column = column_statistique, sticky = W, pady = 0,columnspan = 10)
        
        self.label_fps_calcule = Label(self.window, text = "Fps moments rebonds: ")
        self.label_fps_calcule.grid(row = row_statistique+1, column = column_statistique, sticky = W, pady = 0,columnspan = 40)

        self.label_ecart_duree_rebond = Label(self.window, text = "Nb frame écart entre rebonds: ")
        self.label_ecart_duree_rebond.grid(row = row_statistique+2, column = column_statistique, sticky = W, pady = 0,columnspan = 40)
        
        self.label_ecart_distance_rebond = Label(self.window, text = "Ecart distance entre rebonds: ")
        self.label_ecart_distance_rebond.grid(row = row_statistique+3, column = column_statistique, sticky = W, pady = 0,columnspan = 40)

        self.label_nb_clusters = Label(self.window, text = "Nb clusters trouvés: 0")
        self.label_nb_clusters.grid(row = row_statistique+4, column = column_statistique, sticky = W, pady = 0,columnspan = 40)


        


        px,py = [],[]
        liste_points_perspective = recuperation_points_table(os.path.join("fds","perspective",self.combobox_perspective.get()))
        for point in liste_points_perspective:
            px.append(int(point[0]))#*self.w/self.width))
            py.append(int(point[1]))#*self.h/self.height))

        
        #self.scr_pct1 = json_course['calibration']['srcPct1']
        pts_src = np.float32([[px[0],py[0]],[px[1],py[1]],[px[2],py[2]],[px[3],py[3]]]).reshape(-1,1,2)
        pts_dst = np.float32([[640,0],[640,480],[0,480],[0,0]]).reshape(-1,1,2)
        
        #pts_dst = np.float32([[0,0],[self.width,0],[self.width,self.height],[0,self.height]]).reshape(-1,1,2)
        self.H, status = cv.findHomography(pts_src, pts_dst)# = cv.getPerspectiveTransform(pts_src, pts_dst)
        

        



        # Variables pour le traitement des données du capteur
        self.xs = []
        self.ys_x = []
        self.ys_y = []
        self.ys_z = []
        self.deriv_x = []
        self.deriv_y = []
        self.deriv_z = []
        self.last_sound_time = 0
        self.t0 = None
        self.threshold = 1.0  # Seuil par défaut
        self.min_time = 0.7   # Temps minimum entre deux sons
        self.premier_clic = 0


        #Raccourcis clavier
        # Chargement du son
        self.sound = sound  # Utilise l'objet son global

        # Démarrage du client BLE dans un thread séparé
        self.ble_thread = threading.Thread(target=self.run_ble_client)
        self.ble_thread.daemon = True
        self.ble_thread.start()

        self.delay = int(1000 / self.fps)
        self.temps1 = time.time()
        self.tracer_table_rebond(False)
        self.update()
        self.window.mainloop()

    def clic_droit_annotation_table(self,event):
        #print(len(self.liste_annotation_clic))
        self.liste_annotation_clic.append([event.x,event.y])
        if len(self.liste_annotation_clic) == 4:
            #print('oui')
            pts_src = np.float32([[self.liste_annotation_clic[0][0],self.liste_annotation_clic[0][1]],
                                  [self.liste_annotation_clic[1][0],self.liste_annotation_clic[1][1]],
                                  [self.liste_annotation_clic[2][0],self.liste_annotation_clic[2][1]],
                                  [self.liste_annotation_clic[3][0],self.liste_annotation_clic[3][1]]]).reshape(-1,1,2)
            pts_dst = np.float32([[640,0],[640,480],[0,480],[0,0]]).reshape(-1,1,2)
            
            #pts_dst = np.float32([[0,0],[self.width,0],[self.width,self.height],[0,self.height]]).reshape(-1,1,2)
            self.H, status = cv.findHomography(pts_src, pts_dst)

            self.liste_annotation_clic = []

    def clic_gauche_choix_couleur(self,event):
        event.x
        event.y
        print([event.y, event.x])
        bgr_color = self.frame[event.y, event.x]

        # Convertir la couleur BGR en HSV
        hsv_color = cv.cvtColor(np.uint8([[bgr_color]]), cv.COLOR_BGR2HSV)[0][0]

        # Définir une tolérance autour de la couleur cliquée
        tolerance = np.array([10, 50, 50])

        # Calculer les valeurs minimales et maximales en HSV
        self.lower_orange = np.maximum(hsv_color - tolerance, [0, 50, 50])
        self.upper_orange = np.minimum(hsv_color + tolerance, [179, 255, 255])

        self.sliding_color_lower_h.set(self.lower_orange[0])
        self.sliding_color_lower_s.set(self.lower_orange[1])
        self.sliding_color_lower_v.set(self.lower_orange[2])
        self.sliding_color_upper_h.set(self.upper_orange[0])
        self.sliding_color_upper_s.set(self.upper_orange[1])
        self.sliding_color_upper_v.set(self.upper_orange[2])



    def changer_color_lower(self,val):
        self.lower_orange = np.array([int(self.sliding_color_lower_h.get()), int(self.sliding_color_lower_s.get()), int(self.sliding_color_lower_v.get())])

    
    def changer_color_upper(self,val):
        self.upper_orange = np.array([int(self.sliding_color_upper_h.get()), int(self.sliding_color_upper_s.get()), int(self.sliding_color_upper_v.get())])

    def modif_stats(self):
        self.label_fps_calcule.config(text="Fps moments rebonds: " + str(1/self.fps_calcule))
        self.label_ecart_duree_rebond.config(text="Nb frame écart entre rebonds: " + str(self.num_frame_deuxieme_rebond-self.num_frame_premier_rebond))
        if len(self.val_premier_rebond) > 0 and len(self.val_deuxieme_rebond) > 0:
            self.distance_entre_rebonds = np.sqrt((self.val_premier_rebond[0]-self.val_deuxieme_rebond[0])**2 + (self.val_premier_rebond[1]-self.val_deuxieme_rebond[1])**2)
        self.label_ecart_distance_rebond.config(text="Ecart distance entre rebonds: "+ str(self.distance_entre_rebonds))

            
    def get_color_from_digit(self,digit):
        """
        Associe un entier compris entre 0 et 9 à une couleur unique.

        :param digit: Entier (0-9) pour lequel on veut obtenir une couleur.
        :return: Couleur sous forme de tuple (R, G, B).
        """
        # Liste de 10 couleurs uniques (R, G, B)
        colors = [
            (255, 0, 0),      # Rouge pour 0
            (0, 255, 0),      # Vert pour 1
            (0, 0, 255),      # Bleu pour 2
            (255, 255, 0),    # Jaune pour 3
            (255, 165, 0),    # Orange pour 4
            (128, 0, 128),    # Violet pour 5
            (0, 255, 255),    # Cyan pour 6
            (255, 192, 203),  # Rose pour 7
            (128, 128, 128),  # Gris pour 8
            (0, 0, 0)         # Noir pour 9
        ]
        
        # Vérifier si le chiffre est entre 0 et 9
        if 0 <= digit <= 9:
            return colors[digit]
        else:
            raise ValueError("Le chiffre doit être compris entre 0 et 9.")
        

    def faire_kmeans(self):
        self.tracer_table_rebond(True)

    def tracer_table_rebond(self, cluster=False):
        frame_blanc = tracer_table_vide((276,154,3))
        tracer_rebonds_fleche_sur_table(frame_blanc,self.liste_premiers_rebonds,self.liste_deuxiemes_rebonds,(0,0,0)) 
        
        if len(self.val_premier_rebond) != 0:
            cv.circle(frame_blanc, self.val_premier_rebond, 4, (0, 0, 255), 1)
        if len(self.val_deuxieme_rebond) != 0:
            cv.circle(frame_blanc, self.val_deuxieme_rebond, 4, (0, 0, 255), -1)
            if len(self.val_premier_rebond) != 0:
                cv.arrowedLine(frame_blanc, (self.val_premier_rebond), 
                            (self.val_deuxieme_rebond), (0, 0, 255), 1, tipLength=0.1)
        if cluster:
            self.elbow_point = calcul_nb_cluster_optimal(self.liste_deuxiemes_rebonds)
            kmeans_serveur = KMeans(n_clusters = min(self.elbow_point,9))
            kmeans_serveur.fit(self.liste_deuxiemes_rebonds)
            labels = kmeans_serveur.labels_
            self.label_ecart_distance_rebond.config(text="Ecart distance entre rebonds: "+ str(len(labels)))
            for i in range(len(self.liste_deuxiemes_rebonds)):
                cv.circle(frame_blanc, self.liste_deuxiemes_rebonds[i], 4, self.get_color_from_digit(labels[i]), -1)
        else:
            if len(self.liste_premiers_rebonds) > 0:
                cv.circle(frame_blanc,  (self.liste_premiers_rebonds[-1][0],self.liste_premiers_rebonds[-1][1]), 4, (0, 255, 0), 1)
                cv.circle(frame_blanc,  (self.liste_deuxiemes_rebonds[-1][0],self.liste_deuxiemes_rebonds[-1][1]), 4, (0, 255, 0), 1)
                cv.arrowedLine(frame_blanc, (self.liste_premiers_rebonds[-1][0],self.liste_premiers_rebonds[-1][1]), 
                                (self.liste_deuxiemes_rebonds[-1][0],self.liste_deuxiemes_rebonds[-1][1]), (0, 0, 0), 1, tipLength=0.1)


        self.frame_blanc_avec_premier_rebond = frame_blanc.copy()
        self.photo2_blanc = cv.resize(frame_blanc, dsize=(self.w//2,self.h), interpolation=cv.INTER_CUBIC)

        self.photo_blanc = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_blanc, cv.COLOR_BGR2RGB)))
        self.label_image_blanc.configure(image=self.photo_blanc)
        self.label_image_blanc.image = self.photo

    def save_rebonds(self):
        if len(self.val_premier_rebond) > 0 and len(self.val_deuxieme_rebond) > 0:
            self.liste_premiers_rebonds.append(self.val_premier_rebond)
            self.liste_deuxiemes_rebonds.append(self.val_deuxieme_rebond)
        
        self.val_premier_rebond = []
        self.val_deuxieme_rebond = []

    def maj_sous(self):
        if self.num_frame_general > 10:
            self.tracer_table_rebond()
            print("maj")
            self.bouton_2(self.liste_frame_deuxieme_rebond[self.index_frame_deuxieme_rebond])
            self.bouton_1(self.liste_frame_premier_rebond[self.index_frame_premier_rebond])
            self.modif_stats()


    def moins_1_deuxieme_rebond(self):
        if self.index_frame_deuxieme_rebond != 0:
            self.num_frame_deuxieme_rebond -= 1
        self.index_frame_deuxieme_rebond = max(self.index_frame_deuxieme_rebond-1,0)
        self.tracer_table_rebond()
        print(self.index_frame_deuxieme_rebond)
        self.bouton_2(self.liste_frame_deuxieme_rebond[self.index_frame_deuxieme_rebond])
        self.modif_stats()
        print("moins_1_deuxieme_rebond")
        
    def moins_1_premier_rebond(self):
        if self.index_frame_premier_rebond != 0:
            self.num_frame_premier_rebond -= 1
        self.index_frame_premier_rebond = max(self.index_frame_premier_rebond-1,0)
        self.tracer_table_rebond()
        print(self.index_frame_deuxieme_rebond)
        self.bouton_1(self.liste_frame_premier_rebond[self.index_frame_premier_rebond])
        self.modif_stats()
        print("moins_1_premier_rebond")    

    def plus_1_deuxieme_rebond(self):
        if min(self.index_frame_deuxieme_rebond+1,len(self.liste_frame_deuxieme_rebond)) < len(self.liste_frame_deuxieme_rebond):
            self.num_frame_deuxieme_rebond += 1
            self.index_frame_deuxieme_rebond = min(self.index_frame_deuxieme_rebond+1,len(self.liste_frame_deuxieme_rebond))
        self.tracer_table_rebond()
        print(self.index_frame_deuxieme_rebond)
        self.bouton_2(self.liste_frame_deuxieme_rebond[self.index_frame_deuxieme_rebond])
        self.modif_stats()
        print("plus_1_deuxieme_rebond")
        
    def plus_1_premier_rebond(self):
        if min(self.index_frame_premier_rebond+1,len(self.liste_frame_premier_rebond)) < len(self.liste_frame_premier_rebond):
            self.num_frame_premier_rebond += 1
            self.index_frame_premier_rebond = min(self.index_frame_premier_rebond+1,len(self.liste_frame_deuxieme_rebond))
        self.tracer_table_rebond()
        print(len(self.liste_frame_premier_rebond))
        print(self.index_frame_deuxieme_rebond)
        self.bouton_1(self.liste_frame_premier_rebond[self.index_frame_premier_rebond])
        self.modif_stats()
        print("plus_1_premier_rebond")

    
    def changer_valeur_afficher_from_above(self):
        if self.afficher_from_above == 1:
            self.afficher_from_above = 0
        else:
            self.afficher_from_above = 1

    def changer_valeur_faire_soustraction(self):
        if self.soustraction == 1:
            self.soustraction = 0
        else:
            self.soustraction = 1

    def changer_valeur_faire_yolo_pose(self):
        if self.faire_yolo_pose == 1:
            self.faire_yolo_pose = 0
        else:
            self.faire_yolo_pose = 1
            
    def changer_valeur_faire_detection_orange(self):
        if self.faire_detection_orange == 1:
            self.faire_detection_orange = 0
        else:
            self.faire_detection_orange = 1
    def changer_valeur_faire_detection_orange_et_ellipse(self):
        if self.faire_detection_orange_et_ellipse == 1:
            self.faire_detection_orange_et_ellipse = 0
        else:
            self.faire_detection_orange_et_ellipse = 1

    def changer_valeur_faire_from_above(self):
        if self.faire_from_above == 1:
            self.faire_from_above = 0
        else:
            self.faire_from_above = 1
        self.liste_frame_ref = []

    def change_capteur_liste_box(self,e):
        global address
        address = self.combobox_capteur.get()

    def change_perspective_liste_box(self,e):
        px,py = [],[]
        liste_points_perspective = recuperation_points_table(os.path.join("fds","perspective",self.combobox_perspective.get()))
        for point in liste_points_perspective:
            px.append(int(point[0]))#*self.w/self.width))
            py.append(int(point[1]))#*self.h/self.height))

        
        #self.scr_pct1 = json_course['calibration']['srcPct1']
        pts_src = np.float32([[px[0],py[0]],[px[1],py[1]],[px[2],py[2]],[px[3],py[3]]]).reshape(-1,1,2)
        pts_dst = np.float32([[640,0],[640,480],[0,480],[0,0]]).reshape(-1,1,2)
        #pts_dst = np.float32([[0,0],[self.frame[0],0],[self.frame[0],self.frame[1]],[0,self.frame[1]]]).reshape(-1,1,2)
        #pts_dst = np.float32([[0,0],[self.width,0],[self.width,self.height],[0,self.height]]).reshape(-1,1,2)
        self.H = cv.getPerspectiveTransform(pts_src, pts_dst)
    
    def changer_seuil_blanc(self,val):
        self.seuil_blanc = int(val)

    def changer_seuil_tresh(self,val):
        self.val_thresh = int(float(val))
        self.maj_sous()

    def changer_seuil(self,val):
        self.threshold = float(val)
        
    def changer_min_temps(self,val):
        self.min_time = float(val)

    def ajout_premier_rebond(self,x,y):
        self.tracer_table_rebond()
        self.num_frame_premier_rebond = self.num_frame_general

        cv.imwrite(os.path.join("fds",'tkinter_fds',"Premier_Rebond-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), self.frame)
        self.liste_evenements.append([self.num_frame,"Premier_rebond"])
        with open(os.path.join("fds",'tkinter_fds',"video","annotation-" + self.horaire_video + ".csv"),"w", newline='') as fichier_csv:
            fichier_csv_writer = csv.writer(fichier_csv, delimiter=',')
            for row in self.liste_evenements:
                fichier_csv_writer.writerow(row)
        
    def ajout_deuxieme_rebond(self,x,y):
        self.tracer_table_rebond()
        
        self.num_frame_deuxieme_rebond = self.num_frame_general
        
        self.modif_stats()
        
        self.photo2_blanc = cv.resize(self.frame_blanc_avec_premier_rebond, dsize=(self.w//2,self.h), interpolation=cv.INTER_CUBIC)

        self.photo_blanc = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_blanc, cv.COLOR_BGR2RGB)))
        self.label_image_blanc.configure(image=self.photo_blanc)
        self.label_image_blanc.image = self.photo

        cv.imwrite(os.path.join("fds",'tkinter_fds',"Deuxieme_Rebond-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), self.frame)
        self.liste_evenements.append([self.num_frame,"Deuxieme_rebond"])
        with open(os.path.join("fds",'tkinter_fds',"video","annotation-" + self.horaire_video + ".csv"),"w", newline='') as fichier_csv:
            fichier_csv_writer = csv.writer(fichier_csv, delimiter=',')
            for row in self.liste_evenements:
                fichier_csv_writer.writerow(row)
                
        
        

    def save(self):
        if self.sauver:
            self.sauver = 0
            self.videoWriter.release()
            self.btn_save['text'] = 'enregistrement'
            self.num_frame = 0
            
        else:

            self.sauver = 1
            fourcc = cv.VideoWriter_fourcc('X','V','I','D')
            self.horaire_video = time.strftime("%d-%m-%Y-%H-%M-%S")
            self.videoWriter = cv.VideoWriter(os.path.join("fds",'tkinter_fds','video',"video-" + self.horaire_video + ".avi"), fourcc, 1//self.fps_calcule, (self.frame.shape[1],self.frame.shape[0]))
            
            self.btn_save['text'] = 'stop enregistrement'

    def reinitialiser_soustraction(self):
        self.liste_frame_ref = []


    def reinitialiser_table(self):
        self.premier_clic = 0
        frame_blanc = tracer_table_vide((276,154,3))
        
        self.photo2_blanc = cv.resize(frame_blanc, dsize=(self.w//2,self.h), interpolation=cv.INTER_CUBIC)
        self.photo_blanc = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_blanc, cv.COLOR_BGR2RGB)))
        self.label_image_blanc.configure(image=self.photo_blanc)
        self.label_image_blanc.image = self.photo
        
        frame_deuxieme_rebond = tracer_image_niveau_gris_uniforme((150,180,3),168)
        self.photo2_deuxieme_rebond = cv.resize(frame_deuxieme_rebond, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_deuxieme_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond.configure(image=self.photo_deuxieme_rebond)
        self.label_image_deuxieme_rebond.image = self.photo
        frame_premier_rebond = tracer_image_niveau_gris_uniforme((150,180,3),1)
        self.photo2_premier_rebond = cv.resize(frame_premier_rebond, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_premier_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond.configure(image=self.photo_premier_rebond)
        self.label_image_premier_rebond.image = self.photo

        
        frame_premier_rebond_sous = tracer_image_niveau_gris_uniforme((150,180,3),1)
        self.photo2_premier_rebond_sous = cv.resize(frame_premier_rebond_sous, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_premier_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond_sous.configure(image=self.photo_premier_rebond_sous)
        self.label_image_premier_rebond_sous.image = self.photo
        frame_deuxieme_rebond_sous = tracer_image_niveau_gris_uniforme((150,180,3),1)
        self.photo2_deuxieme_rebond_sous = cv.resize(frame_deuxieme_rebond_sous, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_deuxieme_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond_sous.configure(image=self.photo_deuxieme_rebond_sous)
        self.label_image_deuxieme_rebond_sous.image = self.photo

        self.ecart_rebonds = 0
        self.distance_entre_rebonds = 0

        self.label_ecart_duree_rebond.config(text="Nb frame écart entre rebonds: 0")
        self.label_ecart_distance_rebond.config(text="Ecart distance entre rebonds: 0")

        self.horaire_video = time.strftime("%d-%m-%Y-%H-%M-%S")
        self.tracer_table_rebond()

        print('Réinitialisation')

    
    def change_webcam_liste_box(self,event):
        select = self.combobox.get()
        self.num_webcam = int(select.split(' ')[-1])
        self.liste_frame_ref = []
    
    
    def bouton_1(self,frame=[]):
        print("bouton_1")

        if len(frame) == 0:
            frame = self.liste_frame_premier_rebond[0].copy()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        difference = cv.absdiff(self.first_blurred, blurred)
        _, thresh = cv.threshold(difference, self.val_thresh, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        image_color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        liste_cx,liste_cy = [],[]
        for contour in contours:
            if len(contour) >= 5:  # La fonction cv2.fitEllipse() nécessite au moins 5 points
                ellipse = cv.fitEllipse(contour)
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv.ellipse(mask, ellipse, (255, 255, 255), thickness=-1)
                    masked_region = cv.bitwise_and(frame, mask)
                    cv.imwrite(os.path.join("fds",'tkinter_fds',"Test-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), masked_region)
                    hsv_image = cv.cvtColor(masked_region, cv.COLOR_BGR2HSV)
                    orange_mask = cv.inRange(hsv_image, self.lower_orange, self.upper_orange)
                    #print("autre: ",cv.countNonZero(cv.cvtColor(mask, cv.COLOR_BGR2GRAY)))
                    #print("orange: ",cv.countNonZero(orange_mask))
                    if ((cv.countNonZero(orange_mask)) > 10 and (cv.countNonZero(cv.cvtColor(mask, cv.COLOR_BGR2GRAY)) < cv.countNonZero(orange_mask)*3)): #modifier pour comparer un pourcentage par rapport à l'ellipse totale
                        cv.ellipse(image_color, ellipse, (0, 255, 0), 2)
                        M = cv.moments(contour)
                        if M["m00"] != 0:
                            # Calculer les coordonnées du barycentre (cx, cy)
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            liste_cx.append(cx)
                            liste_cy.append(cy)

        #frame_premier_rebond_sous = cv.absdiff(self.frame, self.im_ref)# self.liste_frame[0])
        self.photo2_premier_rebond_sous = cv.resize(image_color, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_premier_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond_sous.configure(image=self.photo_premier_rebond_sous)
        self.label_image_premier_rebond_sous.image = self.photo_premier_rebond_sous

        if len(liste_cy) > 0:
            cv.circle(self.frame_blanc_avec_premier_rebond, (int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640))), 4, (0, 0, 0), 1)
                
            self.photo2_blanc = cv.resize(self.frame_blanc_avec_premier_rebond, dsize=(self.w//2,self.h), interpolation=cv.INTER_CUBIC)

            self.photo_blanc = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_blanc, cv.COLOR_BGR2RGB)))
            self.label_image_blanc.configure(image=self.photo_blanc)
            self.label_image_blanc.image = self.photo

        if len(frame) == 0:
            self.photo2_premier_rebond = cv.resize(self.liste_frame_premier_rebond[0], dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        else:
            self.photo2_premier_rebond = cv.resize(frame, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)

        self.photo_premier_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_premier_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_premier_rebond.configure(image=self.photo_premier_rebond)
        self.label_image_premier_rebond.image = self.label_image_premier_rebond

        
        #self.ajout_premier_rebond(np.mean(cy),np.mean(cx))
        if len(liste_cx) > 0:
            self.val_premier_rebond = [int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640))]
            self.ajout_premier_rebond(int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640)))
        else:
            self.ajout_premier_rebond(160,240)

        #print(time.time()-self.temps_capteur)
        #print(self.num_frame_general)
        #print("################")
        
    def bouton_2(self,frame=[]):
        print("bouton_2")
        
        if len(frame) == 0:
            frame = self.liste_frame_deuxieme_rebond[0].copy()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        difference = cv.absdiff(self.first_blurred, blurred)
        _, thresh = cv.threshold(difference, self.val_thresh, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        image_color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        liste_cx,liste_cy = [],[]
        for contour in contours:
            if len(contour) >= 5:  # La fonction cv2.fitEllipse() nécessite au moins 5 points
                ellipse = cv.fitEllipse(contour)
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv.ellipse(mask, ellipse, (255, 255, 255), thickness=-1)
                    masked_region = cv.bitwise_and(frame, mask)
                    cv.imwrite(os.path.join("fds",'tkinter_fds',"Test-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), masked_region)
                    hsv_image = cv.cvtColor(masked_region, cv.COLOR_BGR2HSV)
                    orange_mask = cv.inRange(hsv_image, self.lower_orange, self.upper_orange)
                    if ((cv.countNonZero(orange_mask)) > 10 and (cv.countNonZero(cv.cvtColor(mask, cv.COLOR_BGR2GRAY)) < cv.countNonZero(orange_mask)*3)):
                        cv.ellipse(image_color, ellipse, (0, 255, 0), 2)
                        M = cv.moments(contour)
                        if M["m00"] != 0:
                            # Calculer les coordonnées du barycentre (cx, cy)
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            liste_cx.append(cx)
                            liste_cy.append(cy)
        #frame_deuxieme_rebond_sous = cv.absdiff(self.frame, self.im_ref)#self.liste_frame[0])
        self.photo2_deuxieme_rebond_sous = cv.resize(image_color, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        self.photo_deuxieme_rebond_sous = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond_sous, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond_sous.configure(image=self.photo_deuxieme_rebond_sous)
        self.label_image_deuxieme_rebond_sous.image = self.photo_deuxieme_rebond_sous

        
        if len(frame) == 0:
            self.photo2_deuxieme_rebond = cv.resize(self.liste_frame_deuxieme_rebond[0], dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)
        else:
            self.photo2_deuxieme_rebond = cv.resize(frame, dsize=(self.w//2,self.h//2), interpolation=cv.INTER_CUBIC)


        if len(liste_cy) > 0 and len(self.val_premier_rebond) > 0:
            #print(int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640)))
            #print(self.val_premier_rebond)
            cv.circle(self.frame_blanc_avec_premier_rebond, (int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640))), 4, (0, 0, 0), -1)
            cv.arrowedLine(self.frame_blanc_avec_premier_rebond, self.val_premier_rebond, 
                            (int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640))), (0, 0, 0), 1, tipLength=0.1)
            
        self.photo_deuxieme_rebond = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(self.photo2_deuxieme_rebond, cv.COLOR_BGR2RGB)))
        self.label_image_deuxieme_rebond.configure(image=self.photo_deuxieme_rebond)
        self.label_image_deuxieme_rebond.image = self.photo_deuxieme_rebond
        
        
        if len(liste_cx) > 0:
            self.val_deuxieme_rebond = [int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640))]
            self.ajout_deuxieme_rebond(int(np.mean(liste_cy)*152/480),int(274-(np.mean(liste_cx)*274/640)))
        else:
            self.ajout_deuxieme_rebond(160,240)

    def run_ble_client(self):
        asyncio.run(self.ble_main())

    async def ble_main(self):
        device = await BleakScanner.find_device_by_address(address, timeout=20.0)
        if device is None:
            print("Appareil non trouvé.")
            return
        async with BleakClient(device) as client:
            await client.start_notify(
                short_payload_characteristic_uuid, self.handle_short_payload_notification
            )
            payload_mode_values = payload_modes["Free acceleration"]
            payload_mode = payload_mode_values[1]
            measurement_default = b"\x01"
            start_measurement = b"\x01"
            full_turnon_payload = measurement_default + start_measurement + payload_mode
            await client.write_gatt_char(
                measurement_characteristic_uuid, full_turnon_payload, True
            )
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                await client.stop_notify(short_payload_characteristic_uuid)

    def encode_free_accel_bytes_to_string(self, bytes_):
        data_segments = np.dtype(
            [
                ("timestamp", np.uint32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("zero_padding", np.uint32),
            ]
        )
        formatted_data = np.frombuffer(bytes_, dtype=data_segments)
        return formatted_data

    def handle_short_payload_notification(self, sender, data):
        formatted_data = self.encode_free_accel_bytes_to_string(data)
        acceleration_x = formatted_data["x"][0]
        acceleration_y = formatted_data["y"][0]
        acceleration_z = formatted_data["z"][0]

        # Utilisation d'un timer Ã  haute précision
        current_time = time.perf_counter()
        if self.t0 is None:
            self.t0 = current_time  # Initialiser t0 lors de la premiÃ¨re donnée reÃ§ue
        elapsed_time = current_time - self.t0
        self.xs.append(elapsed_time)

        self.ys_x.append(acceleration_x)
        self.ys_y.append(acceleration_y)
        self.ys_z.append(acceleration_z)

        if len(self.xs) > 1:
            delta_t = 0.1  # Vous pouvez ajuster cette valeur
            self.deriv_x.append((self.ys_x[-1] - self.ys_x[-2]) / delta_t)
            self.deriv_y.append((self.ys_y[-1] - self.ys_y[-2]) / delta_t)
            self.deriv_z.append((self.ys_z[-1] - self.ys_z[-2]) / delta_t)
        else:
            self.deriv_x.append(0)
            self.deriv_y.append(0)
            self.deriv_z.append(0)

        # Conserver uniquement les N derniers points de données
        N = 100
        if len(self.xs) > N:
            self.xs = self.xs[-N:]
            self.ys_x = self.ys_x[-N:]
            self.ys_y = self.ys_y[-N:]
            self.ys_z = self.ys_z[-N:]
            self.deriv_x = self.deriv_x[-N:]
            self.deriv_y = self.deriv_y[-N:]
            self.deriv_z = self.deriv_z[-N:]

        # Calculer la somme des valeurs absolues des dérivées
        abs_sum = abs(self.deriv_x[-1]) + abs(self.deriv_y[-1]) + abs(self.deriv_z[-1])

        # Vérifier si le seuil est dépassé et si le temps minimum est écoulé
        self.temps_capteur = time.time()
        if abs_sum > self.threshold and (time.perf_counter() - self.last_sound_time) > self.min_time:
            self.sound.play()
            if self.premier_clic == 0:
                self.liste_frame_premier_rebond = [self.frame.copy()]
                self.bouton_1()
                self.premier_clic = 1
            elif self.premier_clic == 1:
                self.liste_frame_deuxieme_rebond = [self.frame.copy()]
                self.bouton_2()
                self.premier_clic = 2
            self.last_sound_time = time.perf_counter()

    def update(self):
        ret, frame = self.vid[self.num_webcam].get_frame()
        if ret: 
            self.frame_from_above = cv.warpPerspective(cv.cvtColor(frame, cv.COLOR_BGR2RGB), self.H, (self.height,self.width))
            frame = cv.resize(frame, dsize=(self.resolution[0], self.resolution[1]), interpolation=cv.INTER_CUBIC)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            for p in self.liste_annotation_clic:
                cv.circle(frame, p, 1, (0, 255, 0), -1)
            self.frame = frame.copy()
            if self.faire_from_above:
                self.frame = self.frame_from_above.copy()
            temps2 = time.time()
            #if (temps2 - self.temps1) != 0:
            #    cv.putText(frame, str(int(1 / (temps2 - self.temps1))) + ' fps', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            if self.faire_yolo_pose:
                frame = appliquer_pose_estimation(frame)
            if self.afficher_from_above:
                frame = self.frame_from_above.copy()
            
            if self.faire_detection_orange:
                frame = detection_ellipse_couleur(frame,self.lower_orange,self.upper_orange,first_blurred = [])
            elif self.faire_detection_orange_et_ellipse:
                detection_ellipse_couleur(frame,self.lower_orange,self.upper_orange,first_blurred = self.first_blurred)
            
            if self.soustraction:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blurred = cv.GaussianBlur(gray, (5, 5), 0)
                difference = cv.absdiff(self.first_blurred, blurred)
                _, thresh = cv.threshold(difference, self.val_thresh, 255, cv.THRESH_BINARY)
                frame = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)

            self.fps_calcule = temps2 - self.temps1
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
            self.label_image.configure(image=self.photo)
            self.label_image.image = self.photo
            self.num_frame_general += 1
            if self.sauver:
                self.num_frame += 1
                self.videoWriter.write(frame)
            if len(self.liste_frame_premier_rebond) <= 9:
                self.liste_frame_premier_rebond.append(self.frame.copy())
            if len(self.liste_frame_deuxieme_rebond) <= 9:
                self.liste_frame_deuxieme_rebond.append(self.frame.copy())


            if len(self.liste_frame_ref) <= 9:
                self.liste_frame_ref.append(self.frame.copy())
                if len(self.liste_frame_ref) == 10:
                    self.im_ref = self.liste_frame_ref[-1]
                    for i in range(len(self.liste_frame_ref)-1):
                        self.im_ref += self.liste_frame_ref[i]
                    
                    image_avg = self.im_ref/len(self.liste_frame_ref)
                    self.im_ref = cv.convertScaleAbs(image_avg)

                    self.first_gray = cv.cvtColor(self.liste_frame_ref[0], cv.COLOR_BGR2GRAY)

                    self.first_blurred = cv.GaussianBlur(self.first_gray, (5, 5), 0)
                
            self.temps1 = temps2
            self.window.after(self.delay, self.update)
        else:
            print('Fin de la vidéo')

    # Les autres méthodes de la classe App restent inchangées
    # ...

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Ouverture de la source vidéo
        
        if platform.system() == "Windows":
            self.vid = cv.VideoCapture(video_source, cv.CAP_DSHOW)
        else:
            self.vid = cv.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Impossible d'ouvrir la source vidéo", video_source)

        # Récupération de la largeur et de la hauteur de la vidéo
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        ret, frame = self.vid.read()
        self.photo = frame

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Retourne une image convertie en BGR
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
         if self.vid.isOpened():
            self.vid.release()

class MyVideoCapture_fichier_video:
    def __init__(self, video_source):
        # Ouverture de la source vidéo
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Impossible d'ouvrir la source vidéo", video_source)

        # Récupération de la largeur et de la hauteur de la vidéo
        ret, frame = self.vid.read()
        self.photo = frame

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Retourne une image convertie en BGR
                return (ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
         if self.vid.isOpened():
            self.vid.release()





if __name__ == "__main__":
    selecteur = Tk()
    selecteur.geometry("300x300")
    selection = 0
    selection_video = []
    if not os.path.isdir(os.path.join("fds",'tkinter_fds')):
        os.mkdir(os.path.join("fds",'tkinter_fds'))
    if not os.path.isdir(os.path.join("fds",'tkinter_fds',"video")):
        os.mkdir(os.path.join("fds",'tkinter_fds',"video"))

    


    def faire_webcam():
        selecteur.destroy()
        global selection
        selection = 1

    def faire_video():
        btn.destroy()
        btn1.destroy()

        def afficher():
            if ((my_entry1.get() == '') or (my_entry2.get() == '')):
                label1 = Label(selecteur, text="Il faut remplir les deux champs", fg="red")
                label1.pack(pady=10)
            else:
                global selection
                global selection_video
                selection = 2
                selection_video = [my_entry1.get(), my_entry2.get()]
                selecteur.destroy()
        
        bouton = Button(selecteur, text="Valider", command=afficher)
        bouton.pack(padx=50, pady=10)

        label_video1 = Label(selecteur, text="Vidéo 1")
        label_video1.pack(pady=10)
        my_entry1 = Entry(selecteur)
        my_entry1.pack()
        label_video2 = Label(selecteur, text="Vidéo 2")
        label_video2.pack(pady=10)
        my_entry2 = Entry(selecteur)
        my_entry2.pack()

    label = Label(selecteur, text="Sélectionnez ce que vous voulez faire")
    label.pack(pady=10)
    btn = Button(selecteur, text="Interface sur webcam", command=faire_webcam)
    btn.pack(pady=10)
    btn1 = Button(selecteur, text="Interface sur vidéo", command=faire_video)
    btn1.pack(pady=10)

    btn1["state"] = "disable"

    mainloop()

    if selection == 1:
        arr = calculer_liste_webcams_dispo()
        App(Tk(), "Projet tennis de table " + version, arr, video=[])
    elif selection == 2:
        App(Tk(), "Projet tennis de table " + version, video=selection_video)
