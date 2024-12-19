import asyncio
import numpy as np
import sys
import datetime as dt
from PIL import Image, ImageTk
import pygame
import time
from bleak import BleakClient, BleakScanner
import threading
import tkinter as tk
import cv2
from collections import deque
from fonctions_de_detection import methode_3_simplified
import matplotlib
matplotlib.use("TkAgg")  # Backend TK
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from scipy.signal import butter, sosfilt

sys.path.append('./')
from utils_interface import *  # Pour calculer_liste_webcams_dispo

#######################
# Configuration générale
#######################

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sensors_config = {
    "D4:22:CD:00:A0:FD": {"name": "Table", "type": "table"},
    "D4:22:CD:00:9B:E6": {"name": "Raquette A", "type": "raquette"},
    "D4:22:CD:00:9E:2F": {"name": "Raquette B", "type": "raquette"}
}

measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"
complete_quat_characteristic_uuid = "15172003-4947-11e9-8646-d663bd873d93"

payload_modes = {
    "Complete Quaternion": [3, b"\x03"]
}

pygame.mixer.init()
sound = pygame.mixer.Sound("rebond_beep-07a.wav")
sequence = ""

root = tk.Tk()
root.title("Interface Tennis de Table")

playerA_sets = tk.IntVar(value=0)
playerB_sets = tk.IntVar(value=0)
playerA_points = tk.IntVar(value=0)
playerB_points = tk.IntVar(value=0)
server_var = tk.StringVar(value="A")
threshold = 0.40
threshold_var = tk.DoubleVar(value=threshold)

last_bounces = deque(maxlen=10)

#frame_top = tk.Frame(root)
#frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_main = tk.Frame(root)
frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_videos = tk.Frame(frame_main)
frame_videos.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

frame_graph = tk.Frame(frame_main)
frame_graph.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

# On configure les colonnes pour qu’elles s’ajustent
frame_main.grid_columnconfigure(0, weight=1)  # vidéos prend de la place
frame_main.grid_columnconfigure(1, weight=1)  # graph prend de la place
frame_main.grid_rowconfigure(0, weight=1)

frame_score = tk.Frame(root)
frame_score.pack(side=tk.BOTTOM, fill=tk.X)

frame_bounces = tk.Frame(frame_score)
frame_bounces.pack(side=tk.LEFT, padx=10)

frame_scoreboard = tk.Frame(frame_score)
frame_scoreboard.pack(side=tk.LEFT, padx=10)

frame_server = tk.Frame(frame_score)
frame_server.pack(side=tk.LEFT, padx=10)

frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

label_cam1 = tk.Label(frame_videos)
label_cam1.pack(side=tk.LEFT, padx=5, pady=5)

label_cam2 = tk.Label(frame_videos)
label_cam2.pack(side=tk.LEFT, padx=5, pady=5)

cam1_index = 0
cam2_index = 2

bounces_label = tk.Label(frame_bounces, text="Derniers rebonds détectés:")
bounces_label.pack(side=tk.TOP)
bounces_list = tk.Listbox(frame_bounces, height=10, width=40)
bounces_list.pack(side=tk.TOP)

tk.Label(frame_scoreboard, text="Joueur A").grid(row=0, column=1, padx=5)
tk.Label(frame_scoreboard, text="Joueur B").grid(row=0, column=3, padx=5)

tk.Label(frame_scoreboard, text="Sets:").grid(row=1, column=0)
label_A_sets = tk.Label(frame_scoreboard, textvariable=playerA_sets)
label_A_sets.grid(row=1, column=1)
tk.Button(frame_scoreboard, text="+", command=lambda: playerA_sets.set(playerA_sets.get()+1)).grid(row=1, column=2)
tk.Button(frame_scoreboard, text="-", command=lambda: playerA_sets.set(max(0,playerA_sets.get()-1))).grid(row=1, column=4)

label_B_sets = tk.Label(frame_scoreboard, textvariable=playerB_sets)
label_B_sets.grid(row=1, column=3)
tk.Button(frame_scoreboard, text="+", command=lambda: playerB_sets.set(playerB_sets.get()+1)).grid(row=1, column=5)
tk.Button(frame_scoreboard, text="-", command=lambda: playerB_sets.set(max(0,playerB_sets.get()-1))).grid(row=1, column=6)

tk.Label(frame_scoreboard, text="Points:").grid(row=2, column=0)
label_A_points = tk.Label(frame_scoreboard, textvariable=playerA_points)
label_A_points.grid(row=2, column=1)
tk.Button(frame_scoreboard, text="+", command=lambda: playerA_points.set(playerA_points.get()+1)).grid(row=2, column=2)
tk.Button(frame_scoreboard, text="-", command=lambda: playerA_points.set(max(0,playerA_points.get()-1))).grid(row=2, column=4)

label_B_points = tk.Label(frame_scoreboard, textvariable=playerB_points)
label_B_points.grid(row=2, column=3)
tk.Button(frame_scoreboard, text="+", command=lambda: playerB_points.set(playerB_points.get()+1)).grid(row=2, column=5)
tk.Button(frame_scoreboard, text="-", command=lambda: playerB_points.set(max(0,playerB_points.get()-1))).grid(row=2, column=6)

tk.Label(frame_server, text="Serveur actuel:").pack(side=tk.LEFT)
label_server = tk.Label(frame_server, textvariable=server_var, fg="red")
label_server.pack(side=tk.LEFT, padx=5)
tk.Button(frame_server, text="↔", command=lambda: server_var.set("A" if server_var.get()=="B" else "B")).pack(side=tk.LEFT, padx=5)

threshold_label = tk.Label(frame_controls, text="Seuil de détection (g/s) :")
threshold_label.pack(side=tk.LEFT, padx=5, pady=5)
threshold_entry = tk.Entry(frame_controls, textvariable=threshold_var, width=5)
threshold_entry.pack(side=tk.LEFT, padx=5, pady=5)

def update_threshold(*args):
    global threshold
    try:
        threshold = float(threshold_var.get())
    except ValueError:
        threshold_var.set(threshold)
threshold_var.trace_add("write", update_threshold)

delta_t = 0.05
min_interval = 0.5
faire_yolo = False
data_lock = threading.Lock()

for addr in sensors_config.keys():
    sensors_config[addr].update({
        "times": [],
        "xs": [], "ys": [], "zs": [],
        "quat_w": [], "quat_x": [], "quat_y": [], "quat_z": [],
        "deriv_x": [], "deriv_y": [], "deriv_z": [],
        "last_bounce_time": 0,
        "window_size": 5,
        "acc_x_filtered_prev": 0,
        "acc_y_filtered_prev": 0,
        "acc_z_filtered_prev": 0
    })

lowcut = 20.0
highcut = 25.0
order = 4
fs = 100.0
def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= nyq:
        # Ajustement si nécessaire
        return None
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

sos = butter_bandpass(lowcut, highcut, fs, order)

###############################
# AJOUT AUTOMATE
###############################
STATE_WAIT_SERVE_RACKET = 0
STATE_WAIT_FIRST_TABLE = 1
STATE_WAIT_SECOND_TABLE = 2
STATE_WAIT_OTHER_RACKET = 3
STATE_WAIT_TABLE = 4

current_server = "Raquette A"  # vous pouvez changer dynamiquement selon server_var
current_state = STATE_WAIT_SERVE_RACKET
last_racket = None

def reset_automate():
    global current_state, current_server, last_racket
    # réinitialiser la logique si besoin
    current_server = "Raquette A"
    current_state = STATE_WAIT_SERVE_RACKET
    last_racket = None

def next_state_after_racket(racket):
    global current_state, last_racket
    if current_state == STATE_WAIT_SERVE_RACKET:
        last_racket = racket
        current_state = STATE_WAIT_FIRST_TABLE
    elif current_state == STATE_WAIT_OTHER_RACKET:
        last_racket = racket
        current_state = STATE_WAIT_TABLE
    # Sinon pas de changement d'état (événement non attendu dans ce contexte)

def next_state_after_table():
    global current_state
    if current_state == STATE_WAIT_FIRST_TABLE:
        current_state = STATE_WAIT_SECOND_TABLE
    elif current_state == STATE_WAIT_SECOND_TABLE:
        # Après le service complet (raquette du serveur + 2 tables), on attend l'autre raquette
        current_state = STATE_WAIT_OTHER_RACKET
    elif current_state == STATE_WAIT_TABLE:
        # Après la table en échange normal, on attend l'autre raquette
        current_state = STATE_WAIT_OTHER_RACKET

def event_allowed(source):
    # Vérifie si l'événement "source" est autorisé dans l'état courant
    global current_state, last_racket, current_server
    if current_state == STATE_WAIT_SERVE_RACKET:
        # On attend la raquette du serveur
        return source == current_server
    elif current_state == STATE_WAIT_FIRST_TABLE:
        # On attend la table
        return source == "Table"
    elif current_state == STATE_WAIT_SECOND_TABLE:
        # On attend la table
        return source == "Table"
    elif current_state == STATE_WAIT_OTHER_RACKET:
        # On attend la raquette adverse
        if source.startswith("Raquette"):
            return (last_racket is not None and source != last_racket)
        return False
    elif current_state == STATE_WAIT_TABLE:
        # On attend la table
        return source == "Table"
    return False

def update_automate_after_event(source):
    # Met à jour l'état après un événement valide
    if source.startswith("Raquette"):
        next_state_after_racket(source)
    elif source == "Table":
        next_state_after_table()

###############################

# Variables globales
events = []
last_event_time = None
point_timeout_ms = 5000  # 5 secondes
point_ended = False  # Pour savoir si le point est terminé

def end_point():
    global point_ended, events, current_state, current_server, last_racket
    point_ended = True
    last_bounces.append("Fin du point (pas d'événement depuis 3s)")
    update_bounces_display()

    # Déterminer qui gagne le point
    # 1. Vérifier si le service a été complété
    # current_state indique où on en est resté :
    # - STATE_WAIT_SERVE_RACKET : On attendait encore la raquette du serveur => service jamais commencé correctement
    # - STATE_WAIT_FIRST_TABLE ou STATE_WAIT_SECOND_TABLE : le service n'a pas été complété (pas 2 rebonds de table)
    # - STATE_WAIT_OTHER_RACKET ou STATE_WAIT_TABLE : le service a été complété, on est en échange normal

    # Récupérer les deux derniers événements pour l'analyse de fin de point
    last_two_events = events[-2:] if len(events) >= 2 else events[:]
    # last_two_events est une liste de tuples (timestamp, source)

    # Fixer le temps de fin du point pour figer le graphe
    end_of_point_time = last_event_time if last_event_time is not None else time.time()

    def give_point_to_A():
        playerA_points.set(playerA_points.get() + 1)

    def give_point_to_B():
        playerB_points.set(playerB_points.get() + 1)

    if current_state == STATE_WAIT_SERVE_RACKET:
        # Le serveur n'a même pas frappé correctement => l'adversaire gagne
        if current_server == "Raquette A":
            # C'est A qui devait servir, donc B gagne
            give_point_to_B()
        else:
            # C'est B qui devait servir, donc A gagne
            give_point_to_A()

    elif current_state == STATE_WAIT_FIRST_TABLE or current_state == STATE_WAIT_SECOND_TABLE:
        # Le service a commencé (raquette du serveur) mais pas complété (pas 2x table)
        # L'adversaire du serveur gagne
        if current_server == "Raquette A":
            # A servait, pas complété => B gagne
            give_point_to_B()
        else:
            # B servait, pas complété => A gagne
            give_point_to_A()

    else:
        # Le service est complété (on était dans STATE_WAIT_OTHER_RACKET ou STATE_WAIT_TABLE)
        # On regarde les deux derniers événements
        if len(last_two_events) < 2:
            # Moins de 2 événements, situation rare, on peut estimer qu'aucun échange final n'a eu lieu
            # Dans ce cas, considérer que l'adversaire du dernier frappeur gagne, ou autre logique
            # Pour simplifier, si pas assez d'événements, donner le point à l'adversaire du serveur
            if current_server == "Raquette A":
                give_point_to_B()
            else:
                give_point_to_A()
        else:
            # last_two_events contient 2 tuples (timestamp, source)
            e1_source = last_two_events[0][1]
            e2_source = last_two_events[1][1]

            # Cas donnant le point à A : (Raquette A, Table) ou (Table, Raquette B)
            if (e1_source == "Raquette A" and e2_source == "Table") or (e1_source == "Table" and e2_source == "Raquette B"):
                give_point_to_A()
            else:
                # Sinon, point à B
                give_point_to_B()

def check_point_timeout(last_time_checked):
    global last_event_time, point_ended
    # Cette fonction est appelée après 3s du dernier événement
    # On vérifie si last_event_time n'a pas changé
    if last_event_time == last_time_checked and not point_ended:
        # Aucun nouvel événement depuis 3s, fin du point
        end_point()

def add_bounce_event(source, timestamp):
    global sequence, frame1, frame2, results1, results2, faire_yolo, last_event_time, point_ended

    if point_ended:
        # Si le point est déjà terminé, on ignore les événements
        return
    
    # AJOUT AUTOMATE : Vérification de l'événement
    if not event_allowed(source):
        # Événement non attendu, on l'ignore
        return
    else:
        # Événement autorisé
        update_automate_after_event(source)

    if source == "Raquette A":
        sequence += "A"
        if len(results1) > 0 and len(results1[0].keypoints.xy) > 0:
            if results1[0].keypoints.xy[0][10][1] == 0 or results1[0].keypoints.xy[0][8][1] == 0:
                last_bounces.append("Coude ou poignet non visible")
            elif results1[0].keypoints.xy[0][10][0] > results1[0].keypoints.xy[0][8][0]:
                last_bounces.append("revers")
            else:
                last_bounces.append("coup droit")
    elif source == "Raquette B":
        sequence += "B"
        if len(results2) > 0 and len(results2[0].keypoints.xy) > 0:
            if results2[0].keypoints.xy[0][10][1] == 0 or results2[0].keypoints.xy[0][8][1] == 0:
                last_bounces.append("Coude ou poignet non visible")
            elif results2[0].keypoints.xy[0][10][0] > results2[0].keypoints.xy[0][8][0]:
                last_bounces.append("revers")
            else:
                last_bounces.append("coup droit")
    elif source == "Table":
        sequence += "R"

    event_str = f"{source} - {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
    last_bounces.append(event_str)
    update_bounces_display()
    events.append((timestamp, source))

    # Mettre à jour last_event_time et programmer le check timeout
    last_event_time = timestamp
    root.after(point_timeout_ms, lambda: check_point_timeout(timestamp))

# Variables globales pour gérer le temps de début et de fin du point
start_of_point_time = None
end_of_point_time = None

def start_new_point():
    global events, sequence, point_ended, last_racket, current_state, current_server, last_event_time
    global start_of_point_time, end_of_point_time

    reset_automate()
    sequence = ""
    events.clear()
    point_ended = False
    last_event_time = None
    last_bounces.clear()
    update_bounces_display()

    # Enregistrer le temps de début du point
    start_of_point_time = time.time()
    end_of_point_time = None  # Pas encore de fin
def update_bounces_display():
    global sequence
    bounces_list.delete(0, tk.END)
    check_sequence(sequence)
    for bounce in last_bounces:
        bounces_list.insert(tk.END, bounce)

def detect_bounces_table(sensor, deriv_magnitude, current_time):
    if deriv_magnitude > threshold and (current_time - sensor["last_bounce_time"]) >= min_interval:
        sensor["last_bounce_time"] = current_time
        add_bounce_event(sensor["name"], current_time)
        if sound:
            sound.play()

def detect_bounces_raquette(sensor):
    with data_lock:
        times_arr = np.array(sensor["times"])
        freeacc_x = np.array(sensor["xs"])
        freeacc_y = np.array(sensor["ys"])
        freeacc_z = np.array(sensor["zs"])
        qw = np.array(sensor["quat_w"])
        qx = np.array(sensor["quat_x"])
        qy = np.array(sensor["quat_y"])
        qz = np.array(sensor["quat_z"])

    detected_rebounds = methode_3_simplified(times_arr, freeacc_x, freeacc_y, freeacc_z, qw, qx, qy, qz)
    for rebound_time in detected_rebounds:
        if rebound_time - sensor["last_bounce_time"] >= min_interval:
            sensor["last_bounce_time"] = rebound_time
            add_bounce_event(sensor["name"], rebound_time)
            if sound:
                sound.play()

def parse_complete_quaternion_data(data_bytes):
    if len(data_bytes) != 40:
        return None
    data_format = np.dtype([
        ('timestamp', np.uint32),
        ('quat_w', np.float32),
        ('quat_x', np.float32),
        ('quat_y', np.float32),
        ('quat_z', np.float32),
        ('accel_x', np.float32),
        ('accel_y', np.float32),
        ('accel_z', np.float32),
        ('dq_w', np.float32),
        ('dq_x', np.float32)
    ])
    data = np.frombuffer(data_bytes, dtype=data_format)
    return data

def handle_notification_factory(ble_address):
    def handle_notification(sender, data):
        sensor = sensors_config[ble_address]
        parsed = parse_complete_quaternion_data(data)
        if parsed is None:
            return

        accel_x = parsed['accel_x'][0]
        accel_y = parsed['accel_y'][0]
        accel_z = parsed['accel_z'][0]
        qw = parsed['quat_w'][0]
        qx = parsed['quat_x'][0]
        qy = parsed['quat_y'][0]
        qz = parsed['quat_z'][0]

        current_time = time.time()

        with data_lock:
            sensor["times"].append(current_time)
            sensor["xs"].append(accel_x)
            sensor["ys"].append(accel_y)
            sensor["zs"].append(accel_z)
            sensor["quat_w"].append(qw)
            sensor["quat_x"].append(qx)
            sensor["quat_y"].append(qy)
            sensor["quat_z"].append(qz)

            w = sensor["window_size"]
            xs = sensor["xs"]
            ys = sensor["ys"]
            zs = sensor["zs"]

            if len(xs) >= w:
                acc_x_filtered = np.mean(xs[-w:])
                acc_y_filtered = np.mean(ys[-w:])
                acc_z_filtered = np.mean(zs[-w:])
            else:
                acc_x_filtered = xs[-1]
                acc_y_filtered = ys[-1]
                acc_z_filtered = zs[-1]

            if len(xs) == 1:
                sensor["acc_x_filtered_prev"] = acc_x_filtered
                sensor["acc_y_filtered_prev"] = acc_y_filtered
                sensor["acc_z_filtered_prev"] = acc_z_filtered

            dt_ = delta_t
            deriv_x_value = (acc_x_filtered - sensor["acc_x_filtered_prev"]) / dt_
            deriv_y_value = (acc_y_filtered - sensor["acc_y_filtered_prev"]) / dt_
            deriv_z_value = (acc_z_filtered - sensor["acc_z_filtered_prev"]) / dt_

            sensor["deriv_x"].append(deriv_x_value)
            sensor["deriv_y"].append(deriv_y_value)
            sensor["deriv_z"].append(deriv_z_value)

            sensor["acc_x_filtered_prev"] = acc_x_filtered
            sensor["acc_y_filtered_prev"] = acc_y_filtered
            sensor["acc_z_filtered_prev"] = acc_z_filtered

            max_length = 1000
            if len(sensor["times"]) > max_length:
                for k in ["times","xs","ys","zs","quat_w","quat_x","quat_y","quat_z","deriv_x","deriv_y","deriv_z"]:
                    sensor[k] = sensor[k][-max_length:]

            deriv_magnitude = np.sqrt(deriv_x_value**2 + deriv_y_value**2 + deriv_z_value**2)

        # Détection de rebond
        if sensor["type"] == "table":
            detect_bounces_table(sensor, deriv_magnitude, current_time)
        else:
            if len(sensor["xs"]) > 100:
                detect_bounces_raquette(sensor)

    return handle_notification

def try_open_camera(index):
    if index is not None and index >= 0:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        else:
            cap.release()
            return None
    return None

arr = calculer_liste_webcams_dispo()

cap1 = try_open_camera(cam1_index)
cap2 = try_open_camera(cam2_index)

frame1 = []
frame2 = []
results1 = []
results2 = []

appliquer_pose_estimation(np.zeros((100,100,3), dtype=np.uint8))

def update_selected_cameras(*args):
    global cap1, cap2
    new_cam1_index = cam1_var.get()
    new_cam2_index = cam2_var.get()
    if cap1:
        cap1.release()
    if cap2:
        cap2.release()
    cap1 = try_open_camera(new_cam1_index)
    cap2 = try_open_camera(new_cam2_index)

def update_video_frames():
    global frame1, frame2, faire_yolo, results1, results2
    if cap1 and cap1.isOpened():
        ret1, frame1 = cap1.read()
    else:
        ret1, frame1 = False, None

    if cap2 and cap2.isOpened():
        ret2, frame2 = cap2.read()
    else:
        ret2, frame2 = False, None

    if ret1:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        if faire_yolo:
            results1 = appliquer_pose_estimation(frame1)
            frame1 = results1.plot()
        img1 = cv2.resize(frame1, (320, 240))
    else:
        img1 = np.zeros((240,320,3), dtype=np.uint8)

    if ret2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        if faire_yolo:
            results2 = appliquer_pose_estimation(frame2)
            frame2 = results2.plot()
        img2 = cv2.resize(frame2, (320, 240))
    else:
        img2 = np.zeros((240,320,3), dtype=np.uint8)

    imgtk1 = ImageTk.PhotoImage(image=Image.fromarray(img1))
    label_cam1.configure(image=imgtk1)
    label_cam1.image = imgtk1

    imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(img2))
    label_cam2.configure(image=imgtk2)
    label_cam2.image = imgtk2

    root.after(50, update_video_frames)

async def connect_device(ble_address):
    device = await BleakScanner.find_device_by_address(ble_address, timeout=20.0)
    if device is None:
        print(f"Périphérique {ble_address} non trouvé.")
        return
    async with BleakClient(device) as client:
        handler = handle_notification_factory(ble_address)
        await client.start_notify(complete_quat_characteristic_uuid, handler)
        payload_mode_values = payload_modes["Complete Quaternion"]
        payload_mode = payload_mode_values[1]
        measurement_default = b"\x01"
        start_measurement = b"\x01"
        full_turnon_payload = measurement_default + start_measurement + payload_mode
        await client.write_gatt_char(measurement_characteristic_uuid, full_turnon_payload, True)
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await client.stop_notify(complete_quat_characteristic_uuid)

async def main_ble():
    tasks = [connect_device(addr) for addr in sensors_config.keys()]
    await asyncio.gather(*tasks)

def run_ble_client():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_ble())
    except Exception as e:
        print(f"Error in run_ble_client: {e}")

def on_closing():
    if cap1:
        cap1.release()
    if cap2:
        cap2.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

update_video_frames()

show_graph_var = tk.BooleanVar(value=False)
#faire_yolo2 = tk.BooleanVar(value=False)

frame_graph_options = tk.Frame(root)
frame_graph_options.pack(side=tk.BOTTOM, fill=tk.X)
checkbutton1 = tk.Checkbutton(frame_graph_options, text="Afficher accélérations (10s)", variable=show_graph_var, command=lambda: toggle_graph()).pack(side=tk.LEFT, padx=5, pady=5)
checkbutton2 = tk.Checkbutton(frame_graph_options, text="Faire Yolo", command=lambda: faire_yolo_modif_value()).pack(side=tk.LEFT, padx=5, pady=5)


fig = Figure(figsize=(6,4))
ax_table = fig.add_subplot(3,1,1)
ax_raqA = fig.add_subplot(3,1,2)
ax_raqB = fig.add_subplot(3,1,3)

canvas_graph = FigureCanvasTkAgg(fig, master=frame_graph)

ani = None

def animate(i):
    # Si on ne veut pas animer une fois terminé, on peut vérifier :
    # if point_ended:
    #     return

    # On utilise start_of_point_time et éventuellement end_of_point_time
    if start_of_point_time is None:
        # Si pour une raison quelconque, le point n'a pas commencé, rien à afficher
        return

    if not point_ended:
        # Le point est en cours, on affiche jusqu'au temps actuel
        current_upper_time = time.time()
    else:
        # Le point est terminé, on fige le graphique au temps de fin
        if end_of_point_time is None:
            end_of_point_time = time.time()  # au cas où
        current_upper_time = end_of_point_time

    ax_table.clear()
    ax_raqA.clear()
    ax_raqB.clear()

    def plot_sensor(ax, sensor_addr):
        with data_lock:
            times = sensors_config[sensor_addr]["times"]
            xs = sensors_config[sensor_addr]["xs"]
            ys = sensors_config[sensor_addr]["ys"]
            zs = sensors_config[sensor_addr]["zs"]

        # Ici, on ne filtre plus sur les 10 dernières secondes
        # On prend tous les échantillons depuis start_of_point_time jusqu'à current_upper_time
        indices = [j for j,t in enumerate(times) if start_of_point_time <= t <= current_upper_time]

        if len(indices) > 0:
            # Calculer les temps relatifs par rapport au début du point
            times_rel = [times[j] - start_of_point_time for j in indices]
            xs_plot = [xs[j] for j in indices]
            ys_plot = [ys[j] for j in indices]
            zs_plot = [zs[j] for j in indices]

            ax.plot(times_rel, xs_plot, label="Acc_X", color="red")
            ax.plot(times_rel, ys_plot, label="Acc_Y", color="green")
            ax.plot(times_rel, zs_plot, label="Acc_Z", color="blue")

            ax.set_ylabel("g")
            ax.grid(True)
            ax.legend()
            # Ajuster les limites x pour voir tout le point
            ax.set_xlim(0, max(times_rel))
        else:
            ax.text(0.5,0.5,"Aucune donnée", ha='center', va='center')

        # Tracer les événements dans l'intervalle du point
        for (evt_time, evt_source) in events:
            if start_of_point_time <= evt_time <= current_upper_time:
                evt_rel_time = evt_time - start_of_point_time
                if evt_source == "Raquette A":
                    evt_color = "green"
                elif evt_source == "Raquette B":
                    evt_color = "blue"
                elif evt_source == "Table":
                    evt_color = "black"
                else:
                    evt_color = "red"
                ax.axvline(evt_rel_time, color=evt_color, linestyle='--', linewidth=2)

    plot_sensor(ax_table, "D4:22:CD:00:A0:FD")
    ax_table.set_title("Table")

    plot_sensor(ax_raqA, "D4:22:CD:00:9B:E6")
    ax_raqA.set_title("Raquette A")

    plot_sensor(ax_raqB, "D4:22:CD:00:9E:2F")
    ax_raqB.set_title("Raquette B")
    ax_raqB.set_xlabel("Temps depuis début du point (s)")

    fig.tight_layout()

def toggle_graph():
    global ani
    if show_graph_var.get():
        # Lancer l'animation
        ani = animation.FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
        # S'assurer que le widget est visible avant de lancer l'animation
        canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        frame_graph.grid()  # Affiche le frame_graph s'il était caché
    else:
        # Désactiver l'affichage des graphes
        if ani is not None:
            ani.event_source.stop()
            ani = None
        # Cacher le canvas
        canvas_graph.get_tk_widget().pack_forget()
        # Cacher le frame_graph si souhaité
        frame_graph.grid_remove()

def faire_yolo_modif_value():
    global faire_yolo
    faire_yolo = not faire_yolo

cam1_var = tk.IntVar(value=cam1_index)
cam2_var = tk.IntVar(value=cam2_index)

cam1_menu = tk.OptionMenu(frame_controls, cam1_var, *arr)
cam1_menu.pack(side=tk.LEFT, padx=5, pady=5)
cam2_menu = tk.OptionMenu(frame_controls, cam2_var, *arr)
cam2_menu.pack(side=tk.LEFT, padx=5, pady=5)

cam1_var.trace_add("write", update_selected_cameras)
cam2_var.trace_add("write", update_selected_cameras)

tk.Button(frame_controls, text="Rebond Raquette A", command=lambda: add_bounce_event("Raquette A", time.time())).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(frame_controls, text="Rebond Raquette B", command=lambda: add_bounce_event("Raquette B", time.time())).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(frame_controls, text="Rebond Table", command=lambda: add_bounce_event("Table", time.time())).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(frame_controls, text="Nouveau point", command=start_new_point).pack(side=tk.LEFT, padx=5, pady=5)

ble_thread = threading.Thread(target=run_ble_client)
ble_thread.daemon = True
ble_thread.start()

root.mainloop()
