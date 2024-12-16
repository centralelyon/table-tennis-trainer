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

# Import de la fonction methode_3
from fonctions_de_detection import methode_3

#######################
# Configuration générale
#######################

# Capteurs BLE
sensors_config = {
    "D4:22:CD:00:A0:FD": {"name": "Table", "type": "table"},
    "D4:22:CD:00:9B:E6": {"name": "Raquette A", "type": "raquette"},
    "D4:22:CD:00:9E:2F": {"name": "Raquette B", "type": "raquette"}
}

short_payload_characteristic_uuid = "15172004-4947-11e9-8646-d663bd873d93"
measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"

payload_modes = {
    "Free acceleration": [6, b"\x06"],
}

pygame.mixer.init()
sound = None  # Ajuster si nécessaire

root = tk.Tk()
root.title("Interface Tennis de Table")

playerA_sets = tk.IntVar(value=0)
playerB_sets = tk.IntVar(value=0)
playerA_points = tk.IntVar(value=0)
playerB_points = tk.IntVar(value=0)

server_var = tk.StringVar(value="A")  # "A" ou "B"
threshold = 0.40
threshold_var = tk.DoubleVar(value=threshold)

last_bounces = deque(maxlen=10)

frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_videos = tk.Frame(frame_top)
frame_videos.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
cam2_index = 1

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

for addr in sensors_config.keys():
    sensors_config[addr].update({
        "times": [],
        "xs": [],
        "ys": [],
        "zs": [],
        "deriv_x": [],
        "deriv_y": [],
        "deriv_z": [],
        "last_bounce_time": 0,
        "window_size": 5,
        "acc_x_filtered_prev": 0,
        "acc_y_filtered_prev": 0,
        "acc_z_filtered_prev": 0
    })

delta_t = 0.05
min_interval = 0.5
data_lock = threading.Lock()

def add_bounce_event(source, timestamp):
    event_str = f"{source} - {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
    last_bounces.append(event_str)
    update_bounces_display()

def update_bounces_display():
    bounces_list.delete(0, tk.END)
    for bounce in last_bounces:
        bounces_list.insert(tk.END, bounce)

########################
# Méthode de détection
########################

def detect_bounces_table(sensor, deriv_magnitude, current_time):
    # Méthode actuelle pour la table
    if deriv_magnitude > threshold and (current_time - sensor["last_bounce_time"]) >= min_interval:
        sensor["last_bounce_time"] = current_time
        add_bounce_event(sensor["name"], current_time)
        if sound:
            sound.play()

def detect_bounces_raquette(sensor):
    # Méthode 3 : On considère que vous avez un fichier CSV contenant les données de la raquette
    # et une liste t_lines_original (temps annotés).
    # Ce code est purement indicatif. En pratique, vous devrez :
    # - Soit enregistrer les données dans un CSV
    # - Soit adapter methode_3 pour accepter des données en mémoire.

    file_path = "donnees_raquette.csv"  # Chemin vers votre fichier CSV pour la raquette
    t_lines_original = []  # À définir selon vos données réelles

    # Appel à la méthode_3
    detected_rebound_times, precision_metrics = methode_3(file_path, t_lines_original)

    # Parcourir les rebonds détectés et ajouter les événements
    for (_, rebound_time) in detected_rebound_times:
        # Ici, rebound_time est en secondes depuis le début du fichier.
        # Si vous souhaitez aligner ce temps avec l'horloge réelle, vous devrez ajuster.
        # Pour l'exemple, on considère rebound_time comme un timestamp absolu (c'est faux).
        # Vous devrez adapter le code pour convertir rebound_time (issu du CSV)
        # en un temps absolu ou en un temps cohérent avec le moment courant.

        # Par exemple, si rebound_time est relatif (depuis le début de l'enregistrement),
        # et que l'enregistrement a commencé à current_time_start, il faudra faire :
        # event_timestamp = current_time_start + rebound_time
        # Ici on va simplement utiliser le temps actuel, fictivement.
        current_time = time.time()
        add_bounce_event(sensor["name"], current_time)

        if sound:
            sound.play()

########################
# Fonction notification BLE
########################

def encode_free_accel_bytes_to_string(bytes_):
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

def handle_notification(sender, data, ble_address):
    sensor = sensors_config[ble_address]
    formatted_data = encode_free_accel_bytes_to_string(data)
    acceleration_x = formatted_data["x"][0]
    acceleration_y = formatted_data["y"][0]
    acceleration_z = formatted_data["z"][0]

    current_time = time.time()

    with data_lock:
        sensor["times"].append(current_time)
        sensor["xs"].append(acceleration_x)
        sensor["ys"].append(acceleration_y)
        sensor["zs"].append(acceleration_z)

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
            for k in ["times","xs","ys","zs","deriv_x","deriv_y","deriv_z"]:
                sensor[k] = sensor[k][-max_length:]

        deriv_magnitude = np.sqrt(deriv_x_value**2 + deriv_y_value**2 + deriv_z_value**2)

        # Selon le type de capteur, appliquer une méthode différente
        if sensor["type"] == "table":
            # Méthode existante (dérivée, threshold)
            detect_bounces_table(sensor, deriv_magnitude, current_time)
        else:
            # Capteur de raquette : utiliser methode_3
            # Vous pouvez décider d'appeler cette détection à intervalle régulier,
            # par exemple toutes les X secondes, ou une fois que vous avez suffisamment
            # de données. Ici, on l'appelle de manière illustrative.
            #
            # Dans un cas réel, vous feriez par exemple :
            # if len(sensor["xs"]) > 1000:
            #     detect_bounces_raquette(sensor)
            #
            # Pour l'exemple, on appelle directement la fonction
            detect_bounces_raquette(sensor)

###################
# Capture Vidéo
###################
if cam1_index is not None and cam1_index >= 0:
    cap1 = cv2.VideoCapture(cam1_index)
else:
    cap1 = None

if cam2_index is not None and cam2_index >= 0:
    cap2 = cv2.VideoCapture(cam2_index)
else:
    cap2 = None

def update_video_frames():
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
        img1 = cv2.resize(frame1, (320, 240)) 
    else:
        img1 = np.zeros((240,320,3), dtype=np.uint8)

    if ret2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
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

########################
# Thread BLE
########################

async def run_device(ble_address):
    device = await BleakScanner.find_device_by_address(ble_address, timeout=20.0)
    if device is None:
        print(f"Périphérique {ble_address} non trouvé.")
        return
    async with BleakClient(device) as client:
        await client.start_notify(short_payload_characteristic_uuid,
                                  lambda sender, data: handle_notification(sender, data, ble_address))
        payload_mode_values = payload_modes["Free acceleration"]
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
            await client.stop_notify(short_payload_characteristic_uuid)

def run_ble_client(ble_address):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_device(ble_address))

for addr in sensors_config.keys():
    t = threading.Thread(target=run_ble_client, args=(addr,))
    t.daemon = True
    t.start()

def on_closing():
    if cap1:
        cap1.release()
    if cap2:
        cap2.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

update_video_frames()
root.mainloop()
