import asyncio
import numpy as np
import sys
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageTk
import pygame
import time
from bleak import BleakClient, BleakScanner
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from collections import deque

# Import de votre fonction de détection
# from fonctions_de_detection import methode_3

#######################
# Configuration générale
#######################

# Adresse MAC du périphérique BLE (à remplacer par la vôtre)
address = "D4:22:CD:00:A0:FD"

# UUID des caractéristiques BLE
short_payload_characteristic_uuid = "15172004-4947-11e9-8646-d663bd873d93"
measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"

payload_modes = {
    "Free acceleration": [6, b"\x06"],
}

#############
# Données BLE
#############

times = []
xs = []
ys = []
zs = []
deriv_x = []
deriv_y = []
deriv_z = []

last_bounce_time = 0
delta_t = 0.05
sampling_interval = delta_t

threshold = 0.40
min_interval = 0.5
detected_bounce_times = []

window_size = 5
acc_x_filtered_prev = 0
acc_y_filtered_prev = 0
acc_z_filtered_prev = 0

data_lock = threading.Lock()

#########
# Tkinter
#########

root = tk.Tk()
root.title("Interface Tennis de Table")

######################
# Variables Scoreboard
######################
playerA_sets = tk.IntVar(value=0)
playerB_sets = tk.IntVar(value=0)
playerA_points = tk.IntVar(value=0)
playerB_points = tk.IntVar(value=0)

server_var = tk.StringVar(value="A")  # "A" ou "B"
threshold_var = tk.DoubleVar(value=threshold)

# File d'affichage des 10 derniers rebonds détectés (table/raquette)
last_bounces = deque(maxlen=10)

#########################
# Frames pour l'interface
#########################

# Frame du haut pour les vidéos
frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Frame pour les vidéos (à gauche cam 1, à droite cam 2)
frame_videos = tk.Frame(frame_top)
frame_videos.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Frame pour l'affichage du score, en-dessous
frame_score = tk.Frame(root)
frame_score.pack(side=tk.BOTTOM, fill=tk.X)

# Frame de gauche pour la liste des rebonds détectés
frame_bounces = tk.Frame(frame_score)
frame_bounces.pack(side=tk.LEFT, padx=10)

# Frame central pour le tableau de score
frame_scoreboard = tk.Frame(frame_score)
frame_scoreboard.pack(side=tk.LEFT, padx=10)

# Frame pour le serveur actuel
frame_server = tk.Frame(frame_score)
frame_server.pack(side=tk.LEFT, padx=10)

# Frame pour le contrôle du seuil
frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

###################
# Widgets des vidéos
###################
# On utilise des Labels pour afficher les images OpenCV converties en Tkinter
label_cam1 = tk.Label(frame_videos)
label_cam1.pack(side=tk.LEFT, padx=5, pady=5)

label_cam2 = tk.Label(frame_videos)
label_cam2.pack(side=tk.LEFT, padx=5, pady=5)

########################
# Widgets du scoreboard
########################

# Liste des rebonds
bounces_label = tk.Label(frame_bounces, text="Derniers rebonds détectés:")
bounces_label.pack(side=tk.TOP)
bounces_list = tk.Listbox(frame_bounces, height=10, width=40)
bounces_list.pack(side=tk.TOP)

# Scoreboard
tk.Label(frame_scoreboard, text="Joueur A").grid(row=0, column=1, padx=5)
tk.Label(frame_scoreboard, text="Joueur B").grid(row=0, column=3, padx=5)

# Sets
tk.Label(frame_scoreboard, text="Sets:").grid(row=1, column=0)
label_A_sets = tk.Label(frame_scoreboard, textvariable=playerA_sets)
label_A_sets.grid(row=1, column=1)
tk.Button(frame_scoreboard, text="+", command=lambda: playerA_sets.set(playerA_sets.get()+1)).grid(row=1, column=2)
tk.Button(frame_scoreboard, text="-", command=lambda: playerA_sets.set(max(0,playerA_sets.get()-1))).grid(row=1, column=4)

label_B_sets = tk.Label(frame_scoreboard, textvariable=playerB_sets)
label_B_sets.grid(row=1, column=3)

tk.Button(frame_scoreboard, text="+", command=lambda: playerB_sets.set(playerB_sets.get()+1)).grid(row=1, column=5)
tk.Button(frame_scoreboard, text="-", command=lambda: playerB_sets.set(max(0,playerB_sets.get()-1))).grid(row=1, column=6)

# Points
tk.Label(frame_scoreboard, text="Points:").grid(row=2, column=0)
label_A_points = tk.Label(frame_scoreboard, textvariable=playerA_points)
label_A_points.grid(row=2, column=1)
tk.Button(frame_scoreboard, text="+", command=lambda: playerA_points.set(playerA_points.get()+1)).grid(row=2, column=2)
tk.Button(frame_scoreboard, text="-", command=lambda: playerA_points.set(max(0,playerA_points.get()-1))).grid(row=2, column=4)

label_B_points = tk.Label(frame_scoreboard, textvariable=playerB_points)
label_B_points.grid(row=2, column=3)

tk.Button(frame_scoreboard, text="+", command=lambda: playerB_points.set(playerB_points.get()+1)).grid(row=2, column=5)
tk.Button(frame_scoreboard, text="-", command=lambda: playerB_points.set(max(0,playerB_points.get()-1))).grid(row=2, column=6)

# Serveur
tk.Label(frame_server, text="Serveur actuel:").pack(side=tk.LEFT)
label_server = tk.Label(frame_server, textvariable=server_var, fg="red")
label_server.pack(side=tk.LEFT, padx=5)
tk.Button(frame_server, text="↔", command=lambda: server_var.set("A" if server_var.get()=="B" else "B")).pack(side=tk.LEFT, padx=5)

# Contrôles pour le seuil
threshold_label = tk.Label(frame_controls, text="Seuil de détection (g/s) :")
threshold_label.pack(side=tk.LEFT, padx=5, pady=5)

threshold_entry = tk.Entry(frame_controls, textvariable=threshold_var, width=5)
threshold_entry.pack(side=tk.LEFT, padx=5, pady=5)


#####################
# Graphiques Matplotlib
#####################
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
plt.subplots_adjust(hspace=0.5)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

def update_threshold(*args):
    global threshold
    try:
        threshold = float(threshold_var.get())
    except ValueError:
        threshold_var.set(threshold)
threshold_var.trace_add("write", update_threshold)

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

def handle_short_payload_notification(sender, data):
    global times, xs, ys, zs, deriv_x, deriv_y, deriv_z, last_bounce_time, detected_bounce_times
    global acc_x_filtered_prev, acc_y_filtered_prev, acc_z_filtered_prev
    formatted_data = encode_free_accel_bytes_to_string(data)
    timestamp = formatted_data["timestamp"][0] / 1000.0
    acceleration_x = formatted_data["x"][0]
    acceleration_y = formatted_data["y"][0]
    acceleration_z = formatted_data["z"][0]

    current_time = time.time()

    with data_lock:
        times.append(current_time)
        xs.append(acceleration_x)
        ys.append(acceleration_y)
        zs.append(acceleration_z)

        # Filtrage moyenne mobile
        if len(xs) >= window_size:
            acc_x_filtered = np.mean(xs[-window_size:])
            acc_y_filtered = np.mean(ys[-window_size:])
            acc_z_filtered = np.mean(zs[-window_size:])
        else:
            acc_x_filtered = xs[-1]
            acc_y_filtered = ys[-1]
            acc_z_filtered = zs[-1]

        if len(xs) == 1:
            acc_x_filtered_prev = acc_x_filtered
            acc_y_filtered_prev = acc_y_filtered
            acc_z_filtered_prev = acc_z_filtered

        dt_ = delta_t
        deriv_x_value = (acc_x_filtered - acc_x_filtered_prev) / dt_
        deriv_y_value = (acc_y_filtered - acc_y_filtered_prev) / dt_
        deriv_z_value = (acc_z_filtered - acc_z_filtered_prev) / dt_

        deriv_x.append(deriv_x_value)
        deriv_y.append(deriv_y_value)
        deriv_z.append(deriv_z_value)

        acc_x_filtered_prev = acc_x_filtered
        acc_y_filtered_prev = acc_y_filtered
        acc_z_filtered_prev = acc_z_filtered

        max_length = 1000
        if len(times) > max_length:
            times = times[-max_length:]
            xs = xs[-max_length:]
            ys = ys[-max_length:]
            zs = zs[-max_length:]
            deriv_x = deriv_x[-max_length:]
            deriv_y = deriv_y[-max_length:]
            deriv_z = deriv_z[-max_length:]
            detected_bounce_times = [t for t in detected_bounce_times if t >= times[0]]

        deriv_magnitude = np.sqrt(deriv_x_value**2 + deriv_y_value**2 + deriv_z_value**2)

        # Détection rebond table
        if deriv_magnitude > threshold and (current_time - last_bounce_time) >= min_interval:
            last_bounce_time = current_time
            detected_bounce_times.append(current_time)
            # Ajouter un rebond dans la liste
            add_bounce_event("Table", current_time)

###############
# Fonction Rebond
###############
def add_bounce_event(source, timestamp):
    # source: "Table", "Raquette A" ou "Raquette B"
    # timestamp: temps du rebond
    event_str = f"{source} - {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
    last_bounces.append(event_str)
    update_bounces_display()

def update_bounces_display():
    bounces_list.delete(0, tk.END)
    for bounce in last_bounces:
        bounces_list.insert(tk.END, bounce)

###################
# Animation Matplotlib
###################

def animate(i):
    with data_lock:
        times_copy = times.copy()
        xs_copy = xs.copy()
        ys_copy = ys.copy()
        zs_copy = zs.copy()
        deriv_x_copy = deriv_x.copy()
        deriv_y_copy = deriv_y.copy()
        deriv_z_copy = deriv_z.copy()
        detected_bounce_times_copy = detected_bounce_times.copy()

    current_time = time.time()
    indices = [i for i, t in enumerate(times_copy) if t >= current_time - 5]
    if indices:
        times_copy = [times_copy[i] for i in indices]
        xs_copy = [xs_copy[i] for i in indices]
        ys_copy = [ys_copy[i] for i in indices]
        zs_copy = [zs_copy[i] for i in indices]
        deriv_x_copy = [deriv_x_copy[i] for i in indices]
        deriv_y_copy = [deriv_y_copy[i] for i in indices]
        deriv_z_copy = [deriv_z_copy[i] for i in indices]
    else:
        times_copy = []
        xs_copy = []
        ys_copy = []
        zs_copy = []
        deriv_x_copy = []
        deriv_y_copy = []
        deriv_z_copy = []

    times_rel = [t - (current_time - 5) for t in times_copy]
    bounce_times_rel = [t - (current_time - 5) for t in detected_bounce_times_copy if t >= current_time - 5]

    ax1.clear()
    ax2.clear()

    ax1.plot(times_rel, xs_copy, label="Acc_X", color="red")
    ax1.plot(times_rel, ys_copy, label="Acc_Y", color="green")
    ax1.plot(times_rel, zs_copy, label="Acc_Z", color="blue")
    ax1.set_title("Accélération (5s)")
    ax1.set_ylabel("g")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times_rel, deriv_x_copy, label="dAcc_X", color="red")
    ax2.plot(times_rel, deriv_y_copy, label="dAcc_Y", color="green")
    ax2.plot(times_rel, deriv_z_copy, label="dAcc_Z", color="blue")
    ax2.set_title("Dérivée de l'accélération (5s)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("g/s")
    ax2.legend()
    ax2.grid(True)

    for t in bounce_times_rel:
        ax1.axvline(x=t, color='blue', linestyle='--')
        ax2.axvline(x=t, color='blue', linestyle='--')

    canvas.draw()

ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)

###################
# Capture Vidéo
###################
cap1 = cv2.VideoCapture(0)  # Première webcam
cap2 = cv2.VideoCapture(1)  # Deuxième webcam (à adapter selon votre config)

def update_video_frames():
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(frame1, (320, 240)) 
        #imgtk1 = ImageTk.PhotoImage(image=tk.PhotoImage(width=320, height=240))
        # Pour un affichage correct, on peut utiliser PIL
        im1 = Image.fromarray(img1)
        imgtk1 = ImageTk.PhotoImage(image=im1)
        label_cam1.configure(image=imgtk1)
        label_cam1.image = imgtk1

    if ret2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(frame2, (320, 240))
        im2 = Image.fromarray(img2)
        imgtk2 = ImageTk.PhotoImage(image=im2)
        label_cam2.configure(image=imgtk2)
        label_cam2.image = imgtk2

    root.after(50, update_video_frames)  # mettre à jour toutes les 50ms environ

########################
# Thread BLE
########################
async def main(ble_address):
    device = await BleakScanner.find_device_by_address(ble_address, timeout=20.0)
    if device is None:
        print("Périphérique non trouvé.")
        return
    async with BleakClient(device) as client:
        await client.start_notify(short_payload_characteristic_uuid, handle_short_payload_notification)
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

def run_ble_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main(address))

ble_thread = threading.Thread(target=run_ble_client)
ble_thread.daemon = True
ble_thread.start()

###################
# Fermeture
###################
def on_closing():
    cap1.release()
    cap2.release()
    plt.close('all')
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Lancement de la mise à jour vidéo
update_video_frames()

# Boucle principale Tkinter
root.mainloop()
