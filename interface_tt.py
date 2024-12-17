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

#######################
# Configuration générale
#######################

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Capteurs BLE
sensors_config = {
    "D4:22:CD:00:A0:FD": {"name": "Table", "type": "table"},
    "D4:22:CD:00:9B:E6": {"name": "Raquette A", "type": "raquette"},
    "D4:22:CD:00:9E:2F": {"name": "Raquette B", "type": "raquette"}
}

measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"
complete_quat_characteristic_uuid = "15172003-4947-11e9-8646-d663bd873d93"

# On n'utilise que Complete Quaternion
payload_modes = {
    "Complete Quaternion": [3, b"\x03"]
}

pygame.mixer.init()
sound = None

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

delta_t = 0.05
min_interval = 0.5
data_lock = threading.Lock()

# Initialisation des listes pour chaque capteur
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
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

sos = butter_bandpass(lowcut, highcut, fs, order)

def add_bounce_event(source, timestamp):
    event_str = f"{source} - {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
    last_bounces.append(event_str)
    update_bounces_display()

def update_bounces_display():
    bounces_list.delete(0, tk.END)
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

# MODIFICATION ICI : Ajuster le parsing des données pour 40 octets
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
        # On ignore dq_w et dq_x
        # dq_w = parsed['dq_w'][0]
        # dq_x = parsed['dq_x'][0]

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

cap1 = try_open_camera(cam1_index)
cap2 = try_open_camera(cam2_index)

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

frame_graph_options = tk.Frame(root)
frame_graph_options.pack(side=tk.BOTTOM, fill=tk.X)
tk.Checkbutton(frame_graph_options, text="Afficher accélérations (10s)", variable=show_graph_var, command=lambda: toggle_graph()).pack(side=tk.LEFT, padx=5, pady=5)

frame_graph = tk.Frame(root)
fig = Figure(figsize=(6,4))
ax_table = fig.add_subplot(3,1,1)
ax_raqA = fig.add_subplot(3,1,2)
ax_raqB = fig.add_subplot(3,1,3)

canvas_graph = FigureCanvasTkAgg(fig, master=frame_graph)

ani = None

def animate(i):
    current_time = time.time()
    time_window = 10.0

    ax_table.clear()
    ax_raqA.clear()
    ax_raqB.clear()

    def plot_sensor(ax, sensor_addr):
        with data_lock:
            times = sensors_config[sensor_addr]["times"]
            xs = sensors_config[sensor_addr]["xs"]
            ys = sensors_config[sensor_addr]["ys"]
            zs = sensors_config[sensor_addr]["zs"]

        indices = [i for i,t in enumerate(times) if t >= current_time - time_window]
        if len(indices) > 0:
            times_rel = [times[i] - (current_time - time_window) for i in indices]
            xs_plot = [xs[i] for i in indices]
            ys_plot = [ys[i] for i in indices]
            zs_plot = [zs[i] for i in indices]

            ax.plot(times_rel, xs_plot, label="Acc_X", color="red")
            ax.plot(times_rel, ys_plot, label="Acc_Y", color="green")
            ax.plot(times_rel, zs_plot, label="Acc_Z", color="blue")

            ax.set_ylabel("g")
            ax.grid(True)
            ax.legend()
            ax.set_xlim(0, time_window)
        else:
            ax.text(0.5,0.5,"Aucune donnée", ha='center', va='center')

    plot_sensor(ax_table, "D4:22:CD:00:A0:FD")
    ax_table.set_title("Table (10s)")

    plot_sensor(ax_raqA, "D4:22:CD:00:9B:E6")
    ax_raqA.set_title("Raquette A (10s)")

    plot_sensor(ax_raqB, "D4:22:CD:00:9E:2F")
    ax_raqB.set_title("Raquette B (10s)")
    ax_raqB.set_xlabel("Temps (s)")

    fig.tight_layout()

def toggle_graph():
    global ani
    if show_graph_var.get():
        frame_graph.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)
    else:
        if ani is not None:
            ani.event_source.stop()
            ani._stop()
            ani = None
        canvas_graph.get_tk_widget().pack_forget()
        frame_graph.pack_forget()

ble_thread = threading.Thread(target=run_ble_client)
ble_thread.daemon = True
ble_thread.start()

root.mainloop()
