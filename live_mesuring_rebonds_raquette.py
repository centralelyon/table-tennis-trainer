import asyncio
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import time
from bleak import BleakClient, BleakScanner
import threading
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, sosfilt
from collections import deque
from threading import Lock  # Import Lock for thread synchronization

# Handle asyncio issues on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Path to sound file (adjust if necessary)
sound_path_filename = os.path.join("sound", "rebond_beep-07a.wav")
sound = pygame.mixer.Sound(sound_path_filename)

# BLE device MAC addresses # MODIFICATION : ajouter vos 3 adresses
device_addresses = [
    "D4:22:CD:00:A0:FD",
    "D4:22:CD:00:9B:E6",
    "D4:22:CD:00:9E:2F"
]

# BLE characteristic UUIDs (verify with your device)
measurement_characteristic_uuid = "15172001-4947-11e9-8646-d663bd873d93"
medium_payload_characteristic_uuid = "15172003-4947-11e9-8646-d663bd873d93"

# Available payload modes
payload_modes = {
    "Complete Quaternion": [3, b"\x03"],  # Payload 3 for Timestamp, Quaternion, Free Acceleration
}

# Filtrage et variables globales
lowcut = 20.0
highcut = 25.0
order = 4
fs = 100.0  # Estimated sampling frequency (adjust if necessary)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

# Filter coefficients
sos = butter_bandpass(lowcut, highcut, fs, order)

min_interval_between_rebounds = 0.4  # seconds
min_no_threshold_time = 0.05  # seconds
n_samples_0_05s = int(fs * min_no_threshold_time)

# MODIFICATION : On va stocker les données de chaque capteur dans un dictionnaire
# La clé sera l'adresse MAC du capteur, la valeur un dictionnaire contenant toutes les listes et variables
devices_data = {}

for addr in device_addresses:
    devices_data[addr] = {
        "xs": [],
        "ys_x": [],
        "ys_y": [],
        "ys_z": [],
        "ys_x_filtered": [],
        "ys_y_filtered": [],
        "ys_z_filtered": [],
        "quat_w": [],
        "quat_x": [],
        "quat_y": [],
        "quat_z": [],
        "quat_w_filtered": [],
        "quat_x_filtered": [],
        "quat_y_filtered": [],
        "quat_z_filtered": [],
        "rebound_times": [],
        
        # Filtrage : états du filtre
        "accel_filter_state": np.zeros((sos.shape[0], 3, 2)),
        "quat_filter_state": np.zeros((sos.shape[0], 4, 2)),

        # Buffers et détection
        "quat_buffer": deque(),
        "condition_buffer": deque(),
        "in_interval": False,
        "interval_start_time": None,
        "last_rebound_time": -np.inf,
        "interval_data": [],

        # Temps initial pour chaque capteur
        "t0": None
    }

# Lock pour la synchro
data_lock = Lock()

def parse_channel_3_data(data_bytes):
    expected_size = 40
    if len(data_bytes) != expected_size:
        print(f"Unexpected data size: {len(data_bytes)} bytes. Expected size: {expected_size} bytes.")
        return None

    data_format = np.dtype([
        ('timestamp', np.uint32),    # 4 bytes
        ('quat_w', np.float32),      # 4 bytes
        ('quat_x', np.float32),      # 4 bytes
        ('quat_y', np.float32),      # 4 bytes
        ('quat_z', np.float32),      # 4 bytes
        ('accel_x', np.float32),     # 4 bytes
        ('accel_y', np.float32),     # 4 bytes
        ('accel_z', np.float32),     # 4 bytes
        ('dq_w', np.float32),        # 4 bytes
        ('dq_x', np.float32),        # 4 bytes
    ])
    data = np.frombuffer(data_bytes, dtype=data_format)
    return data

def sync_lists(*lists):
    """Synchronize all lists to have the same length as the shortest one."""
    min_len = min(len(lst) for lst in lists)
    return [lst[-min_len:] for lst in lists]

# MODIFICATION : On crée une closure ou une factory de callback pour chaque device
def make_notification_handler(address):
    def handle_medium_payload_notification(sender, data):
        # Cette fonction utilise les données pour l'adresse spécifique
        device = devices_data[address]

        formatted_data = parse_channel_3_data(data)
        if formatted_data is None:
            # Incorrect data, ignore this notification
            return

        timestamp = formatted_data['timestamp'][0]
        quaternion_w = formatted_data['quat_w'][0]
        quaternion_x = formatted_data['quat_x'][0]
        quaternion_y = formatted_data['quat_y'][0]
        quaternion_z = formatted_data['quat_z'][0]
        acceleration_x = formatted_data['accel_x'][0]
        acceleration_y = formatted_data['accel_y'][0]
        acceleration_z = formatted_data['accel_z'][0]

        current_time = time.perf_counter()
        if device["t0"] is None:
            device["t0"] = current_time
        elapsed_time = current_time - device["t0"]

        with data_lock:
            device["xs"].append(elapsed_time)
            device["ys_x"].append(acceleration_x)
            device["ys_y"].append(acceleration_y)
            device["ys_z"].append(acceleration_z)
            device["quat_w"].append(quaternion_w)
            device["quat_x"].append(quaternion_x)
            device["quat_y"].append(quaternion_y)
            device["quat_z"].append(quaternion_z)

            # Filter
            accel_sample = np.array([acceleration_x, acceleration_y, acceleration_z]).reshape(3, 1)
            quat_sample = np.array([quaternion_w, quaternion_x, quaternion_y, quaternion_z]).reshape(4, 1)

            accel_filtered, device["accel_filter_state"] = sosfilt(sos, accel_sample, zi=device["accel_filter_state"])
            quat_filtered, device["quat_filter_state"] = sosfilt(sos, quat_sample, zi=device["quat_filter_state"])

            device["ys_x_filtered"].append(accel_filtered[0, 0])
            device["ys_y_filtered"].append(accel_filtered[1, 0])
            device["ys_z_filtered"].append(accel_filtered[2, 0])
            device["quat_w_filtered"].append(quat_filtered[0, 0])
            device["quat_x_filtered"].append(quat_filtered[1, 0])
            device["quat_y_filtered"].append(quat_filtered[2, 0])
            device["quat_z_filtered"].append(quat_filtered[3, 0])

            max_len = int(fs * 10)
            for key in ["xs", "ys_x", "ys_y", "ys_z", "ys_x_filtered", "ys_y_filtered", "ys_z_filtered",
                        "quat_w", "quat_x", "quat_y", "quat_z",
                        "quat_w_filtered", "quat_x_filtered", "quat_y_filtered", "quat_z_filtered"]:
                device[key] = device[key][-max_len:]

        # Détection de rebond
        threshold = float(quat_threshold_value.get())

        max_abs_quat = max(
            abs(device["quat_w_filtered"][-1]),
            abs(device["quat_x_filtered"][-1]),
            abs(device["quat_y_filtered"][-1]),
            abs(device["quat_z_filtered"][-1])
        )

        condition = max_abs_quat > threshold
        device["quat_buffer"].append({
            'time': elapsed_time,
            'quat_w': device["quat_w_filtered"][-1],
            'quat_x': device["quat_x_filtered"][-1],
            'quat_y': device["quat_y_filtered"][-1],
            'quat_z': device["quat_z_filtered"][-1],
        })
        device["condition_buffer"].append(condition)

        buffer_size = int(fs * 2)
        if len(device["quat_buffer"]) > buffer_size:
            device["quat_buffer"].popleft()
        if len(device["condition_buffer"]) > buffer_size:
            device["condition_buffer"].popleft()

        if not device["in_interval"] and condition:
            device["interval_start_time"] = elapsed_time
            device["in_interval"] = True
            device["interval_data"] = [device["quat_buffer"][-1]]
        elif device["in_interval"]:
            device["interval_data"].append(device["quat_buffer"][-1])
            if not any(list(device["condition_buffer"])[-n_samples_0_05s:]):
                max_overall = max(
                    max(abs(d['quat_w']) for d in device["interval_data"]),
                    max(abs(d['quat_x']) for d in device["interval_data"]),
                    max(abs(d['quat_y']) for d in device["interval_data"]),
                    max(abs(d['quat_z']) for d in device["interval_data"]),
                )
                min_required = max_overall / 10.0
                quat_max_values = {
                    'quat_w': max(abs(d['quat_w']) for d in device["interval_data"]),
                    'quat_x': max(abs(d['quat_x']) for d in device["interval_data"]),
                    'quat_y': max(abs(d['quat_y']) for d in device["interval_data"]),
                    'quat_z': max(abs(d['quat_z']) for d in device["interval_data"]),
                }
                if all(v >= min_required for v in quat_max_values.values()):
                    if elapsed_time - device["last_rebound_time"] >= min_interval_between_rebounds:
                        device["rebound_times"].append(device["interval_start_time"])
                        device["last_rebound_time"] = device["interval_start_time"]
                        print(f"Rebound detected on {address} at {device['interval_start_time']:.2f} s")
                        sound.play()
                device["in_interval"] = False
                device["interval_start_time"] = None
                device["interval_data"] = []
        else:
            pass

    return handle_medium_payload_notification

def validate_threshold(*args):
    try:
        value = float(quat_threshold_value.get())
        if value <= 0:
            raise ValueError
    except ValueError:
        quat_threshold_value.set("0.002")

async def connect_device(address):
    device = await BleakScanner.find_device_by_address(address, timeout=20.0)
    if device is None:
        print(f"Device {address} not found.")
        return
    async with BleakClient(device) as client:
        handler = make_notification_handler(address)
        await client.start_notify(medium_payload_characteristic_uuid, handler)
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
            await client.stop_notify(medium_payload_characteristic_uuid)

async def main():
    tasks = [connect_device(addr) for addr in device_addresses]
    await asyncio.gather(*tasks)

def run_ble_client():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except Exception as e:
        print(f"Error in run_ble_client: {e}")

# Tkinter interface
root = tk.Tk()
root.title("Real-Time Rebound Detection")
root.geometry("1600x900")

quat_threshold_value = tk.StringVar(value="0.002")
quat_threshold_value.trace_add("write", validate_threshold)

frame_controls = ttk.Frame(root)
frame_controls.pack(side=tk.TOP, fill=tk.X)

quat_threshold_label = ttk.Label(frame_controls, text="Quaternion Threshold:")
quat_threshold_label.pack(side=tk.LEFT, padx=5, pady=5)

quat_threshold_entry = ttk.Entry(frame_controls, textvariable=quat_threshold_value, width=5)
quat_threshold_entry.pack(side=tk.LEFT, padx=5, pady=5)

frame_main = ttk.Frame(root)
frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_main.grid_rowconfigure(0, weight=1)
frame_main.grid_rowconfigure(1, weight=1)
frame_main.grid_columnconfigure(0, weight=1)
frame_main.grid_columnconfigure(1, weight=1)

frame_accel_raw = ttk.Frame(frame_main)
frame_accel_raw.grid(row=0, column=0, sticky="nsew")

frame_accel_filtered = ttk.Frame(frame_main)
frame_accel_filtered.grid(row=0, column=1, sticky="nsew")

frame_quat_raw = ttk.Frame(frame_main)
frame_quat_raw.grid(row=1, column=0, sticky="nsew")

frame_quat_filtered = ttk.Frame(frame_main)
frame_quat_filtered.grid(row=1, column=1, sticky="nsew")

fig_accel_raw = plt.Figure(figsize=(8, 4), dpi=100)
ax_accel_raw = fig_accel_raw.add_subplot(111)
canvas_accel_raw = FigureCanvasTkAgg(fig_accel_raw, master=frame_accel_raw)
canvas_accel_raw.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

fig_accel_filtered = plt.Figure(figsize=(8, 4), dpi=100)
ax_accel_filtered = fig_accel_filtered.add_subplot(111)
canvas_accel_filtered = FigureCanvasTkAgg(fig_accel_filtered, master=frame_accel_filtered)
canvas_accel_filtered.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

fig_quat_raw = plt.Figure(figsize=(8, 4), dpi=100)
ax_quat_raw = fig_quat_raw.add_subplot(111)
canvas_quat_raw = FigureCanvasTkAgg(fig_quat_raw, master=frame_quat_raw)
canvas_quat_raw.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

fig_quat_filtered = plt.Figure(figsize=(8, 4), dpi=100)
ax_quat_filtered = fig_quat_filtered.add_subplot(111)
canvas_quat_filtered = FigureCanvasTkAgg(fig_quat_filtered, master=frame_quat_filtered)
canvas_quat_filtered.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def animate_plot(i):
    with data_lock:
        ax_accel_raw.clear()
        ax_accel_filtered.clear()
        ax_quat_raw.clear()
        ax_quat_filtered.clear()

        # On va afficher les données de tous les capteurs sur les mêmes graphiques
        # Pour les différencier, on utilise des couleurs différentes
        colors = ["red", "green", "blue"]  # Ajuster si plus de devices
        for idx, addr in enumerate(device_addresses):
            device = devices_data[addr]

            xs = device["xs"]
            ys_x = device["ys_x"]
            ys_y = device["ys_y"]
            ys_z = device["ys_z"]
            ys_xf = device["ys_x_filtered"]
            ys_yf = device["ys_y_filtered"]
            ys_zf = device["ys_z_filtered"]
            qw = device["quat_w"]
            qx = device["quat_x"]
            qy = device["quat_y"]
            qz = device["quat_z"]
            qwf = device["quat_w_filtered"]
            qxf = device["quat_x_filtered"]
            qyf = device["quat_y_filtered"]
            qzf = device["quat_z_filtered"]
            rebounds = device["rebound_times"]

            if len(xs) == 0:
                continue

            # Synchronisation
            xs_raw_accel, ys_x_sync, ys_y_sync, ys_z_sync = sync_lists(xs, ys_x, ys_y, ys_z)
            xs_filt_accel, ys_xf_sync, ys_yf_sync, ys_zf_sync = sync_lists(xs, ys_xf, ys_yf, ys_zf)
            xs_raw_quat, qw_sync, qx_sync, qy_sync, qz_sync = sync_lists(xs, qw, qx, qy, qz)
            xs_filt_quat, qw_f_sync, qx_f_sync, qy_f_sync, qz_f_sync = sync_lists(xs, qwf, qxf, qyf, qzf)

            max_display_points = 1000
            step_raw_accel = max(1, len(xs_raw_accel) // max_display_points)
            step_filt_accel = max(1, len(xs_filt_accel) // max_display_points)
            step_raw_quat = max(1, len(xs_raw_quat) // max_display_points)
            step_filt_quat = max(1, len(xs_filt_quat) // max_display_points)

            xs_raw_accel_dec = xs_raw_accel[::step_raw_accel]
            ys_x_sync_dec = ys_x_sync[::step_raw_accel]
            ys_y_sync_dec = ys_y_sync[::step_raw_accel]
            ys_z_sync_dec = ys_z_sync[::step_raw_accel]

            xs_filt_accel_dec = xs_filt_accel[::step_filt_accel]
            ys_xf_sync_dec = ys_xf_sync[::step_filt_accel]
            ys_yf_sync_dec = ys_yf_sync[::step_filt_accel]
            ys_zf_sync_dec = ys_zf_sync[::step_filt_accel]

            xs_raw_quat_dec = xs_raw_quat[::step_raw_quat]
            qw_sync_dec = qw_sync[::step_raw_quat]
            qx_sync_dec = qx_sync[::step_raw_quat]
            qy_sync_dec = qy_sync[::step_raw_quat]
            qz_sync_dec = qz_sync[::step_raw_quat]

            xs_filt_quat_dec = xs_filt_quat[::step_filt_quat]
            qw_f_sync_dec = qw_f_sync[::step_filt_quat]
            qx_f_sync_dec = qx_f_sync[::step_filt_quat]
            qy_f_sync_dec = qy_f_sync[::step_filt_quat]
            qz_f_sync_dec = qz_f_sync[::step_filt_quat]

            c = colors[idx % len(colors)]

            # Raw accel
            ax_accel_raw.plot(xs_raw_accel_dec, ys_x_sync_dec, label=f"{addr} Accel X Raw", color=c, linestyle='-')
            ax_accel_raw.plot(xs_raw_accel_dec, ys_y_sync_dec, color=c, linestyle='--')
            ax_accel_raw.plot(xs_raw_accel_dec, ys_z_sync_dec, color=c, linestyle=':')

            # Filtered accel
            ax_accel_filtered.plot(xs_filt_accel_dec, ys_xf_sync_dec, label=f"{addr} Accel X Filt", color=c, linestyle='-')
            ax_accel_filtered.plot(xs_filt_accel_dec, ys_yf_sync_dec, color=c, linestyle='--')
            ax_accel_filtered.plot(xs_filt_accel_dec, ys_zf_sync_dec, color=c, linestyle=':')

            # Raw quat
            ax_quat_raw.plot(xs_raw_quat_dec, qw_sync_dec, label=f"{addr} QW Raw", color=c, linestyle='-')
            ax_quat_raw.plot(xs_raw_quat_dec, qx_sync_dec, color=c, linestyle='--')
            ax_quat_raw.plot(xs_raw_quat_dec, qy_sync_dec, color=c, linestyle=':')
            ax_quat_raw.plot(xs_raw_quat_dec, qz_sync_dec, color=c, linestyle='-.')

            # Filtered quat
            ax_quat_filtered.plot(xs_filt_quat_dec, qw_f_sync_dec, label=f"{addr} QW Filt", color=c, linestyle='-')
            ax_quat_filtered.plot(xs_filt_quat_dec, qx_f_sync_dec, color=c, linestyle='--')
            ax_quat_filtered.plot(xs_filt_quat_dec, qy_f_sync_dec, color=c, linestyle=':')
            ax_quat_filtered.plot(xs_filt_quat_dec, qz_f_sync_dec, color=c, linestyle='-.')

            # Ajout des lignes verticales pour les rebounds
            for t in rebounds:
                ax_accel_filtered.axvline(x=t, color=c, linewidth=1, linestyle='--')
                ax_quat_filtered.axvline(x=t, color=c, linewidth=1, linestyle='--')

        ax_accel_raw.set_ylim(-20, 20)
        ax_accel_raw.set_title("Raw Acceleration")
        ax_accel_raw.set_ylabel("Acceleration (m/s²)")
        ax_accel_raw.set_xlabel("Time (s)")
        ax_accel_raw.legend()
        ax_accel_raw.grid(True)

        ax_accel_filtered.set_ylim(-5, 5)
        ax_accel_filtered.set_title("Filtered Acceleration")
        ax_accel_filtered.set_ylabel("Acceleration (m/s²)")
        ax_accel_filtered.set_xlabel("Time (s)")
        ax_accel_filtered.legend()
        ax_accel_filtered.grid(True)

        ax_quat_raw.set_ylim(-1, 1)
        ax_quat_raw.set_title("Raw Quaternions")
        ax_quat_raw.set_ylabel("Quaternion Value")
        ax_quat_raw.set_xlabel("Time (s)")
        ax_quat_raw.legend()
        ax_quat_raw.grid(True)

        ax_quat_filtered.set_ylim(-0.1, 0.1)
        ax_quat_filtered.set_title("Filtered Quaternions")
        ax_quat_filtered.set_ylabel("Quaternion Value")
        ax_quat_filtered.set_xlabel("Time (s)")
        ax_quat_filtered.legend()
        ax_quat_filtered.grid(True)

ani_accel_raw = animation.FuncAnimation(fig_accel_raw, animate_plot, interval=400, cache_frame_data=False)
ani_accel_filtered = animation.FuncAnimation(fig_accel_filtered, animate_plot, interval=400, cache_frame_data=False)
ani_quat_raw = animation.FuncAnimation(fig_quat_raw, animate_plot, interval=400, cache_frame_data=False)
ani_quat_filtered = animation.FuncAnimation(fig_quat_filtered, animate_plot, interval=400, cache_frame_data=False)

def on_closing():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

ble_thread = threading.Thread(target=run_ble_client)
ble_thread.daemon = True
ble_thread.start()

root.mainloop()
