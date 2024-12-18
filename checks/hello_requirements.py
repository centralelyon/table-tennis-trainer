def check_requirements():
    print("Checking requirements...\n")
    
    try:
        import asyncio
        print("✅ asyncio imported successfully (built-in).")
    except ImportError as e:
        print(f"❌ Failed to import asyncio: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully. Version:", np.__version__)
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully. Version:", pd.__version__)
    except ImportError as e:
        print(f"❌ Failed to import pandas: {e}")
    
    try:
        from PIL import Image, ImageTk
        print("✅ Pillow (PIL) imported successfully. Version:", Image.__version__)
    except ImportError as e:
        print(f"❌ Failed to import Pillow: {e}")
    
    try:
        import pygame
        print("✅ pygame imported successfully. Version:", pygame.ver)
    except ImportError as e:
        print(f"❌ Failed to import pygame: {e}")
    
    try:
        from bleak import BleakClient, BleakScanner
        print("✅ bleak imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import bleak: {e}")
    
    try:
        import tkinter as tk
        print("✅ tkinter imported successfully (built-in).")
    except ImportError as e:
        print(f"❌ Failed to import tkinter: {e}")
    
    try:
        import cv2
        print("✅ opencv-python (cv2) imported successfully. Version:", cv2.__version__)
    except ImportError as e:
        print(f"❌ Failed to import opencv-python: {e}")
    
    try:
        import matplotlib
        print("✅ matplotlib imported successfully. Version:", matplotlib.__version__)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("✅ matplotlib's TkAgg backend imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import matplotlib or its TkAgg backend: {e}")
    
    try:
        from scipy.signal import butter, sosfilt
        print("✅ scipy.signal imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import scipy.signal: {e}")
    

if __name__ == "__main__":
    check_requirements()
