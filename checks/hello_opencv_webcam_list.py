import cv2

def list_webcams():
    """Detect and list all available webcams."""
    print("Scanning for available webcams...")
    index = 0
    available_cams = []
    
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            available_cams.append(index)
            print(f"✅ Webcam found at index {index}")
        cap.release()
        index += 1

    if not available_cams:
        print("❌ No webcams detected.")
    else:
        print("\nAvailable webcams:", available_cams)

if __name__ == "__main__":
    list_webcams()
