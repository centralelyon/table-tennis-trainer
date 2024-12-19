import cv2

def get_camera_details(index):
    """Retrieve and print details of a specific webcam."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    # Fetch properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return {
        "Index": index,
        "Resolution": f"{int(width)}x{int(height)}",
        "FPS": int(fps) if fps > 0 else "Unknown"
    }

def list_webcams():
    """Detect and list all available webcams with details."""
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
        print(f"\n {len(available_cams)} webcams found: {available_cams}\n")

        print("Fetching details of each webcam...")
        for cam_index in available_cams:
            details = get_camera_details(cam_index)
            if details:
                print(f"- Webcam {details['Index']}:")
                print(f"  Resolution: {details['Resolution']}")
                print(f"  FPS: {details['FPS']}")

if __name__ == "__main__":
    list_webcams()
