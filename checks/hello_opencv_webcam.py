import cv2

def show_webcam(nb=0):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        print("Webcam opened successfully. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            cv2.imshow('Webcam Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_webcam(1)

