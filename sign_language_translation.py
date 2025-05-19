import cv2
import numpy as np

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        if b * c == 0:
            continue

        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / np.pi)

        if angle < 90 and d > 10000:  # Strong filter to avoid noise
            finger_count += 1
            cv2.circle(drawing, far, 5, (0, 255, 0), -1)

    return finger_count + 1 if finger_count > 0 else 0

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

print("📷 Webcam started. Show your hand in the green box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert to HSV and apply skin mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    text = "No hand"
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 4000:
            drawing = np.zeros(roi.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (255, 255, 255), 2)

            fingers = count_fingers(contour, drawing)

            if fingers == 0:
                text = "Fist ✊"
            elif fingers == 1:
                text = "OK 👍"
            elif fingers == 2:
                text = "Peace ✌"
            elif fingers >= 4:
                text = "Hi-Fi ✋"
            else:
                text = f"{fingers} fingers"

            cv2.imshow("Hand Detection", drawing)

    cv2.putText(frame, f'Sign: {text}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()