import cv2
import numpy as np
import mss
import time
import keyboard
from vgamepad import VX360Gamepad

# Screen capture region
monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

# Global variables
parking_spot = []
drawing = False
gamepad = VX360Gamepad()

# Mouse callback for drawing parking area
def draw_parking_area(event, x, y, flags, param):
    global parking_spot, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        parking_spot = [(x, y)]  # Start point
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        parking_spot.append((x, y))  # End point
        drawing = False
        print(f"Selected Parking Spot: {parking_spot}")

# Capture screen
def capture_screen():
    with mss.mss() as sct:
        screen = np.array(sct.grab(monitor))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    return screen

# **Detect obstacles using edge detection**
def detect_obstacles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # **Increase Canny threshold to reduce noise**
    edges = cv2.Canny(gray, 150, 250)  # Was (100, 200) - Now (150, 250)
    
    # **Limit ROI to only bottom 30% of screen**
    height, width = edges.shape
    roi = edges[int(height * 0.7):, :]  # Before 60%, now 70%

    # **Count white pixels (edges)**
    edge_count = np.count_nonzero(roi)

    # Debug view
    cv2.imshow("Edges", edges)
    cv2.waitKey(1)

    # **Adjust edge threshold (was 3000, now 5000)**
    return edge_count > 5000


# **Control the car**
def move_vehicle():
    global parking_spot
    if len(parking_spot) < 2:
        print("Parking spot not selected!")
        return

    print("Navigating to parking spot...")

    # Get parking center
    x1, y1 = parking_spot[0]
    x2, y2 = parking_spot[1]
    parking_center_x = (x1 + x2) // 2

    # Define screen center
    screen_center_x = monitor["width"] // 2

    # **Main driving loop**
    start_time = time.time()
    while time.time() - start_time < 5:  # Move for 5 seconds max
        frame = capture_screen()
        # Check if the vehicle is within the parking area
        if (x1 <= parking_center_x <= x2) and (y1 <= frame.shape[0] <= y2):
            print("Car stopped at the parking area!")
            gamepad.right_trigger_float(0.0)  # Stop accelerating
            gamepad.update()
            break
        frame = capture_screen()
        if detect_obstacles(frame):
            print("Obstacle detected! Braking...")
            gamepad.right_trigger_float(0.0)  # Stop accelerating
            gamepad.update()
            time.sleep(1)  # Pause to avoid crashing
            break

        # Steering logic
        if parking_center_x < screen_center_x - 50:
            steering = -0.5  # Turn left
            print("Turning LEFT")
        elif parking_center_x > screen_center_x + 50:
            steering = 0.5  # Turn right
            print("Turning RIGHT")
        else:
            steering = 0.0  # Go straight
            print("Going STRAIGHT")

        # Apply gradual acceleration
        gamepad.right_trigger_float(0.6)
        gamepad.right_trigger_float(0.6)
        gamepad.left_joystick_float(steering, 0.0)
        gamepad.update()

        time.sleep(0.5)  # Small movement step

    # Stop the car after reaching
    gamepad.right_trigger_float(0.0)
    gamepad.left_joystick_float(0.0, 0.0)
    gamepad.update()
    print("Car Parked Successfully!")

# **Main function**
def auto_park():
    global parking_spot

    # Step 1: Let user draw parking area
    frame = capture_screen()
    cv2.imshow("Select Parking Spot", frame)
    cv2.setMouseCallback("Select Parking Spot", draw_parking_area)

    print("Click and drag to select the parking area. Press 'Enter' when done.")

    while True:
        temp_frame = frame.copy()
        if len(parking_spot) == 2:
            cv2.rectangle(temp_frame, parking_spot[0], parking_spot[1], (0, 255, 0), 2)

        cv2.imshow("Select Parking Spot", temp_frame)
        cv2.waitKey(1)  # Prevent window freezing

        if keyboard.is_pressed("enter") and len(parking_spot) == 2:
            break
        if keyboard.is_pressed("esc"):
            print("Parking canceled.")
            return

    cv2.destroyAllWindows()

    # Step 2: Move the car to the parking area
    move_vehicle()

# **Run script**
if __name__ == "__main__":
    auto_park()




# import mss
# import numpy as np
# import cv2
# from ultralytics import YOLO

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# # Known real-world widths (in meters)
# KNOWN_WIDTHS = {
#     "car": 1.8,
#     "bus": 2.5,
#     "truck": 2.5,
#     "person": 0.5  # Average width of a person
# }

# FOCAL_LENGTH = 700  # Approximate focal length (needs calibration)

# # Function to capture a fixed region of the screen
# def capture_screen():
#     with mss.mss() as sct:
#         monitor = {"top": 40, "left": 0, "width": 800, "height": 600}  # GTA 5 screen region
#         screenshot = sct.grab(monitor)
#         img = np.array(screenshot)
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
#         return img

# # Function to estimate distance
# def estimate_distance(label, pixel_width):
#     if label in KNOWN_WIDTHS:
#         return (FOCAL_LENGTH * KNOWN_WIDTHS[label]) / pixel_width
#     return None

# # Main loop for detection
# while True:
#     frame = capture_screen()
#     results = model(frame)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
#             cls = int(box.cls[0])  # Class index
#             label = model.names[cls]  # Object class name

#             # Detect vehicles and pedestrians
#             if label in ["car", "bus", "truck", "person"]:
#                 pixel_width = x2 - x1  # Width of the bounding box
#                 distance = estimate_distance(label, pixel_width)

#                 # Draw bounding box and distance text
#                 color = (0, 255, 0) if label != "person" else (255, 0, 0)  # Green for vehicles, Blue for pedestrians
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 text = f"{label} {distance:.2f}m" if distance else label
#                 cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # Display the result
#     cv2.imshow("GTA V - Vehicle & Pedestrian Detection", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cv2.destroyAllWindows()
