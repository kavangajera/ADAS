import mss
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 models - one for general objects, one specifically for traffic lights
model = YOLO("yolov8n.pt")  # General detection model

# For better traffic light detection from distance, consider using a model with higher resolution
# If available, use a model fine-tuned for traffic lights
# traffic_light_model = YOLO("yolov8n-traffic.pt")  # Specialized model if available
traffic_light_model = YOLO("yolov8n.pt")  # Using same model if specialized one isn't available

# Known real-world widths (in meters)
KNOWN_WIDTHS = {
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.5,
    "person": 0.5,  # Average width of a person
    "traffic light": 0.3  # Typical width of a traffic light housing
}

FOCAL_LENGTH = 700  # Approximate focal length (needs calibration)

# Function to capture a fixed region of the screen
def capture_screen():
    with mss.mss() as sct:
        # Capture a wider and taller area to see more of the environment
        monitor = {"top": 40, "left": 0, "width": 800, "height": 600}  # Increased capture area
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
        return img

# Function to estimate distance
def estimate_distance(label, pixel_width):
    if label in KNOWN_WIDTHS:
        return (FOCAL_LENGTH * KNOWN_WIDTHS[label]) / pixel_width
    return None

# Enhanced traffic light detection function
def detect_traffic_lights(frame):
    # 1. Run YOLOv8 detection with higher confidence threshold for general objects
    results = traffic_light_model(frame, conf=0.25)  # Lower threshold to catch distant traffic lights
    
    traffic_lights = []
    
    # 2. Process YOLO detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = traffic_light_model.names[cls]
            confidence = float(box.conf[0])
            
            # Check if the detection is a traffic light
            if label == "traffic light":
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Make bounding box integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate distance based on pixel width
                pixel_width = x2 - x1
                distance = estimate_distance("traffic light", pixel_width)
                
                # Analyze color - expand bounding box slightly for better color detection
                expand_factor = 0.1
                height = y2 - y1
                width = x2 - x1
                
                # Expand box slightly
                x1_exp = max(0, int(x1 - width * expand_factor))
                y1_exp = max(0, int(y1 - height * expand_factor))
                x2_exp = min(frame.shape[1], int(x2 + width * expand_factor))
                y2_exp = min(frame.shape[0], int(y2 + height * expand_factor))
                
                # Get the traffic light color
                color, color_conf = determine_traffic_light_color(frame, (x1_exp, y1_exp, x2_exp, y2_exp))
                
                # Add to detected traffic lights list with distance info
                if color is not None:
                    traffic_lights.append({
                        "bbox": (x1, y1, x2, y2),
                        "type": color,
                        "confidence": (confidence + color_conf) / 2,
                        "distance": distance if distance is not None else "unknown"
                    })
    
    # 3. Check the upper portion of the frame specifically for traffic lights
    # This helps detect distant traffic lights that might be small
    height, width = frame.shape[:2]
    upper_region = frame[0:int(height/2), :]
    
    # Run detection specifically on the upper region
    upper_results = traffic_light_model(upper_region, conf=0.2)  # Even lower threshold for distance
    
    for result in upper_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = traffic_light_model.names[cls]
            confidence = float(box.conf[0])
            
            if label == "traffic light":
                # Adjust coordinates to match the original frame
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate distance
                pixel_width = x2 - x1
                distance = estimate_distance("traffic light", pixel_width)
                
                # Check color
                color, color_conf = determine_traffic_light_color(upper_region, (x1, y1, x2, y2))
                
                # Only add if it's not already in the list
                if color is not None:
                    new_detection = True
                    for light in traffic_lights:
                        x1_existing, y1_existing, x2_existing, y2_existing = light["bbox"]
                        
                        # Check if this is the same traffic light
                        existing_center = ((x1_existing + x2_existing) / 2, (y1_existing + y2_existing) / 2)
                        new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        
                        distance_between = np.sqrt((existing_center[0] - new_center[0])**2 + 
                                                  (existing_center[1] - new_center[1])**2)
                        
                        if distance_between < 50:  # If centers are close, it's the same light
                            new_detection = False
                            break
                    
                    if new_detection:
                        traffic_lights.append({
                            "bbox": (x1, y1, x2, y2),
                            "type": color,
                            "confidence": (confidence + color_conf) / 2,
                            "distance": distance if distance is not None else "unknown"
                        })
    
    return traffic_lights

# Function to determine traffic light color
def determine_traffic_light_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    # Extract traffic light region
    light_roi = frame[y1:y2, x1:x2]
    
    if light_roi.size == 0:
        return None, 0.0  # Invalid ROI
    
    # Convert to HSV for better color detection
    hsv_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for traffic light colors
    # Broader ranges for more robust detection at distance
    red_lower1 = np.array([0, 70, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 70, 70])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([15, 70, 70])
    yellow_upper = np.array([35, 255, 255])
    
    green_lower = np.array([35, 70, 70])
    green_upper = np.array([85, 255, 255])
    
    # Create masks for each color
    red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    yellow_mask = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
    
    # Count pixels of each color
    red_pixel_count = cv2.countNonZero(red_mask)
    yellow_pixel_count = cv2.countNonZero(yellow_mask)
    green_pixel_count = cv2.countNonZero(green_mask)
    
    # Calculate ratios and density
    total_pixels = light_roi.shape[0] * light_roi.shape[1]
    red_ratio = red_pixel_count / total_pixels if total_pixels > 0 else 0
    yellow_ratio = yellow_pixel_count / total_pixels if total_pixels > 0 else 0
    green_ratio = green_pixel_count / total_pixels if total_pixels > 0 else 0
    
    # Lower threshold for distant traffic lights
    color_threshold = 0.03  # Reduce threshold for distant detection
    
    # Determine the dominant color
    max_ratio = max(red_ratio, yellow_ratio, green_ratio)
    confidence = max_ratio * 2  # Scale confidence
    
    # Thresholding to avoid false positives
    if max_ratio < color_threshold:
        return None, 0.0
    
    if red_ratio == max_ratio:
        return "red", min(confidence, 1.0)
    elif yellow_ratio == max_ratio:
        return "yellow", min(confidence, 1.0)
    elif green_ratio == max_ratio:
        return "green", min(confidence, 1.0)
    else:
        return None, 0.0

# Create warning icons
def create_warning_icons():
    # Create a brake warning icon (red circle with "BRAKE" text)
    brake_icon = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(brake_icon, (50, 50), 45, (0, 0, 255), -1)
    cv2.putText(brake_icon, "BRAKE", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create a slow down warning icon (yellow triangle)
    slow_icon = np.zeros((100, 100, 3), dtype=np.uint8)
    triangle_pts = np.array([[50, 10], [10, 90], [90, 90]], np.int32)
    cv2.fillPoly(slow_icon, [triangle_pts], (0, 255, 255))
    cv2.putText(slow_icon, "SLOW", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return brake_icon, slow_icon

# Create warning icons
brake_warning, slow_warning = create_warning_icons()

# Initialize variables
previous_frame = None
last_warning_time = 0
warning_duration = 1.5  # seconds to display warning
debug_mode = True  # Set to True to see additional debug information

# Keep track of traffic light detections over multiple frames to reduce false positives
traffic_light_history = []
history_length = 3  # Number of frames to track
confidence_threshold = 0.4  # Lower confidence threshold for distant detection

# Configurable parameters for distance detection
min_traffic_light_distance_warning = 40.0  # Start warning at this distance (meters)
critical_traffic_light_distance = 20.0  # Critical warning threshold

# Main loop
while True:
    start_time = time.time()
    frame = capture_screen()
    
    # Get traffic light detections
    traffic_lights = detect_traffic_lights(frame)
    
    # Add current detections to history
    traffic_light_history.append(traffic_lights)
    if len(traffic_light_history) > history_length:
        traffic_light_history.pop(0)
    
    # Get consistent traffic lights across frames
    consistent_traffic_lights = []
    
    if len(traffic_light_history) == history_length:
        # Analyze lights across frames for consistency
        for current_light in traffic_lights:
            matches = 1
            total_confidence = current_light["confidence"]
            
            # Check for matches in previous frames
            for prev_idx in range(len(traffic_light_history) - 1):
                prev_lights = traffic_light_history[prev_idx]
                best_match = None
                best_match_distance = float('inf')
                
                for prev_light in prev_lights:
                    # Calculate centers
                    current_center = (
                        (current_light["bbox"][0] + current_light["bbox"][2]) / 2,
                        (current_light["bbox"][1] + current_light["bbox"][3]) / 2
                    )
                    prev_center = (
                        (prev_light["bbox"][0] + prev_light["bbox"][2]) / 2,
                        (prev_light["bbox"][1] + prev_light["bbox"][3]) / 2
                    )
                    
                    # Calculate distance between centers
                    center_distance = np.sqrt(
                        (current_center[0] - prev_center[0])**2 + 
                        (current_center[1] - prev_center[1])**2
                    )
                    
                    # Find closest match
                    if center_distance < best_match_distance and center_distance < 60:  # Increased threshold
                        best_match_distance = center_distance
                        best_match = prev_light
                
                # If we found a match
                if best_match is not None:
                    matches += 1
                    total_confidence += best_match["confidence"]
            
            # Calculate average confidence
            avg_confidence = total_confidence / matches
            
            # If light is consistent across majority of frames
            if matches >= 2 and avg_confidence > confidence_threshold:
                current_light["confidence"] = avg_confidence
                consistent_traffic_lights.append(current_light)
    
    # Detect general objects for distance warnings
    results = model(frame)
    min_distance = float('inf')
    closest_object = None
    
    # Process general objects (vehicles, pedestrians)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Only process vehicles and pedestrians for distance warning
            if label in ["car", "bus", "truck", "person"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                pixel_width = x2 - x1  # Width of the bounding box
                distance = estimate_distance(label, pixel_width)
                
                if distance is not None and distance < min_distance:
                    min_distance = distance
                    closest_object = label
                
                # Draw bounding box and distance text
                color = (0, 255, 0) if label != "person" else (255, 0, 0)  # Green for vehicles, Blue for pedestrians
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {distance:.2f}m" if distance else label
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Process traffic light detections
    red_lights = []
    
    for light in consistent_traffic_lights:
        x1, y1, x2, y2 = light["bbox"]
        
        # Determine color for bounding box based on traffic light state
        if light["type"] == "red":
            color = (0, 0, 255)  # Red
            red_lights.append(light)
        elif light["type"] == "yellow":
            color = (0, 255, 255)  # Yellow
        elif light["type"] == "green":
            color = (0, 255, 0)  # Green
        else:
            color = (255, 255, 255)  # White for unknown
        
        # Draw a box around the traffic light
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence and distance
        distance_info = f" {light['distance']:.1f}m" if isinstance(light['distance'], (int, float)) else ""
        label = f"TRAFFIC {light['type'].upper()} ({light['confidence']:.2f}){distance_info}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Check for traffic light warnings (highest priority)
    show_red_light_warning = False
    if red_lights:
        for red_light in red_lights:
            # If distance is known, use it for warning intensity
            if isinstance(red_light['distance'], (int, float)):
                distance = red_light['distance']
                
                # Update warning text based on distance
                if distance < critical_traffic_light_distance:
                    warning_text = f"STOP: RED LIGHT AHEAD! ({distance:.1f}m)"
                    warning_color = (0, 0, 255)  # Red
                    show_red_light_warning = True
                elif distance < min_traffic_light_distance_warning:
                    warning_text = f"Warning: Red Light Ahead ({distance:.1f}m)"
                    warning_color = (0, 165, 255)  # Orange
                    show_red_light_warning = True
            else:
                # If distance unknown but confidence is high
                if red_light['confidence'] > 0.7:
                    warning_text = "Warning: Red Light Detected"
                    warning_color = (0, 165, 255)  # Orange
                    show_red_light_warning = True
        
        if show_red_light_warning:
            cv2.putText(frame, warning_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2) # type: ignore
            
            # Place brake warning icon in the bottom right
            h, w = frame.shape[:2]
            icon_h, icon_w = brake_warning.shape[:2]
            frame[h-icon_h-20:h-20, w-icon_w-20:w-20] = brake_warning
            
            # Update last warning time
            last_warning_time = time.time()
    
    # Check for distance warnings (vehicles and pedestrians)
    if min_distance != float('inf') and min_distance <= 3.8 and not show_red_light_warning:
        object_type = "pedestrian" if closest_object == "person" else "vehicle"
        
        # Determine warning severity based on distance
        if min_distance <= 2.0:
            warning_text = f"DANGER: {object_type.upper()} very close ({min_distance:.2f}m)!"
            warning_color = (0, 0, 255)  # Red
            # Place brake warning icon
            h, w = frame.shape[:2]
            icon_h, icon_w = brake_warning.shape[:2]
            frame[h-icon_h-20:h-20, w-icon_w-20:w-20] = brake_warning
        else:
            warning_text = f"Warning: {object_type} ahead ({min_distance:.2f}m)"
            warning_color = (0, 165, 255)  # Orange
            # Place slow down warning icon
            h, w = frame.shape[:2]
            icon_h, icon_w = slow_warning.shape[:2]
            frame[h-icon_h-20:h-20, w-icon_w-20:w-20] = slow_warning
        
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2)
        
        # Update last warning time
        last_warning_time = time.time()
    
    # Display debug info
    if debug_mode:
        debug_text = f"Traffic lights detected: {len(consistent_traffic_lights)}/{len(traffic_lights)}"
        cv2.putText(frame, debug_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display traffic light coordinates
        for i, light in enumerate(consistent_traffic_lights):
            x1, y1, x2, y2 = light["bbox"]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            distance_info = f" {light['distance']:.1f}m" if isinstance(light['distance'], (int, float)) else ""
            conf_text = f"Light {i+1}: ({center_x},{center_y}) {light['type']}{distance_info} conf:{light['confidence']:.2f}"
            cv2.putText(frame, conf_text, (10, frame.shape[0] - 30 - i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calculate and display FPS
    fps = 1.0 / (time.time() - start_time)
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow("GTA V - Advanced Detection System", frame)
    
    # Update previous frame for next iteration
    previous_frame = frame.copy()
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()