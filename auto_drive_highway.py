import cv2
import numpy as np
import os
import time
import keyboard
import mss
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from collections import Counter

# Define constants
MONITOR = {"top": 40, "left": 0, "width": 800, "height": 600}
DATASET_DIR = "gta5_driving_dataset"
TRAINING_DATA_CSV = "training_data.csv"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MODEL_PATH = "gta5_driving_model.h5"

# Create necessary directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Key mappings
KEY_MAP = {
    'w': 0,  # Forward
    'a': 1,  # Left
    'd': 2,  # Right
    's': 3   # Brake
}

# Part 1: Data Collection Functions
def capture_screen():
    """Capture screen using mss"""
    with mss.mss() as sct:
        img = np.array(sct.grab(MONITOR))
        # Convert to BGR (OpenCV format) from BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def process_image(img):
    """Process the image for the model input"""
    # Resize to a smaller dimension to reduce computational load
    img = cv2.resize(img, (160, 120))
    # Convert to grayscale to simplify the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    img = img / 255.0
    return img

def collect_data(total_samples_per_class=1000, preview=True, resume=True):
    """
    Collect data for each of the four actions
    
    Parameters:
    total_samples_per_class (int): Target number of samples for each key class
    preview (bool): Whether to show preview window
    resume (bool): Whether to resume from existing dataset
    """
    # Initialize counters for each key
    key_counts = {'w': 0, 'a': 0, 'd': 0, 's': 0}
    
    # Initialize or load existing dataframe
    csv_path = os.path.join(DATASET_DIR, TRAINING_DATA_CSV)
    
    if resume and os.path.exists(csv_path):
        # Load existing data and count samples for each key
        existing_df = pd.read_csv(csv_path)
        print(f"Found existing dataset with {len(existing_df)} total samples")
        
        # Count existing samples for each key
        for key in KEY_MAP:
            key_count = len(existing_df[existing_df['key_pressed'] == key])
            key_counts[key] = key_count
            print(f"Found {key_count} existing samples for key '{key}'")
        
        # Initialize dataframe with existing data
        df = existing_df
    else:
        # Initialize a new dataframe
        columns = ['image_path', 'key_pressed', 'key_code', 'pressed_time']
        df = pd.DataFrame(columns=columns)
        print("Starting new dataset collection")
    
    # Calculate remaining samples needed
    remaining_counts = {key: max(0, total_samples_per_class - count) for key, count in key_counts.items()}
    
    # If all keys have enough samples, exit
    if all(count == 0 for count in remaining_counts.values()):
        print("All keys already have the required number of samples.")
        return df
    
    print("Starting data collection. Press 'q' to quit.")
    print("Drive in GTA and press W, A, S, or D to record data")
    print(f"Target: {total_samples_per_class} samples per key")
    print(f"Remaining samples needed: W: {remaining_counts['w']}, A: {remaining_counts['a']}, "
          f"S: {remaining_counts['s']}, D: {remaining_counts['d']}")
    
    while any(remaining_counts.values()):
        img = capture_screen()
        processed_img = process_image(img)
        
        # Display real-time info
        if preview:
            # Calculate progress for each key
            current_counts = {key: key_counts[key] for key in KEY_MAP}
            status_text = f"Progress - W: {current_counts['w']}/{total_samples_per_class}, " \
                          f"A: {current_counts['a']}/{total_samples_per_class}, " \
                          f"S: {current_counts['s']}/{total_samples_per_class}, " \
                          f"D: {current_counts['d']}/{total_samples_per_class}"
            
            cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('GTA5 Screen Capture', img)
            
            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Check key presses
        for key in KEY_MAP:
            if keyboard.is_pressed(key) and remaining_counts[key] > 0:
                # Record the key press time
                start_time = time.time()
                
                # Wait until key is released to measure duration
                while keyboard.is_pressed(key):
                    time.sleep(0.01)
                
                # Calculate pressed time
                pressed_time = time.time() - start_time
                
                # Save image with timestamp as filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{key}_{timestamp}.jpg"
                image_path = os.path.join(IMAGE_DIR, filename)
                
                # Save original image
                cv2.imwrite(image_path, img)
                
                # Add entry to dataframe
                new_row = {
                    'image_path': image_path,
                    'key_pressed': key,
                    'key_code': KEY_MAP[key],
                    'pressed_time': pressed_time
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Update counters
                key_counts[key] += 1
                remaining_counts[key] -= 1
                
                print(f"Captured {key} press ({key_counts[key]}/{total_samples_per_class}), "
                      f"held for {pressed_time:.2f}s, {remaining_counts[key]} more needed")
                
                # Short sleep to avoid duplicate captures
                time.sleep(0.1)
    
    # Save dataframe to CSV
    df.to_csv(csv_path, index=False)
    print(f"Data collection completed. Dataset saved to {csv_path}")
    print(f"Total samples per key: W: {key_counts['w']}, A: {key_counts['a']}, "
          f"S: {key_counts['s']}, D: {key_counts['d']}")
    
    # Clean up
    cv2.destroyAllWindows()
    return df

# Part 2: Model Training

def build_cnn_model(input_shape, num_classes=4): # total params : 128,328
    """Build a CNN model for driving classification"""
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the results to feed into dense layer
        Flatten(),
        
        # Dense layers
        Dense(256, activation='relu'),
        Dropout(0.5),  # Add dropout to prevent overfitting
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output layer (4 classes: forward, left, right, brake)
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data(csv_path):
    """Load images and labels from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Initialize lists to store data
    X = []
    y = []
    
    print(f"Loading {len(df)} samples from dataset...")
    
    for _, row in df.iterrows():
        # Load and process image
        img_path = row['image_path']
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            processed_img = process_image(img)
            X.append(processed_img)
            
            # Get label
            y.append(row['key_code'])
        else:
            print(f"Warning: Image file not found: {img_path}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Successfully loaded {len(X)} samples")
    
    # Reshape X for CNN input (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)
    
    return X, y, num_classes

def train_model(csv_path=os.path.join(DATASET_DIR, TRAINING_DATA_CSV)):
    """Train the CNN model using collected data"""
    # Load and preprocess data
    X, y, num_classes = load_and_preprocess_data(csv_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    input_shape = (120, 160, 1)  # Grayscale image shape
    model = build_cnn_model(input_shape, num_classes=num_classes)
    
    # Display model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model, history

# Part 3: Testing and Validation

def test_model_realtime(model_path=MODEL_PATH, run_time=60):
    """Test the model in real-time on GTA 5"""
    # Load the trained model
    from tensorflow.keras.models import load_model # type: ignore
    model = load_model(model_path)
    
    print(f"Starting real-time testing for {run_time} seconds...")
    print("Press 'q' to quit early")
    
    start_time = time.time()
    frame_count = 0
    
    # For tracking predictions
    action_map = {0: 'W (Forward)', 1: 'A (Left)', 2: 'D (Right)', 3: 'S (Brake)'}
    predictions = []
    
    while time.time() - start_time < run_time:
        # Capture and process screen
        img = capture_screen()
        processed_img = process_image(img)
        
        # Reshape for prediction
        input_img = processed_img.reshape(1, 120, 160, 1)
        
        # Make prediction
        prediction = model.predict(input_img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Store prediction
        predictions.append(predicted_class)
        
        # Display prediction on screen
        prediction_text = f"Action: {action_map[predicted_class]} ({confidence:.2f})" # type: ignore
        cv2.putText(img, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('GTA5 AI Driver', img)
        
        frame_count += 1
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\nTesting completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / min(time.time() - start_time, run_time):.1f}")
    print(f"Action distribution: W: {predictions.count(0)}, A: {predictions.count(1)}, D: {predictions.count(2)}, S: {predictions.count(3)}")

# Part 4: Advanced feature - Auto-drive mode

def auto_drive(model_path=MODEL_PATH, run_time=300):
    """Autonomous driving mode that actually controls the game"""
    # Load the trained model
    from tensorflow.keras.models import load_model # type: ignore
    model = load_model(model_path)
    
    print(f"Starting autonomous driving for {run_time} seconds...")
    print("Press 'q' to quit early")
    
    start_time = time.time()
    frame_count = 0
    
    # For tracking predictions
    action_map = {0: 'W', 1: 'A', 2: 'D', 3: 'S'}
    prediction_history = []  # Keep history for smoothing predictions
    
    # To avoid jerky movement
    last_action = None
    action_hold_frames = 0
    
    while time.time() - start_time < run_time:
        # Capture and process screen
        img = capture_screen()
        processed_img = process_image(img)
        
        # Reshape for prediction
        input_img = processed_img.reshape(1, 120, 160, 1)
        
        # Make prediction
        prediction = model.predict(input_img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Use prediction history for smoothing (avoid jerky control)
        prediction_history.append(predicted_class)
        if len(prediction_history) > 5:
            prediction_history.pop(0)
        
        # Get most common prediction in recent history
        smoothed_action = Counter(prediction_history).most_common(1)[0][0]
        
        # Execute action in game if confidence is high enough
        action_key = action_map[smoothed_action]
        
        # Release all keys first - make sure we don't have multiple keys pressed
        for key in KEY_MAP:
            if keyboard.is_pressed(key):
                keyboard.release(key)
        
        # Press the predicted key
        if confidence > 0.6:  # Only act if confidence is high enough
            keyboard.press(action_key.lower())
            
            # Display action on screen
            action_text = f"Executing: {action_key} ({confidence:.2f})"
            color = (0, 255, 0)  # Green
        else:
            # If low confidence, don't do anything
            action_text = f"Low confidence: {action_key} ({confidence:.2f})"
            color = (0, 0, 255)  # Red
            
        cv2.putText(img, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('GTA5 AI Driver - Autonomous Mode', img)
        
        frame_count += 1
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Make sure to release all keys before quitting
            for key in KEY_MAP:
                keyboard.release(key)
            break
    
    # Make sure to release all keys before quitting
    for key in KEY_MAP:
        keyboard.release(key)
    
    # Clean up
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\nAutonomous driving completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / min(time.time() - start_time, run_time):.1f}")

# Data statistics function
def show_dataset_stats():
    """Display statistics about the collected dataset"""
    csv_path = os.path.join(DATASET_DIR, TRAINING_DATA_CSV)
    if not os.path.exists(csv_path):
        print("No dataset found. Collect data first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    
    # Count samples per key
    for key in KEY_MAP:
        key_count = len(df[df['key_pressed'] == key])
        print(f"Key '{key}': {key_count} samples")
    
    # Check for missing image files
    missing_files = 0
    for _, row in df.iterrows():
        if not os.path.exists(row['image_path']):
            missing_files += 1
    
    if missing_files > 0:
        print(f"Warning: {missing_files} image files are missing")
    
    # Calculate average press duration for each key
    for key in KEY_MAP:
        key_data = df[df['key_pressed'] == key]
        if not key_data.empty:
            avg_press = key_data['pressed_time'].mean()
            print(f"Average '{key}' press duration: {avg_press:.2f}s")

# Main execution
if __name__ == "__main__":
    while True:
        print("\nGTA5 Driving AI - Choose an option:")
        print("1: Collect data")
        print("2: Show dataset statistics")
        print("3: Train model")
        print("4: Test model in real-time (visualization only)")
        print("5: Autonomous driving mode (AI controls the game)")
        print("6: Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            samples_per_class = int(input("Enter target number of samples per class (default 1000): ") or 1000)
            resume = input("Resume from existing dataset? (y/n, default: y): ").lower() != 'n'
            collect_data(total_samples_per_class=samples_per_class, resume=resume)
        elif choice == '2':
            show_dataset_stats()
        elif choice == '3':
            train_model()
        elif choice == '4':
            test_duration = int(input("Enter test duration in seconds (default 60): ") or 60)
            test_model_realtime(run_time=test_duration)
        elif choice == '5':
            drive_duration = int(input("Enter driving duration in seconds (default 300): ") or 300)
            auto_drive(run_time=drive_duration)
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

# import mss
# import numpy as np
# import cv2
# import time
# import keyboard
# import os
# from ultralytics import YOLO
# from tensorflow.keras.models import load_model
# from collections import Counter

# # Configuration parameters
# MONITOR = {"top": 40, "left": 0, "width": 800, "height": 600}
# MODEL_PATH = "gta5_driving_model.h5"
# DATASET_DIR = "gta5_driving_dataset"
# DETECTION_THRESHOLD = 10.0  # Distance threshold in meters
# AUTO_DRIVE_DURATION = 300  # Duration in seconds

# # Create necessary directories
# os.makedirs(DATASET_DIR, exist_ok=True)

# # Initialize YOLO models
# detection_model = YOLO("yolov8n.pt")  # General detection model

# # Key mappings
# KEY_MAP = {
#     'w': 0,  # Forward
#     'a': 1,  # Left
#     'd': 2,  # Right
#     's': 3   # Brake
# }

# # Known real-world widths (in meters)
# KNOWN_WIDTHS = {
#     "car": 1.8,
#     "bus": 2.5,
#     "truck": 2.5,
#     "person": 0.5,  # Average width of a person
#     "traffic light": 0.3  # Typical width of a traffic light housing
# }

# FOCAL_LENGTH = 700  # Approximate focal length (needs calibration)

# # Function to capture screen
# def capture_screen():
#     """Capture screen using mss"""
#     with mss.mss() as sct:
#         img = np.array(sct.grab(MONITOR))
#         # Convert to BGR (OpenCV format) from BGRA
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# # Function to process image for ML model
# def process_image_for_model(img):
#     """Process the image for the model input"""
#     # Resize to model input dimensions
#     img = cv2.resize(img, (160, 120))
#     # Convert to grayscale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Normalize pixel values
#     img = img / 255.0
#     return img

# # Function to estimate distance
# def estimate_distance(label, pixel_width):
#     """Estimate distance based on known object width and pixel width"""
#     if label in KNOWN_WIDTHS:
#         return (FOCAL_LENGTH * KNOWN_WIDTHS[label]) / pixel_width
#     return None

# # Create warning icons
# def create_warning_icons():
#     """Create warning icons for display"""
#     # Create a brake warning icon (red circle with "BRAKE" text)
#     brake_icon = np.zeros((100, 100, 3), dtype=np.uint8)
#     cv2.circle(brake_icon, (50, 50), 45, (0, 0, 255), -1)
#     cv2.putText(brake_icon, "BRAKE", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
#     # Create a slow down warning icon (yellow triangle)
#     slow_icon = np.zeros((100, 100, 3), dtype=np.uint8)
#     triangle_pts = np.array([[50, 10], [10, 90], [90, 90]], np.int32)
#     cv2.fillPoly(slow_icon, [triangle_pts], (0, 255, 255))
#     cv2.putText(slow_icon, "SLOW", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
#     # Create an AI driving icon (blue rectangle)
#     ai_icon = np.zeros((100, 100, 3), dtype=np.uint8)
#     cv2.rectangle(ai_icon, (10, 10), (90, 90), (255, 0, 0), -1)
#     cv2.putText(ai_icon, "AI", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
#     return brake_icon, slow_icon, ai_icon

# # Detect objects and get distances
# def detect_objects(frame):
#     """Detect cars, pedestrians and other objects in the frame"""
#     results = detection_model(frame)
    
#     objects = []
#     min_distance = float('inf')
#     closest_object = None
    
#     # Process detections
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls[0])
#             label = detection_model.names[cls]
#             confidence = float(box.conf[0])
            
#             # Only process vehicles and pedestrians
#             if label in ["car", "bus", "truck", "person"] and confidence > 0.3:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 pixel_width = x2 - x1  # Width of the bounding box
#                 distance = estimate_distance(label, pixel_width)
                
#                 # Store this object
#                 objects.append({
#                     "label": label,
#                     "bbox": (x1, y1, x2, y2),
#                     "confidence": confidence,
#                     "distance": distance
#                 })
                
#                 # Update closest object
#                 if distance is not None and distance < min_distance:
#                     min_distance = distance
#                     closest_object = {
#                         "label": label,
#                         "distance": distance,
#                         "bbox": (x1, y1, x2, y2)
#                     }
    
#     return objects, closest_object, min_distance

# # Main function for advanced driving
# def advanced_driving(model_path=MODEL_PATH, run_time=AUTO_DRIVE_DURATION):
#     """
#     Advanced driving system that combines:
#     1. Object detection for cars and pedestrians
#     2. Default driving (W key) when path is clear
#     3. ML model for decision-making when objects are nearby
#     """
#     # Check if ML model exists
#     if not os.path.exists(model_path):
#         print(f"ML model not found at {model_path}. Please train the model first.")
#         return
    
#     # Load the trained model
#     ml_model = load_model(model_path)
#     print("ML driving model loaded successfully")
    
#     print(f"Starting advanced driving for {run_time} seconds...")
#     print("Press 'q' to quit early")
    
#     # Initialize variables
#     start_time = time.time()
#     frame_count = 0
#     action_map = {0: 'W', 1: 'A', 2: 'D', 3: 'S'}
#     prediction_history = []  # Keep history for smoothing predictions
    
#     # Create warning icons
#     brake_warning, slow_warning, ai_warning = create_warning_icons()
    
#     # Mode tracking
#     using_ml_mode = False
#     mode_switch_time = 0
#     ml_mode_duration = 0  # Track how long we've been in ML mode
    
#     # Debug variables
#     debug_info = []
    
#     while time.time() - start_time < run_time:
#         # Capture screen
#         frame = capture_screen()
        
#         # Get objects and distances
#         objects, closest_object, min_distance = detect_objects(frame)
        
#         # Current timestamp
#         current_time = time.time()
#         elapsed = current_time - start_time
        
#         # Process frame for ML model
#         processed_img = process_image_for_model(frame)
#         input_img = processed_img.reshape(1, 120, 160, 1)
        
#         # Make prediction (we'll decide later if we use it)
#         prediction = ml_model.predict(input_img, verbose=0)
#         predicted_class = np.argmax(prediction[0])
#         confidence = prediction[0][predicted_class]
        
#         # Update prediction history
#         prediction_history.append(predicted_class)
#         if len(prediction_history) > 5:
#             prediction_history.pop(0)
        
#         # Get most common prediction in recent history
#         smoothed_action = Counter(prediction_history).most_common(1)[0][0]
#         action_key = action_map[smoothed_action]
        
#         # Decision making based on object proximity
#         if min_distance <= DETECTION_THRESHOLD and closest_object is not None:
#             # Switch to ML mode if not already
#             if not using_ml_mode:
#                 using_ml_mode = True
#                 mode_switch_time = current_time
#                 print(f"Object detected at {min_distance:.2f}m - Switching to ML mode")
            
#             # Update ML mode duration
#             ml_mode_duration = current_time - mode_switch_time
            
#             # Execute ML model decision
#             for key in KEY_MAP:
#                 if keyboard.is_pressed(key):
#                     keyboard.release(key)
            
#             # Only use high confidence predictions
#             if confidence > 0.6:
#                 keyboard.press(action_key.lower())
                
#                 # Visual feedback
#                 action_text = f"AI DRIVING: {action_key} ({confidence:.2f})"
#                 color = (255, 0, 0)  # Blue for AI driving
                
#                 # Display AI driving icon
#                 h, w = frame.shape[:2]
#                 icon_h, icon_w = ai_warning.shape[:2]
#                 frame[h-icon_h-20:h-20, w-icon_w-20:w-20] = ai_warning
#             else:
#                 # If low confidence, go back to normal driving
#                 action_text = f"Low confidence - Default driving"
#                 color = (0, 165, 255)  # Orange
#                 keyboard.press('w')  # Default to moving forward
            
#             # Add to debug info
#             debug_info = [
#                 f"ML Mode: {ml_mode_duration:.1f}s",
#                 f"Closest: {closest_object['label']} at {min_distance:.2f}m", 
#                 f"Action: {action_key} (conf: {confidence:.2f})"
#             ]
            
#         else:
#             # Default driving mode - just press W
#             if using_ml_mode:
#                 using_ml_mode = False
#                 print("No close objects - Switching to default driving")
                
#                 # Release all keys first
#                 for key in KEY_MAP:
#                     if keyboard.is_pressed(key):
#                         keyboard.release(key)
            
#             # Default driving - press W
#             if not keyboard.is_pressed('w'):
#                 keyboard.press('w')
            
#             action_text = "Default Driving: W"
#             color = (0, 255, 0)  # Green
            
#             # Add to debug info
#             closest_info = f"Closest: {closest_object['label']} at {min_distance:.2f}m" if closest_object else "No objects detected"
#             debug_info = [
#                 "Default Driving Mode",
#                 closest_info
#             ]
        
#         # Draw object bounding boxes
#         for obj in objects:
#             x1, y1, x2, y2 = obj["bbox"]
#             label = obj["label"]
#             distance = obj["distance"]
            
#             # Color based on type and distance
#             color = (0, 255, 0)  # Green for vehicles
#             if label == "person":
#                 color = (255, 0, 0)  # Blue for pedestrians
            
#             # Make color brighter if object is close
#             if distance is not None and distance < DETECTION_THRESHOLD:
#                 if label == "person":
#                     color = (255, 0, 255)  # Purple for close pedestrians
#                 else:
#                     color = (0, 255, 255)  # Yellow for close vehicles
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
#             # Add label with distance
#             distance_text = f"{distance:.1f}m" if distance is not None else "?"
#             label_text = f"{label} {distance_text}"
#             cv2.putText(frame, label_text, (x1, y1-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#         # Draw current driving mode info
#         cv2.putText(frame, action_text, (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
#         # Draw debug info
#         for i, info in enumerate(debug_info):
#             cv2.putText(frame, info, (10, 60 + i*25), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
#         # Calculate and display FPS
#         if frame_count > 0:
#             fps = frame_count / elapsed
#             cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Display the resulting frame
#         cv2.imshow('GTA5 Advanced Driving Assistant', frame)
#         frame_count += 1
        
#         # Press q to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Make sure to release all keys before quitting
#     for key in KEY_MAP:
#         keyboard.release(key)
    
#     # Clean up
#     cv2.destroyAllWindows()
    
#     # Print statistics
#     print("\nAdvanced driving completed!")
#     print(f"Total frames processed: {frame_count}")
#     print(f"Average FPS: {frame_count / min(time.time() - start_time, run_time):.1f}")

# # Main execution
# if __name__ == "__main__":
#     print("\nGTA5 Advanced Driving Assistant")
#     print("This system combines object detection with ML-based driving")
#     print("It will drive normally (W key) when the path is clear")
#     print("When objects are detected within 10m, it switches to ML-based driving")
    
#     while True:
#         print("\nChoose an option:")
#         print("1: Start advanced driving")
#         print("2: Change detection threshold (currently", DETECTION_THRESHOLD, "meters)")
#         print("3: Change driving duration (currently", AUTO_DRIVE_DURATION, "seconds)")
#         print("4: Exit")
        
#         choice = input("Enter your choice (1-4): ")
        
#         if choice == '1':
#             try:
#                 advanced_driving(run_time=AUTO_DRIVE_DURATION)
#             except Exception as e:
#                 print(f"Error during advanced driving: {e}")
#                 # Release all keys in case of error
#                 for key in KEY_MAP:
#                     if keyboard.is_pressed(key):
#                         keyboard.release(key)
#         elif choice == '2':
#             try:
#                 new_threshold = float(input(f"Enter new detection threshold in meters (currently {DETECTION_THRESHOLD}): "))
#                 if new_threshold > 0:
#                     DETECTION_THRESHOLD = new_threshold
#                     print(f"Detection threshold updated to {DETECTION_THRESHOLD} meters")
#                 else:
#                     print("Threshold must be positive")
#             except ValueError:
#                 print("Please enter a valid number")
#         elif choice == '3':
#             try:
#                 new_duration = int(input(f"Enter new driving duration in seconds (currently {AUTO_DRIVE_DURATION}): "))
#                 if new_duration > 0:
#                     AUTO_DRIVE_DURATION = new_duration
#                     print(f"Driving duration updated to {AUTO_DRIVE_DURATION} seconds")
#                 else:
#                     print("Duration must be positive")
#             except ValueError:
#                 print("Please enter a valid number")
#         elif choice == '4':
#             print("Exiting program.")
#             break
#         else:
#             print("Invalid choice. Please try again.")