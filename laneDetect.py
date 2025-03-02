import cv2
import numpy as np
import pyautogui
import vgamepad as vg
import time

# Initialize virtual Xbox controller
gamepad = vg.VX360Gamepad()

def canny(img):
    # Check if frame is valid
    if img is None:
        exit()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # Apply Canny edge detection
    canny = cv2.Canny(gray, 50, 150)
    return canny

def region_of_interest(canny):
    # Create a mask for the region of interest (triangle)
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    
    # Adjust triangle points to fit the screen dimensions
    triangle = np.array([[
        (100, height),
        (width//2, height//2),
        (width-100, height),
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, triangle, 255) # type: ignore
    # Apply the mask to the canny image
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    # Detect lines using Hough Transform
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=30, maxLineGap=5)

def addWeighted(frame, line_image):
    # Combine the original frame with the line image
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def display_lines(img, lines):
    # Create an image to draw the lines on
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    return line_image

def make_points(image, line):
    # Calculate endpoints for a line defined by slope and intercept
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0/5)      
    
    # Handle potential division by zero
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    # Keep points within image boundaries
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    
    return np.array([[x1, y1, x2, y2]], dtype=np.int32)

def average_slope_intercept(image, lines):
    # Separate lines into left and right based on slope
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Avoid division by zero
            if x2 - x1 == 0:
                continue
                
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    # Calculate average line for left and right lanes
    left_line = right_line = None
    
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    
    averaged_lines = []
    if left_line is not None:
        averaged_lines.append(left_line)
    if right_line is not None:
        averaged_lines.append(right_line)
        
    return averaged_lines if averaged_lines else None

def calculate_lane_center(image, lines):
    """Calculate the center point between detected lanes"""
    image_width = image.shape[1]
    image_center = image_width // 2
    
    if lines is None or len(lines) == 0:
        return image_center  # Return center of image if no lanes detected
    
    # Initialize the bottom x-coordinates for both lanes
    left_x = None
    right_x = None
    
    # Extract lane positions from detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Use both bottom points (x1) and calculate their average position
        if x1 < image_center:  # Left lane
            left_x = x1
        elif x1 > image_center:  # Right lane
            right_x = x1
    
    # Calculate the lane center based on available lane detections
    if left_x is not None and right_x is not None:
        # Both lanes detected - use the midpoint
        lane_center = (left_x + right_x) // 2
    elif left_x is not None:
        # Only left lane detected - estimate center by adding lane width
        # Typically, lanes are about 300-400 pixels apart in the capture
        lane_center = left_x + 350
    elif right_x is not None:
        # Only right lane detected - estimate center by subtracting lane width
        lane_center = right_x - 350
    else:
        # Fallback to image center
        lane_center = image_center
    
    return lane_center

def control_car(image, lanes):
    """Generate controller inputs based on lane detection"""
    # Get image center and detected lane center
    image_center = image.shape[1] // 2
    lane_center = calculate_lane_center(image, lanes)
    
    # Calculate offset - reinverted to match your game's steering logic:
    # If lane_center is to the right of image_center, steer right (positive)
    # If lane_center is to the left of image_center, steer left (negative)
    offset = lane_center - image_center
    
    # Convert offset to steering value (-32768 to 32767 for Xbox controller)
    max_joystick = 32767
    sensitivity = 150  # Adjust sensitivity as needed
    
    # Calculate steering value and ensure it's within the controller's range
    steering_float = np.clip(offset * sensitivity, -max_joystick, max_joystick)
    steering_value = int(steering_float)  # Convert to int for vgamepad
    
    # Apply steering to left thumbstick X-axis
    gamepad.left_joystick_float(x_value_float=steering_float/max_joystick, y_value_float=0.0)
    
    # Always press accelerator (RT) with moderate pressure
    gamepad.right_trigger_float(value_float=0.5)
    
    # Update controller
    gamepad.update()
    
    # Display steering info on image
    cv2.putText(image, f"Steering: {steering_value/max_joystick:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Lane Center: {lane_center}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Offset: {offset}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw lane center and image center lines
    cv2.line(image, (lane_center, 0), (lane_center, image.shape[0]), (0, 255, 0), 2)
    cv2.line(image, (image_center, 0), (image_center, image.shape[0]), (255, 0, 0), 2)
    
    return image

# Main processing loop
try:
    # Use the provided monitor dimensions
    monitor = {"top": 40, "left": 0, "width": 800, "height": 600}
    
    # Create a capture region from monitor dict
    capture_region = (monitor["left"], monitor["top"], monitor["width"], monitor["height"])
    
    # Small delay to let the gamepad initialize
    print("Controller initialized. Starting in 3 seconds...")
    time.sleep(3)
    
    while True:
        # Capture screen
        screenshot = pyautogui.screenshot(region=capture_region)
        # Convert PIL image to OpenCV format
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process the captured frame
        canny_image = canny(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = houghLines(cropped_canny)
        averaged_lines = average_slope_intercept(frame, lines)
        
        # Control the car based on lane detection
        frame = control_car(frame, averaged_lines)
        
        # Create visual output
        line_image = display_lines(frame, averaged_lines)
        combo_image = addWeighted(frame, line_image)
        
        # Display the result
        cv2.imshow("Lane Detection & Control", combo_image)
        cv2.imshow("Region of Interest", cropped_canny)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Release the gamepad when done
    if 'gamepad' in locals():
        # Reset controls using float methods
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.right_trigger_float(value_float=0.0)
        gamepad.update()
        print("Controller reset")
    
    cv2.destroyAllWindows()
    print("Program terminated")