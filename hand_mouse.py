import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import speech_recognition as sr
import threading
import queue

# --- Application Configuration & Constants ---

# Webcam and Screen Configuration
WEBCAM_ID = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Mouse Control Parameters
SMOOTHING_FACTOR = 4  # Higher values mean smoother but slower mouse movement
PREV_X, PREV_Y = 0, 0 # Stores previous mouse coordinates for smoothing

# Gesture Cooldowns (in seconds)
LCLICK_COOLDOWN = 0.4
RCLICK_COOLDOWN = 0.5
GESTURE_COOLDOWN_AFTER_DRAG = 0.3 # Prevents accidental clicks immediately after drag
SPEECH_ACTIVATION_COOLDOWN = 1.0
PAUSE_GESTURE_COOLDOWN = 0.7
SCROLL_MODE_ACTIVATION_COOLDOWN = 0.7

# Timestamps for Cooldown Management
last_lclick_time = 0
last_rclick_time = 0
last_drag_release_time = 0
last_speech_activation_time = 0
last_pause_toggle_time = 0
last_scroll_mode_activation_time = 0

# State Flags
is_lclicking_state = False          # True if a left click was just performed (for cooldown)
is_rclicking_state = False          # True if a right click was just performed (for cooldown)
is_dragging_state = False           # True if the mouse is currently dragging
is_listening_speech_active = False  # True if speech recognition is active
speech_thread_running = False       # True if the speech recognition worker thread is running
mouse_tracking_paused = False       # True if hand tracking for mouse control is paused
is_scrolling_mode_active = False    # True if scroll mode is active

# Scroll Mode Parameters
last_scroll_y_normalized = 0.5  # Last y-coordinate used for scrolling calculation
SCROLL_SENSITIVITY = 30         # Percentage determining scroll speed

# Speech Recognition Visual Feedback
speech_visual_status = ""       # String to display current speech recognition status
speech_status_lock = threading.Lock() # Lock for thread-safe updates to speech_visual_status

# Application State Machine
APP_STATE = "CALIBRATING_START" # Initial state: "CALIBRATING_START", "CALIBRATING_TOP_LEFT", "CALIBRATING_BOTTOM_RIGHT", "RUNNING"

# Calibration Parameters
calib_points_normalized = {"top_left": None, "bottom_right": None} # Stores normalized calibration points
# Default input area for mapping if calibration is skipped (as a percentage of frame size)
calib_input_x_min, calib_input_y_min = 0.05, 0.05
calib_input_x_max, calib_input_y_max = 0.95, 0.95

# Finger Gesture Detection Thresholds
FINGER_EXTENDED_ANGLE_THRESHOLD = 160  # Angle (degrees) to consider a finger joint extended
FINGER_CURLED_ANGLE_THRESHOLD = 100    # Angle (degrees) to consider a finger joint curled

# UI Control Parameters (dynamically adjusted via trackbars)
camera_zoom_level = 1.0         # Camera digital zoom level
mapping_inset_percent = 0.0     # Percentage to inset the mapping area from calibrated bounds

# MediaPipe Hands Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
landmark_names = mp_hands.HandLandmark # Enum for easy access to landmark names

# Speech Recognition Initialization
recognizer = sr.Recognizer()
microphone = sr.Microphone()
recognizer.pause_threshold = 0.7    # Seconds of non-speaking audio before phrase is considered complete
recognizer.energy_threshold = 400   # Minimum audio energy to consider for recording

# Inter-thread Communication for Speech
text_to_paste_queue = queue.Queue() # Queue for passing recognized text to the main thread
speech_control_event = threading.Event() # Event to signal the speech worker thread

def on_zoom_trackbar(val: int):
    """Callback function for the zoom trackbar. Updates the global zoom level."""
    global camera_zoom_level
    camera_zoom_level = max(1.0, val / 10.0)
    print(f"Zoom: {camera_zoom_level:.1f}x")

def on_inset_trackbar(val: int):
    """Callback function for the inset trackbar. Updates the global mapping inset percentage."""
    global mapping_inset_percent
    mapping_inset_percent = val / 100.0
    print(f"Inset: {mapping_inset_percent*100:.0f}%")

def update_speech_status(new_status: str):
    """
    Thread-safely updates the global string for speech recognition visual feedback.
    Args:
        new_status: The new status message to display.
    """
    global speech_visual_status
    with speech_status_lock:
        speech_visual_status = new_status

def speech_to_text_worker():
    """
    Worker thread for handling speech recognition.
    Listens for audio input when activated and uses Google Web Speech API to transcribe.
    Communicates recognized text back to the main thread via a queue.
    Manages its own lifecycle based on `speech_thread_running` and `speech_control_event`.
    """
    global is_listening_speech_active, speech_thread_running
    speech_thread_running = True
    print("Speech worker thread started.")

    while speech_thread_running:
        speech_control_event.wait() # Wait until speech recognition is activated
        if not speech_thread_running:
            break # Exit if thread is flagged to stop

        if not is_listening_speech_active:
            update_speech_status("")
            speech_control_event.clear() # Clear event if speech became inactive while waiting
            continue

        update_speech_status("Listening...")
        print("Listening for speech...")
        audio_data = None
        try:
            with microphone as source:
                # Adjust for ambient noise once at the start of listening
                # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Optional: can improve accuracy
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            if not is_listening_speech_active: # Check again if speech was deactivated during listen
                update_speech_status("")
                speech_control_event.clear()
                continue

            if audio_data:
                update_speech_status("Recognizing...")
                print("Recognizing speech...")
                text = recognizer.recognize_google(audio_data)
                print(f"Speech recognized: {text}")
                text_to_paste_queue.put(text + " ") # Add space for continuous typing
                update_speech_status("✓") # Success indicator
                time.sleep(0.5) # Display success briefly
            else:
                update_speech_status("") # No audio data
        except sr.WaitTimeoutError:
            print("No speech detected within timeout.")
            update_speech_status("Timeout")
            time.sleep(0.5)
        except sr.UnknownValueError:
            print("Could not understand audio.")
            update_speech_status("Unknown")
            time.sleep(0.5)
        except sr.RequestError as e:
            print(f"Speech service error; {e}")
            update_speech_status("Error")
            time.sleep(0.5)
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred in speech worker: {e}")
            update_speech_status("SysError")
            time.sleep(0.5)
        finally:
            speech_control_event.clear() # Ready for next activation
            if not is_listening_speech_active: # Ensure status is cleared if deactivated
                update_speech_status("")

    update_speech_status("") # Clear status on exit
    print("Speech worker thread stopped.")

def calculate_angle(p1_lm, p2_lm, p3_lm) -> float:
    """
    Calculates the angle between three 2D landmarks.
    Args:
        p1_lm: First landmark (e.g., wrist).
        p2_lm: Second landmark, the vertex of the angle (e.g., MCP joint).
        p3_lm: Third landmark (e.g., PIP joint).
    Returns:
        The angle in degrees. Returns 180 if landmarks are collinear or form a zero-magnitude vector.
    """
    # Create vectors from p2 to p1 and p2 to p3
    v1_x, v1_y = p1_lm.x - p2_lm.x, p1_lm.y - p2_lm.y
    v2_x, v2_y = p3_lm.x - p2_lm.x, p3_lm.y - p2_lm.y

    # Calculate dot product and magnitudes
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

    # Prevent division by zero
    if mag_v1 * mag_v2 == 0:
        return 180.0  # Or a suitable default for collinear/zero-length vectors

    # Calculate cosine of the angle
    acos_arg = dot_product / (mag_v1 * mag_v2)
    # Clamp acos_arg to [-1, 1] to prevent domain errors due to floating point inaccuracies
    acos_arg = max(-1.0, min(1.0, acos_arg))

    angle_rad = math.acos(acos_arg)
    return math.degrees(angle_rad)

def is_finger_extended(landmarks, finger_name: str) -> bool:
    """
    Checks if a specific finger is extended based on joint angles.
    Args:
        landmarks: List of hand landmarks.
        finger_name: Name of the finger ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY").
    Returns:
        True if the finger is considered extended, False otherwise.
    """
    if finger_name == "PINKY":
        mcp_lm = landmarks[landmark_names.PINKY_MCP]
        pip_lm = landmarks[landmark_names.PINKY_PIP]
        dip_lm = landmarks[landmark_names.PINKY_DIP]
        tip_lm = landmarks[landmark_names.PINKY_TIP]
    elif finger_name == "THUMB":
        # Thumb has different joint names and structure
        cmc_lm = landmarks[landmark_names.THUMB_CMC]
        mcp_lm = landmarks[landmark_names.THUMB_MCP]
        ip_lm = landmarks[landmark_names.THUMB_IP] # Intermediate phalanx
        tip_lm = landmarks[landmark_names.THUMB_TIP]
        # Check angles at MCP and IP joints for thumb extension
        angle_mcp = calculate_angle(cmc_lm, mcp_lm, ip_lm)
        angle_ip = calculate_angle(mcp_lm, ip_lm, tip_lm)
        # Thumb uses a slightly more relaxed threshold
        return angle_mcp > (FINGER_EXTENDED_ANGLE_THRESHOLD - 30) and \
               angle_ip > (FINGER_EXTENDED_ANGLE_THRESHOLD - 30)
    else: # INDEX, MIDDLE, RING
        mcp_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_MCP")]
        pip_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_PIP")]
        dip_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_DIP")]
        tip_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_TIP")]

    # For other fingers, check angles at PIP and DIP joints
    angle_pip = calculate_angle(mcp_lm, pip_lm, dip_lm)
    angle_dip = calculate_angle(pip_lm, dip_lm, tip_lm)
    return angle_pip > FINGER_EXTENDED_ANGLE_THRESHOLD and \
           angle_dip > FINGER_EXTENDED_ANGLE_THRESHOLD

def is_finger_curled(landmarks, finger_name: str) -> bool:
    """
    Checks if a specific finger is curled based on joint angles and tip position.
    Args:
        landmarks: List of hand landmarks.
        finger_name: Name of the finger ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY").
    Returns:
        True if the finger is considered curled, False otherwise.
    """
    if finger_name == "PINKY":
        mcp_lm = landmarks[landmark_names.PINKY_MCP]
        pip_lm = landmarks[landmark_names.PINKY_PIP]
        tip_lm = landmarks[landmark_names.PINKY_TIP]
    elif finger_name == "THUMB":
        cmc_lm = landmarks[landmark_names.THUMB_CMC]
        mcp_lm = landmarks[landmark_names.THUMB_MCP]
        ip_lm = landmarks[landmark_names.THUMB_IP]
        tip_lm = landmarks[landmark_names.THUMB_TIP]
        # Check angles at MCP and IP joints for thumb curling
        angle_mcp = calculate_angle(cmc_lm, mcp_lm, ip_lm)
        angle_ip = calculate_angle(mcp_lm, ip_lm, tip_lm)
        # Thumb uses a slightly more relaxed threshold for curled
        return angle_mcp < (FINGER_CURLED_ANGLE_THRESHOLD + 20) and \
               angle_ip < (FINGER_CURLED_ANGLE_THRESHOLD + 20)
    else: # INDEX, MIDDLE, RING
        mcp_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_MCP")]
        pip_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_PIP")]
        tip_lm = landmarks[getattr(landmark_names, f"{finger_name}_FINGER_TIP")]

    # Angle between MCP-PIP and PIP-TIP vectors
    angle_pip_tip = calculate_angle(mcp_lm, pip_lm, tip_lm)
    # Additional check: finger tip is significantly lower (higher y-value) than PIP joint
    # This helps distinguish curling from just bending slightly.
    is_vertically_curled = tip_lm.y > (pip_lm.y + 0.01) # Small threshold for vertical curl

    return angle_pip_tip < FINGER_CURLED_ANGLE_THRESHOLD and is_vertically_curled

def is_pause_gesture_active(landmarks) -> bool:
    """
    Checks for the pause gesture: Index and Pinky fingers extended, Middle and Ring curled.
    Args:
        landmarks: List of hand landmarks.
    Returns:
        True if the pause gesture is active, False otherwise.
    """
    middle_curled = is_finger_curled(landmarks, "MIDDLE")
    ring_curled = is_finger_curled(landmarks, "RING")
    pinky_extended = is_finger_extended(landmarks, "PINKY")
    index_extended = is_finger_extended(landmarks, "INDEX")
    return index_extended and middle_curled and ring_curled and pinky_extended

def is_all_fingers_extended(landmarks) -> bool:
    """
    Checks if all five fingers are extended. Used to activate speech mode.
    Args:
        landmarks: List of hand landmarks.
    Returns:
        True if all fingers are extended, False otherwise.
    """
    return (is_finger_extended(landmarks, "THUMB") and
            is_finger_extended(landmarks, "INDEX") and
            is_finger_extended(landmarks, "MIDDLE") and
            is_finger_extended(landmarks, "RING") and
            is_finger_extended(landmarks, "PINKY"))

def is_three_fingers_scroll_gesture(landmarks) -> bool:
    """
    Checks for the scroll mode gesture: Middle, Ring, and Pinky fingers extended,
    Index finger curled. (Thumb state is not explicitly checked for this gesture).
    Args:
        landmarks: List of hand landmarks.
    Returns:
        True if the scroll gesture is active, False otherwise.
    """
    index_curled = is_finger_curled(landmarks, "INDEX")
    # Thumb curled (or not extended) could be an implicit part of the gesture,
    # but we only explicitly check the required fingers for activation.
    # thumb_curled = is_finger_curled(landmarks, "THUMB") # or not is_finger_extended(landmarks, "THUMB")
    middle_extended = is_finger_extended(landmarks, "MIDDLE")
    ring_extended = is_finger_extended(landmarks, "RING")
    pinky_extended = is_finger_extended(landmarks, "PINKY")
    # The original logic only explicitly required index curled and other three extended.
    return index_curled and middle_extended and ring_extended and pinky_extended

def map_coordinates_calibrated(norm_x: float, norm_y: float) -> tuple[int, int]:
    """
    Maps normalized hand coordinates (0.0-1.0) from the camera frame
    to screen pixel coordinates, using the calibrated input area and inset.
    Args:
        norm_x: Normalized x-coordinate from hand tracking.
        norm_y: Normalized y-coordinate from hand tracking.
    Returns:
        A tuple (screen_x, screen_y) of pixel coordinates.
    """
    global mapping_inset_percent, calib_input_x_min, calib_input_x_max, calib_input_y_min, calib_input_y_max

    # Calculate the width and height of the calibrated input area
    current_calib_width = calib_input_x_max - calib_input_x_min
    current_calib_height = calib_input_y_max - calib_input_y_min

    # Apply inset to the calibrated area to create an effective mapping area
    inset_x_amount = current_calib_width * mapping_inset_percent
    inset_y_amount = current_calib_height * mapping_inset_percent

    effective_map_x_min = calib_input_x_min + inset_x_amount
    effective_map_x_max = calib_input_x_max - inset_x_amount
    effective_map_y_min = calib_input_y_min + inset_y_amount
    effective_map_y_max = calib_input_y_max - inset_y_amount

    # Ensure min is less than max, handle cases where inset might make them equal or inverted
    if effective_map_x_min >= effective_map_x_max:
        center_x = (calib_input_x_min + calib_input_x_max) / 2
        effective_map_x_min = center_x - 0.01 # Create a tiny valid range
        effective_map_x_max = center_x + 0.01
    if effective_map_y_min >= effective_map_y_max:
        center_y = (calib_input_y_min + calib_input_y_max) / 2
        effective_map_y_min = center_y - 0.01 # Create a tiny valid range
        effective_map_y_max = center_y + 0.01

    # Interpolate normalized coordinates to screen coordinates
    screen_x = np.interp(norm_x, [effective_map_x_min, effective_map_x_max], [0, SCREEN_WIDTH])
    screen_y = np.interp(norm_y, [effective_map_y_min, effective_map_y_max], [0, SCREEN_HEIGHT])

    return int(screen_x), int(screen_y)

# --- Main Application Setup ---
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit() # Use sys.exit in larger applications for cleaner exit codes

WINDOW_NAME = 'Hand Tracking Mouse Control'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
pyautogui.FAILSAFE = False # Disable PyAutoGUI's failsafe (moving mouse to corner to stop)

# Create UI trackbars for adjusting zoom and mapping inset
cv2.createTrackbar('Zoom (x10)', WINDOW_NAME, 10, 30, on_zoom_trackbar) # 1.0x to 3.0x
cv2.createTrackbar('Inset (%)', WINDOW_NAME, 0, 40, on_inset_trackbar)   # 0% to 40%

# Start the speech recognition worker thread
speech_thread = threading.Thread(target=speech_to_text_worker, daemon=True)
speech_thread.start()

print("Hand Tracking Mouse Control Initialized.")
print("Controls:")
print("  - Pause Tracking: Index + Pinky extended, Middle + Ring curled.")
print("  - Scroll Mode: Middle + Ring + Pinky extended, Index curled.")
print("  - Speech-to-Text: All fingers extended.")
print("  - Calibration: Follow on-screen prompts ('c' to start, 's' to skip).")
print("  - Quit: Press 'q'.")

# --- Main Application Loop ---
try:
    while cap.isOpened():
        # Process any recognized text from the speech queue
        try:
            recognized_text = text_to_paste_queue.get_nowait()
            if recognized_text:
                pyautogui.typewrite(recognized_text, interval=0.01) # Type with small delay
        except queue.Empty:
            pass # No text to paste

        success, image_raw = cap.read()
        if not success:
            print("Warning: Failed to grab frame from webcam.")
            time.sleep(0.1) # Wait a bit before trying again
            continue

        # Flip the image horizontally for a more natural "mirror" view
        image_orig_flipped = cv2.flip(image_raw, 1)
        orig_h, orig_w = image_orig_flipped.shape[:2]

        # Apply digital zoom if enabled
        image_to_process = image_orig_flipped
        if camera_zoom_level > 1.001: # Check with a small epsilon for floating point comparison
            crop_w = max(1, int(orig_w / camera_zoom_level))
            crop_h = max(1, int(orig_h / camera_zoom_level))
            mid_x, mid_y = orig_w // 2, orig_h // 2

            # Calculate crop boundaries, ensuring they are within image dimensions
            x1 = max(0, mid_x - crop_w // 2)
            x2 = min(orig_w, mid_x + crop_w // 2)
            y1 = max(0, mid_y - crop_h // 2)
            y2 = min(orig_h, mid_y + crop_h // 2)

            if x2 > x1 and y2 > y1: # Ensure valid crop dimensions
                cropped_image = image_orig_flipped[y1:y2, x1:x2]
                image_to_process = cv2.resize(cropped_image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                image_to_process = image_orig_flipped.copy() # Fallback if crop is invalid
        else:
            image_to_process = image_orig_flipped.copy()

        # Prepare image for display and processing
        image_display = image_to_process.copy()
        image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = image_to_process.shape

        # Process the image with MediaPipe Hands
        results = hands_model.process(image_rgb)
        key_pressed = cv2.waitKey(5) & 0xFF

        # --- Application State Logic ---
        if APP_STATE == "CALIBRATING_START":
            cv2.putText(image_display, "Adjust Zoom/Inset, then press 'c' to Calibrate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(image_display, "or 's' to Skip Calibration (uses default area)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            if key_pressed == ord('c'):
                APP_STATE = "CALIBRATING_TOP_LEFT"
                print("Calibration started: Aim for TOP-LEFT point.")
            elif key_pressed == ord('s'):
                APP_STATE = "RUNNING"
                pyautogui.PAUSE = 0.0 # PyAutoGUI delay between actions
                print("Calibration skipped. Using default mapping area.")

        elif APP_STATE == "CALIBRATING_TOP_LEFT":
            dot_tl_x = int(0.15 * frame_width) # Target dot position
            dot_tl_y = int(0.15 * frame_height)
            cv2.putText(image_display, "Aim INDEX FINGER TIP at the GREEN dot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_display, "Press SPACE to set TOP-LEFT corner", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(image_display, (dot_tl_x, dot_tl_y), 15, (0, 255, 0), -1) # Draw target dot

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                calib_lm = hand_landmarks.landmark[landmark_names.INDEX_FINGER_TIP]
                # Draw user's current pointer tip for aiming
                cv2.circle(image_display, (int(calib_lm.x * frame_width), int(calib_lm.y * frame_height)), 7, (0, 0, 255), -1)
                if key_pressed == ord(' '):
                    calib_points_normalized["top_left"] = (calib_lm.x, calib_lm.y)
                    APP_STATE = "CALIBRATING_BOTTOM_RIGHT"
                    print(f"Top-Left point calibrated: ({calib_lm.x:.2f}, {calib_lm.y:.2f}). Aim for BOTTOM-RIGHT.")

        elif APP_STATE == "CALIBRATING_BOTTOM_RIGHT":
            dot_br_x = int(0.85 * frame_width) # Target dot position
            dot_br_y = int(0.85 * frame_height)
            cv2.putText(image_display, "Aim INDEX FINGER TIP at the GREEN dot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_display, "Press SPACE to set BOTTOM-RIGHT corner", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(image_display, (dot_br_x, dot_br_y), 15, (0, 255, 0), -1) # Draw target dot

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                calib_lm = hand_landmarks.landmark[landmark_names.INDEX_FINGER_TIP]
                cv2.circle(image_display, (int(calib_lm.x * frame_width), int(calib_lm.y * frame_height)), 7, (0, 0, 255), -1)
                if key_pressed == ord(' '):
                    calib_points_normalized["bottom_right"] = (calib_lm.x, calib_lm.y)
                    print(f"Bottom-Right point calibrated: ({calib_lm.x:.2f}, {calib_lm.y:.2f}).")

                    # Finalize calibration input area
                    tl_x, tl_y = calib_points_normalized["top_left"]
                    br_x, br_y = calib_points_normalized["bottom_right"]
                    calib_input_x_min = min(tl_x, br_x)
                    calib_input_x_max = max(tl_x, br_x)
                    calib_input_y_min = min(tl_y, br_y)
                    calib_input_y_max = max(tl_y, br_y)

                    # Ensure a minimum valid calibration range
                    if (calib_input_x_max - calib_input_x_min) < 0.01: # Too small X range
                        calib_input_x_min = 0.0; calib_input_x_max = 1.0 # Reset to default
                        print("Warning: X calibration range too small, using default X.")
                    if (calib_input_y_max - calib_input_y_min) < 0.01: # Too small Y range
                        calib_input_y_min = 0.0; calib_input_y_max = 1.0 # Reset to default
                        print("Warning: Y calibration range too small, using default Y.")

                    APP_STATE = "RUNNING"
                    pyautogui.PAUSE = 0.0
                    print(f"Calibration complete. Mapping area: X({calib_input_x_min:.2f}-{calib_input_x_max:.2f}), Y({calib_input_y_min:.2f}-{calib_input_y_max:.2f}).")

        elif APP_STATE == "RUNNING":
            current_gesture_text = "Tracking Hand" # Default status text
            if mouse_tracking_paused:
                current_gesture_text = "PAUSED"
            if is_scrolling_mode_active:
                current_gesture_text = "SCROLL MODE"

            # Get local copy of speech status for display (thread-safe)
            with speech_status_lock:
                local_speech_status = speech_visual_status

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_list = hand_landmarks.landmark # For easier access
                mp_drawing.draw_landmarks(image_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                time_now = time.time() # Current time for cooldown checks

                # --- Gesture Detection ---
                current_is_pause_gesture = is_pause_gesture_active(landmarks_list)
                current_is_scroll_gesture = is_three_fingers_scroll_gesture(landmarks_list)
                all_fingers_ext = is_all_fingers_extended(landmarks_list)

                # Pause Gesture Logic (Highest precedence after hand detection)
                if current_is_pause_gesture and not current_is_scroll_gesture and not all_fingers_ext:
                    if time_now - last_pause_toggle_time > PAUSE_GESTURE_COOLDOWN:
                        mouse_tracking_paused = not mouse_tracking_paused
                        last_pause_toggle_time = time_now
                        if mouse_tracking_paused:
                            print("Mouse control PAUSED.")
                            current_gesture_text = "PAUSED"
                            # Release drag/scroll/speech if active when pausing
                            if is_dragging_state:
                                pyautogui.mouseUp()
                                is_dragging_state = False
                            if is_listening_speech_active:
                                is_listening_speech_active = False
                                speech_control_event.clear()
                                update_speech_status("")
                            if is_scrolling_mode_active:
                                is_scrolling_mode_active = False
                                print("Scroll mode DEACTIVATED (due to pause).")
                        else:
                            print("Mouse control RESUMED.")
                            current_gesture_text = "RESUMED"

                # --- Active Mouse Control (if not paused) ---
                if not mouse_tracking_paused:
                    # Finger state checks for other gestures
                    pointer_finger_extended = is_finger_extended(landmarks_list, "INDEX")
                    middle_finger_extended = is_finger_extended(landmarks_list, "MIDDLE")
                    ring_finger_extended = is_finger_extended(landmarks_list, "RING")
                    pinky_finger_extended = is_finger_extended(landmarks_list, "PINKY")

                    # Scroll Mode Activation/Deactivation
                    if current_is_scroll_gesture:
                        current_gesture_text = "SCROLL MODE"
                        if not is_scrolling_mode_active and (time_now - last_scroll_mode_activation_time > SCROLL_MODE_ACTIVATION_COOLDOWN):
                            is_scrolling_mode_active = True
                            last_scroll_mode_activation_time = time_now
                            # Use wrist y-position as the initial reference for scrolling
                            scroll_control_point_y = landmarks_list[landmark_names.WRIST].y
                            last_scroll_y_normalized = scroll_control_point_y
                            print("Scroll mode ACTIVATED.")
                            # Deactivate other states when entering scroll mode
                            if is_dragging_state: pyautogui.mouseUp(); is_dragging_state = False
                            if is_listening_speech_active: is_listening_speech_active = False; speech_control_event.clear(); update_speech_status("")
                            is_lclicking_state = False; is_rclicking_state = False

                        if is_scrolling_mode_active:
                            current_scroll_y = landmarks_list[landmark_names.WRIST].y
                            delta_y_normalized = current_scroll_y - last_scroll_y_normalized
                            # Scale scroll amount by frame height and sensitivity
                            # Negative sign inverts scroll direction to be more natural (move hand up to scroll up)
                            scroll_amount = -int(delta_y_normalized * frame_height * (SCROLL_SENSITIVITY / 100.0))
                            if abs(scroll_amount) > 1: # Apply a threshold to avoid jittery scrolling
                                pyautogui.scroll(scroll_amount)
                            last_scroll_y_normalized = current_scroll_y # Update reference for next frame

                    elif is_scrolling_mode_active and not current_is_scroll_gesture: # Exiting Scroll Mode
                        is_scrolling_mode_active = False
                        last_scroll_mode_activation_time = time_now # Reset cooldown for re-activation
                        print("Scroll mode DEACTIVATED.")
                        current_gesture_text = "Tracking Hand"

                    # Mouse Pointer Movement (if not in scroll mode)
                    if not is_scrolling_mode_active:
                        # Use Index Finger MCP (knuckle) as the control point for mouse movement
                        mouse_control_landmark_type = landmark_names.INDEX_FINGER_MCP
                        control_lm = landmarks_list[mouse_control_landmark_type]
                        target_x, target_y = map_coordinates_calibrated(control_lm.x, control_lm.y)

                        # Apply smoothing to mouse movement
                        current_x = PREV_X + (target_x - PREV_X) / SMOOTHING_FACTOR
                        current_y = PREV_Y + (target_y - PREV_Y) / SMOOTHING_FACTOR

                        # Clamp coordinates to screen boundaries
                        clamped_x = max(0, min(SCREEN_WIDTH - 1, int(current_x)))
                        clamped_y = max(0, min(SCREEN_HEIGHT - 1, int(current_y)))

                        if not is_listening_speech_active: # Don't move mouse during speech input
                            pyautogui.moveTo(clamped_x, clamped_y, duration=0) # Instantaneous move

                        PREV_X, PREV_Y = current_x, current_y

                    # Speech Mode Activation/Deactivation (if not scrolling)
                    if not is_scrolling_mode_active and all_fingers_ext:
                        current_gesture_text = "SPEECH MODE"
                        if not is_listening_speech_active and (time_now - last_speech_activation_time > SPEECH_ACTIVATION_COOLDOWN):
                            is_listening_speech_active = True
                            speech_control_event.set() # Signal speech worker to start listening
                            last_speech_activation_time = time_now
                            print("Speech recognition ACTIVATED.")
                        # Ensure other actions are off during speech
                        if is_dragging_state: pyautogui.mouseUp(); is_dragging_state = False
                        is_lclicking_state = False; is_rclicking_state = False
                    elif is_listening_speech_active and not all_fingers_ext: # Exiting speech mode
                        is_listening_speech_active = False
                        speech_control_event.clear() # Stop listening
                        update_speech_status("")
                        last_speech_activation_time = time_now # Reset cooldown
                        print("Speech recognition DEACTIVATED.")

                    # Click and Drag Gestures (if not scrolling and not in speech mode)
                    if not is_scrolling_mode_active and not is_listening_speech_active:
                        # Drag Gesture: Index, Middle, Ring extended; Pinky curled
                        if pointer_finger_extended and middle_finger_extended and ring_finger_extended and (not pinky_finger_extended):
                            current_gesture_text = "DRAG"
                            if not is_dragging_state and (time_now - last_drag_release_time > GESTURE_COOLDOWN_AFTER_DRAG):
                                pyautogui.mouseDown()
                                is_dragging_state = True
                                print("Drag STARTED.")
                            is_lclicking_state = False; is_rclicking_state = False # Reset click states
                        # Release Drag: If drag was active and gesture changes
                        elif is_dragging_state and not (pointer_finger_extended and middle_finger_extended and ring_finger_extended):
                            pyautogui.mouseUp()
                            is_dragging_state = False
                            last_drag_release_time = time_now # Cooldown for gestures after drag
                            print("Drag RELEASED.")
                        # Right-Click Gesture: Index, Middle extended; Pinky, Ring curled (and not dragging)
                        elif not is_dragging_state and pointer_finger_extended and middle_finger_extended and \
                             (not pinky_finger_extended) and (not ring_finger_extended):
                            current_gesture_text = "R-CLICK Gesture"
                            if not is_rclicking_state and \
                               (time_now - last_rclick_time > RCLICK_COOLDOWN) and \
                               (time_now - last_drag_release_time > GESTURE_COOLDOWN_AFTER_DRAG):
                                pyautogui.rightClick()
                                print("Right-CLICKED.")
                                last_rclick_time = time_now
                                is_rclicking_state = True # Set state for cooldown
                            is_lclicking_state = False
                        # Left-Click Gesture: Index extended; Middle, Ring, Pinky curled (and not dragging)
                        elif not is_dragging_state and pointer_finger_extended and \
                             (not middle_finger_extended) and (not ring_finger_extended) and (not pinky_finger_extended):
                            current_gesture_text = "L-CLICK Gesture"
                            if not is_lclicking_state and \
                               (time_now - last_lclick_time > LCLICK_COOLDOWN) and \
                               (time_now - last_drag_release_time > GESTURE_COOLDOWN_AFTER_DRAG):
                                pyautogui.click()
                                print("Left-CLICKED.")
                                last_lclick_time = time_now
                                is_lclicking_state = True # Set state for cooldown
                            is_rclicking_state = False
                        else: # Default state: just tracking, reset click flags
                            if not is_dragging_state and not is_scrolling_mode_active and not mouse_tracking_paused:
                                current_gesture_text = "Tracking"
                            is_lclicking_state = False
                            is_rclicking_state = False
            else: # No hand detected
                # Reset states if hand is lost
                if is_dragging_state:
                    pyautogui.mouseUp()
                    is_dragging_state = False
                    print("Drag RELEASED (hand lost).")
                if is_listening_speech_active:
                    is_listening_speech_active = False
                    speech_control_event.clear()
                    update_speech_status("")
                    print("Speech DEACTIVATED (hand lost).")
                if is_scrolling_mode_active:
                    is_scrolling_mode_active = False
                    print("Scroll mode DEACTIVATED (hand lost).")

                if mouse_tracking_paused: # Maintain paused state if it was set
                    current_gesture_text = "PAUSED (No Hand)"
                else:
                    current_gesture_text = "No Hand Detected"

            # Display current gesture/status text on the image
            cv2.putText(image_display, current_gesture_text, (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Display speech recognition status
            if local_speech_status:
                speech_status_color = (0, 165, 255) # Default: Orange for "Listening", "Recognizing"
                if local_speech_status == "✓": # Green for success
                    speech_status_color = (0, 255, 0)
                elif local_speech_status in ["Timeout", "Unknown", "Error", "SysError"]: # Red for errors
                    speech_status_color = (0, 0, 255)
                cv2.putText(image_display, f"Speech: {local_speech_status}", (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speech_status_color, 2)

        # Quit application if 'q' is pressed
        if key_pressed == ord('q'):
            print("Quit key pressed. Exiting application...")
            break

        cv2.imshow(WINDOW_NAME, image_display)

finally:
    print("Cleaning up and exiting application...")
    # Signal the speech worker thread to stop and wait for it to join
    speech_thread_running = False
    speech_control_event.set() # Wake up the thread if it's waiting
    if 'speech_thread' in locals() and speech_thread.is_alive():
        speech_thread.join(timeout=1) # Wait for up to 1 second
        if speech_thread.is_alive():
            print("Warning: Speech thread did not terminate cleanly.")

    # Release resources
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'hands_model' in locals():
        hands_model.close()

    pyautogui.FAILSAFE = True # Re-enable PyAutoGUI failsafe
    print("Application closed successfully.")