# Hand Gesture Mouse & Speech Control

Control your computer's mouse cursor, perform clicks, drag, scroll, and dictate text using hand gestures captured via your webcam. This project leverages MediaPipe for real-time hand tracking and SpeechRecognition for voice commands.

![Default Hand Pose](images/Default.png)
*Default tracking state*

## Features

*   **Mouse Movement:** Control the cursor by moving your hand.
*   **Left Click:** Perform a left click with a specific gesture.
*   **Right Click:** Perform a right click with a specific gesture.
*   **Drag & Drop:** Click and drag items or select text.
*   **Scroll Mode:** Activate a dedicated mode for scrolling up and down.
*   **Speech-to-Text:** Activate speech recognition to type with your voice.
*   **Pause/Resume Tracking:** Temporarily pause mouse control.
*   **Calibration System:** Calibrate the hand tracking area to your screen for better accuracy.
*   **Adjustable Camera Zoom & Mapping Inset:** Fine-tune camera input and cursor mapping via UI trackbars.

## Gestures & Demonstrations

The system recognizes the following hand gestures:

1.  **Left Click:** Index finger extended, other fingers curled.
    ![Left Click Gesture](images/Left_Click.png)

2.  **Right Click:** Index and Middle fingers extended, Ring and Pinky fingers curled.
    ![Right Click Gesture](images/Right_Click.png)

3.  **Drag:** Index, Middle, and Ring fingers extended, Pinky finger curled. Move your hand to drag. Gesture release stops dragging.
    ![Drag Gesture](images/Drag.png)

4.  **Scroll Mode:** Middle, Ring, and Pinky fingers extended, Index finger curled. Move your wrist up/down to scroll. Releasing gesture deactivates scroll mode.
    ![Scroll Gesture](images/Scroll.png)

5.  **Pause Tracking:** Index and Pinky fingers extended, Middle and Ring fingers curled. Repeat gesture to resume.
    ![Pause Gesture](images/Pause.png)

6.  **Speech-to-Text Activation:** All fingers extended. Releasing gesture deactivates speech mode.
    ![Speech Gesture](images/Speech.png)

## Requirements

*   Python 3.7+
*   A webcam
*   Internet connection (for Google Speech Recognition API)

You'll need the following Python libraries:

*   `opencv-python`
*   `mediapipe`
*   `pyautogui`
*   `numpy`
*   `SpeechRecognition`
*   `PyAudio` (often required by SpeechRecognition for microphone access)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install opencv-python mediapipe pyautogui numpy SpeechRecognition PyAudio
    ```
    *Note: If `PyAudio` installation fails, you might need to install system dependencies first using sudo or brew (e.g., `portaudio` on Linux: `sudo apt-get install portaudio19-dev`).*

4.  **Ensure Microphone Access:** Your operating system must allow Python/your terminal application to access the microphone for speech recognition to work.

## Usage

1.  **Run the script:**
    ```bash
    python hand_mouse.py
    ```

2.  **Initial Setup Window:**
    *   A window titled "Hand Tracking Mouse Control" will appear, showing your webcam feed.
    *   **Zoom & Inset Trackbars:** Adjust these before calibration if needed.
        *   `Zoom (x10)`: Digitally zooms into the center of the camera feed.
        *   `Inset (%)`: Reduces the effective mapping area within the calibrated bounds.

3.  **Calibration:**
    *   The application starts in `CALIBRATING_START` state.
    *   Press **'c'** to begin calibration:
        *   **Top-Left Point:** Aim your index finger tip at the green dot on the top-left of the camera feed window or wherever you want the top left of your functional window to be and press **SPACE**.
        *   **Bottom-Right Point:** Aim your index finger tip at the green dot on the bottom-right or wherever you want the bottom right of your functional window to be and press **SPACE**.
    *   Press **'s'** to skip calibration and use default full-frame mapping (with a small default inset).
    *   Calibration defines the area of your camera feed that will be mapped to your entire screen.

4.  **Controlling the Mouse:**
    *   Once calibrated or skipped, the application enters `RUNNING` state.
    *   Perform the gestures listed above to control your mouse, scroll, and use speech-to-text.
    *   The current gesture or state (e.g., "PAUSED", "SCROLL MODE", "Speech: Listening...") will be displayed on the webcam feed window.

5.  **Exiting:**
    *   Press **'q'** in the webcam window to close the application.

## How It Works

*   **OpenCV:** Captures video from the webcam.
*   **MediaPipe Hands:** Detects and tracks hand landmarks in real-time from the video feed.
*   **Gesture Recognition:** Custom logic analyzes the angles between hand landmarks to identify specific finger poses (extended, curled).
*   **Coordinate Mapping:** Calibrated hand landmark positions (typically the index finger's MCP joint for cursor movement, or wrist for scrolling) are mapped to screen coordinates. Smoothing is applied for less jittery cursor movement.
*   **PyAutoGUI:** Programmatically controls the mouse (move, click, scroll) and keyboard (type recognized text).
*   **SpeechRecognition & PyAudio:** Captures audio from the microphone when the speech gesture is active and uses Google's Web Speech API to transcribe it to text.

## Configuration (In-Code)

The Python script contains several constants at the top that can be tweaked for finer control:

*   `WEBCAM_ID`: Change if you have multiple webcams (e.g., `0`, `1`).
*   `SMOOTHING_FACTOR`: Affects mouse cursor smoothness.
*   `*_COOLDOWN`: Various cooldown timers to prevent accidental rapid actions.
*   `FINGER_*_ANGLE_THRESHOLD`: Angle thresholds for detecting if fingers are extended or curled.
*   `SCROLL_SENSITIVITY`: Adjusts scrolling speed.
