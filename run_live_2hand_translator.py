import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# --- 1. Load Your Trained 2-Hand Model and Labels ---
print("Loading 2-Hand model and labels...")
try:
    # --- MODIFIED: Load your new 2-hand model ---
    model = tf.keras.models.load_model('isl_static_2hand_model.keras')
    labels = np.load('isl_static_2hand_labels.npy')
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    print("Please make sure 'isl_static_2hand_model.keras' and 'isl_static_2hand_labels.npy'")
    print("are in the same folder as this script.")
    exit()

# --- 2. Initialize MediaPipe Hands (MODIFIED FOR 2 HANDS) ---
print("Initializing MediaPipe Hands (for 2 hands)...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # We are on a live video stream
    max_num_hands=2,         # <--- MODIFIED: Now detects 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 3. Landmark Extraction Function (MODIFIED FOR 2 HANDS) ---
# This MUST be the same function as in your feature extraction script
def extract_2hand_landmarks(image, model):
    """
    Extracts 126 landmarks (2 hands * 21 * 3) from a single image frame.
    Returns a flattened, *normalized* NumPy array of size 126.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True # Set back to writable for drawing

    # Create an empty array for all 126 features
    landmarks_array = np.zeros(2 * 21 * 3) 
    
    if results.multi_hand_landmarks:
        # Loop through all detected hands
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            try:
                hand_label = results.multi_handedness[hand_index].classification[0].label
            except Exception as e:
                # Fallback if label is not found
                continue 
                
            # --- Normalization (relative to each hand's *own* wrist) ---
            landmarks = []
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z
            
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - wrist_x)
                landmarks.append(lm.y - wrist_y)
                landmarks.append(lm.z - wrist_z)
            
            flat_landmarks = np.array(landmarks).flatten()
            
            # --- Fill the correct slots ---
            if hand_label == "Left":
                landmarks_array[0:63] = flat_landmarks  # First 63 slots
            elif hand_label == "Right":
                landmarks_array[63:126] = flat_landmarks # Second 63 slots
    
    # Return the 126 features, and the results object for drawing
    return landmarks_array, results

# --- 4. Start OpenCV Webcam Loop ---
print("Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.8 # Only show predictions above 80%
current_prediction = "..."

with hands as hands_model:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1) # Flip for selfie view
        display_frame = frame.copy()

        # --- MediaPipe processing ---
        # Use the new 2-hand extraction function
        landmarks_data, results = extract_2hand_landmarks(frame, hands_model)
        
        if results.multi_hand_landmarks:
            # Draw landmarks for all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- Make a Prediction ---
        # We can predict on every frame.
        # The 126-feature array will be mostly zeros if no hands are found,
        # but the model will learn this means "no sign".
        input_data = np.expand_dims(landmarks_data, axis=0)
        prediction_array = model.predict(input_data, verbose=0)[0]
        confidence = np.max(prediction_array)
        
        if confidence > CONFIDENCE_THRESHOLD:
            prediction_index = np.argmax(prediction_array)
            predicted_sign = labels[prediction_index]
            current_prediction = f"{predicted_sign} ({confidence*100:.0f}%)"
        else:
            current_prediction = "..."
            
        if not results.multi_hand_landmarks:
            current_prediction = "No Hand(s) Detected"

        # --- Draw the prediction on the screen ---
        cv2.rectangle(display_frame, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, current_prediction, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        cv2.imshow('ISL Static 2-Hand Translator', display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

print("Shutting down...")
cap.release()
cv2.destroyAllWindows()