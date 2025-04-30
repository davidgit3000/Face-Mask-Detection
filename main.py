from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime

# Load your trained model
best_face_mask_model = load_model('./models/best_face_mask_model1.h5')

# Define the labels
labels = ['Mask Worn Correctly', 'Mask Worn Incorrectly', 'No Mask']

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the image 
def preprocess_image(image):
    # Resize image to stabilize frame size
    image = cv2.resize(image, (640, 480))
    
    # Increase brightness & contrast
    image = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.05,       # more sensitive
        minNeighbors=3,         # fewer strict matches
        minSize=(30, 30))
    
    # If no faces are detected, return None
    if len(faces) == 0:
        return None, None
    
    # Get the first detected face
    (x, y, w, h) = faces[0]
    
    # Extract the face region from the image
    face = image[y:y+h, x:x+w]
    
    # Resize the face to the required input size for the model (128x128)
    face = cv2.resize(face, (128, 128))
    
    # Convert the face to a numpy array
    face = np.array(face)
    
    # Normalize pixel values to [0, 1]
    face = face / 255.0
    
    # Expand dimensions to create batch size of 1
    face = np.expand_dims(face, axis=0)
    
    # Return the preprocessed face and the face coordinates
    return face, (x, y, w, h)

# Function to predict the mask type
def predict_mask_type(frame):
    # Preprocess the face
    face, coords = preprocess_image(frame)
    
    # If no face is detected, return None
    if face is None:
        return None, None, None
    
    # Use the model to predict the mask type
    prediction = best_face_mask_model.predict(face)
    
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction)
    
    # Get the probabilities as percentages
    probabilities = prediction[0] * 100
    
    # Return the predicted class, coordinates, and probabilities
    return predicted_class, coords, probabilities

# Function to display the results
def display_results(image, predicted_class, coords, probabilities):
    if coords is None:
        return image
    
    # Get the coordinates
    x, y, w, h = coords
    
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Calculate positions for text
    text_x = x + 5  # Small margin from left edge
    start_y = y - 10  # Just above the face
    font_scale = 0.7
    thickness = 2
    outline_color = (0, 0, 0)
    
    # Display predictions for each label with percentage
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        # Position text going upward from the face
        text_y = start_y - (30 * (2-i))  # Reverse order, more spacing
        
        # Create text with percentage
        text = f'{label}: {prob:.1f}%'
        
        # Determine color based on prediction
        color = (0, 255, 0) if i == predicted_class else (255, 255, 255)
        
        # Draw thicker outline for better visibility
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
            cv2.putText(image, text, 
                        (text_x + dx, text_y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, outline_color, 
                        thickness + 2)
        
        # Draw the text in color
        cv2.putText(image, text, 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color,
                    thickness)
    
    return image

# Function to run the real-time face mask detection
def run_real_time_face_mask_detection():
    # Create output directory if it doesn't exist
    output_dir = 'captured_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Loop through the frames
    while True:
        # Read the frame
        ret, frame = cap.read()
        # If the frame is not read, break the loop
        if not ret:
            break
        
        # Get the predicted class, face coordinates, and probabilities
        predicted_class, coords, probabilities = predict_mask_type(frame)
        
        # If no face is detected, show the frame without annotations
        if predicted_class is None:
            cv2.imshow('Real-time Face Mask Detection', frame)
        else:
            # Display the results with probabilities
            frame = display_results(frame, predicted_class, coords, probabilities)
            # Show the frame
            cv2.imshow('Real-time Face Mask Detection', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # If 's' is pressed and a face is detected, save the frame
        if key == ord('s') and predicted_class is not None:
            # Generate filename with timestamp and prediction
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prediction = labels[predicted_class].replace(' ', '_')
            confidence = probabilities[predicted_class]
            filename = f'{timestamp}_{prediction}_{confidence:.1f}.jpg'
            filepath = os.path.join(output_dir, filename)
            
            # Save the frame
            cv2.imwrite(filepath, frame)
            print(f'Frame saved as: {filename}')
        
        # If 'q' is pressed, break the loop
        elif key == ord('q'):
            break
    
    # Release the camera
    cap.release()
    # Destroy all windows
    cv2.destroyAllWindows()

# Run the real-time face mask detection
run_real_time_face_mask_detection()