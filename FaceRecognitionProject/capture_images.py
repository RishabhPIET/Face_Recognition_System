import cv2
import os

def capture_images(name):
    video_capture = cv2.VideoCapture(0)  # Open the webcam
    count = 0  # Counter for images
    
    # Create a directory for the person if it doesn't exist
    if not os.path.exists(f"dataset/{name}"):
        os.makedirs(f"dataset/{name}")
    
    print(f"Capturing images for {name}. Press 'c' to capture and 'q' to quit.")
    
    while True:
        ret, frame = video_capture.read()  # Read a frame from the webcam
        cv2.imshow('Video', frame)  # Display the frame
        
        # Capture image when 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            image_path = f"dataset/{name}/{name}_{count}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Saved {image_path}")
            count += 1
        
        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the window

# Ask for the person's name
name = input("Enter the name of the person: ")
capture_images(name)