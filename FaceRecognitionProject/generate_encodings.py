import os
import face_recognition
import pickle

def generate_encodings():
    known_face_encodings = []
    known_face_names = []
    
    for person_name in os.listdir("dataset"):
        person_dir = os.path.join("dataset", person_name)
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
    
    # Save encodings to a file
    with open("encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
    print("Encodings generated and saved.")

# Generate encodings
generate_encodings()