import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance

video_capture = cv2.VideoCapture(0)

image = cv2.imread('C:\\Users\\Ahana Drall\\Downloads\\amitabh_1.jpg')
amitabh = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.imread('C:\\Users\\Ahana Drall\\Downloads\\abhishek_1.jpg')
abhishek = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.imread('C:\\Users\\Ahana Drall\\Downloads\\myimage25.jpeg')
ahana = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.imread('C:\\Users\\Ahana Drall\\Downloads\\tanmay.jpg')
tanmay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

amitabh_encoding = face_recognition.face_encodings(amitabh)[0]
abhishek_encoding = face_recognition.face_encodings(abhishek)[0]
ahana_encoding = face_recognition.face_encodings(ahana)[0]
tanmay_encoding = face_recognition.face_encodings(tanmay)[0]

known_face_encodings = [amitabh_encoding, abhishek_encoding, ahana_encoding, tanmay_encoding]

known_face_names = ["amitabh bachchan", "abhishek bachchan", "ahana", "tanmay"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    ret, frame = video_capture.read()
    
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    
    
    if process_this_frame:
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            
            #matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
            
            #name = "Unknown"
            
            #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            
            #if matches[best_match_index]:
                #name = known_face_names[best_match_index]
                
            #face_names.append(name)
            
            results = []
            for known_face_encoding in known_face_encodings:
                d = distance.euclidean(known_face_encoding, face_encoding)
                results.append(d)
            threshold = 0.6
            results = np.array(results) <= threshold
    
            name = 'Unknown'
    
            if results[0]:
                name = 'Amitabh Bachchan'
            elif results[1]:
                name = 'Abhishek bachchan'
            elif results[2]:
                name = 'Ahana'
            elif results[3]:
                name = 'Tanmay'
            #print (f"found {name} in the photo!")
            
    process_this_frame = not process_this_frame
    
    
    #for (top, right, bottom, left), name in zip(face_locations, face_names):
    for (top, right, bottom, left) in face_locations:   
        top *= 4
        right *=4
        bottom *= 4
        left *= 4
        
        
        cv2.rectangle(frame, (left, top), (right, bottom), (255,255,255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
        