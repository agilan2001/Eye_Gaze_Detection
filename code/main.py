import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(r'code\images\img1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, minNeighbors=10)
for (x,y,w,h) in faces:
    # Draw rectange around detected Face
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    # Assume eyes to be at 25% to 50% of the Face
    eyes = np.uint16(np.around([(0, 0.25*h, 0.5*w, 0.25*h),(0.5*w, 0.25*h, w, 0.25*h)]))
    for (ex,ey,ew,eh) in eyes:

        # Draw rectange around assumed Eye region
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        eye_region_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_region_color = roi_color[ey:ey+eh, ex:ex+ew]

        # Use Hough Circles to detect Circles - Pupils
        detected_circles = cv2.HoughCircles(eye_region_gray, cv2.HOUGH_GRADIENT, 0.8, 10, 
            param1 = 220, param2 = 20)
        
        cv2.imshow('eye_reg_gray', eye_region_gray)
        if detected_circles is not None :
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw circle around the pupils
                cv2.circle(eye_region_color, (a, b), r, (0, 255, 0), 2)
        

cv2.imshow('img',img)
cv2.waitKey()
