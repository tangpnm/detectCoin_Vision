
#!/usr/bin/env
# -*- coding: utf-8 -*- 

import cv2
import numpy as np

cap = cv2.VideoCapture("video/Sample_Coin.mp4") 

while(cap.read()) : 

    ref, frame = cap.read() 
    roi = frame[:1000, 0:1320] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_blur = cv2.GaussianBlur(gray, (15,15), 0) 
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    result_img = closing.copy()
    contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 35000:
            continue
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(roi, ellipse, (0, 255, 0), 3) #color of cycle
        counter += 1

    cv2.putText(roi, str(counter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 2, cv2.LINE_AA) #color of number
    cv2.imshow("ShowCoin", roi)  

    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break 

cap.release() 
cv2.destroyAllWindows() 



