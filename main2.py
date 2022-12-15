import cv2
from tracker import *
cap = cv2.VideoCapture("./video.mp4")
import numpy as np
import time
#create tracker
tracker= EuclideanDistTracker()
#OBJECT DETECTION
count_line_pos=550
offset =6
counter=0
start=0
t=20
def centroid(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
detect=[]

def check_time(counter):
    #if (stop-start) >= 10:
    if counter >5 and counter<10:
        print("Decrease speed limit")
    else:
        print("increase speed limit")
        #counter=0
    #return counter
def check_speed(start,stop):
    time_of_vehicle= stop-start
    return 100//time_of_vehicle
object_detector= cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, frame = cap.read()
    height, width,_ = frame.shape
    #region of interest
    roi = frame[340:720,500:800]
    
    blur=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(3,3),5)
    img_sub= object_detector.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    contours,h= cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    
            
    
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter=(w>=80) and (h>=80)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"VEHICLES "+str(counter),(x,y-20),cv2.FONT_HERSHEY_PLAIN,1,(255,244,0),5)
        
        center = centroid(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4,(0,0,255),-1)
        
        for (x,y) in detect:
            if y<(count_line_pos+offset) and y> (count_line_pos-offset):
                start= time.perf_counter()
                counter+=1
            cv2.line(frame,(25,count_line_pos),(1200,count_line_pos),(0,127,255),3)
            stop = time.perf_counter()
            speed = check_speed(start,stop)
            detect.remove((x,y))
            print("Vehicle found",counter)
            #counter =check_time(counter)
    print("Vehicle ",counter," Speed ",speed)
    cv2.putText(frame,"VEHICLES "+str(counter),(450,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
    cv2.putText(frame,"Speed "+str(speed)+" m/s",(680,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),5)
    
    
    #cv2.imshow("mask",mask)
    #cv2.imshow("roi",roi)
    
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(10)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()