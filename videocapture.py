from PIL import ImageGrab
import numpy as np
import cv2

img = ImageGrab.grab() #bbox specifies specific region (bbox= x,y,width,height)
img_np = np.array(img)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


cv2.imshow("zone", frame)

box = [0,0,0,0]
dragging = False

def update_image():
    global dragging
    img = frame.copy()
    color = (255,0,0) if not dragging else (0,255,255)
    cv2.circle(img,(box[0],box[1]),3,color = color,thickness=4)
    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color = color,thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('zone', img)
    cv2.waitKey(1)

def Mouse_Event(event, x, y, flags, param):
    global dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        box[0]=x
        box[1]=y
        update_image()
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        update_image()
    elif(dragging):
        box[2]=x
        box[3]=y
        update_image()

# set Mouse Callback method
cv2.setMouseCallback('zone', Mouse_Event)
cv2.waitKey(0)


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
size = (box[2]-box[0], box[3]-box[1]) 
vid = cv2.VideoWriter('output.avi', fourcc, 21, size)

while(True):
    img = ImageGrab.grab(bbox=box) #bbox specifies specific region (bbox= x,y,width,height)
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    vid.write(frame)
    cv2.imshow("capture", frame)
    key = cv2.waitKey(1000//21)
    if(key == ord('q')):
        break

cv2.destroyAllWindows()