import cv2
from board import Board
from tools import ask_json_url, add_transparent_image
from callback_thread import CommandsAndCallbackThread

from granboard import GranboardApi


board = Board("generator/3D/Boards/canaveral_t520.json")
button_overlay = cv2.imread('images/pause.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
button_overlay[:,:,3] = (button_overlay[:,:,2]) *0.65


api = GranboardApi()
# example using BaseThread with callback
def score(x,y):
    scores = board.get_dart_scores(board.image_cal_pts,([[x,y]]))
    sc = scores[0]
    api.score(sc)

def drawfunction(event,x,y,flags,param):
    #if event == cv2.EVENT_MOUSEMOVE:
    if event == cv2.EVENT_LBUTTONUP:
        if(api.button_state):
            api.click_button()
            return
        # if(debug_test is not None):
            # draw_inference_boxes(debug_test,[{"x1":x,"y1":y,"x2":x,"y2":y,"conf":1.0,"cls":0}],filter=[0],detector = detector)
            # cv2.imshow("Dbg", debug_test)
        
        score(x,y)
        #cv2.displayOverlay("Dbg",scores[0])
cv2.namedWindow('Scorer', cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow("Scorer", 800, 800) 
cv2.setMouseCallback('Scorer', drawfunction)

img = cv2.imread("generator/3D/Boards/canaveral_t520.jpg")

while(True):
    if(api.button_state):
        tmp = img.copy()
        offx = 1024-button_overlay.shape[1]//2
        offy = 1024-button_overlay.shape[0]//2
        add_transparent_image(tmp, button_overlay, offx, offy)
        cv2.imshow("Scorer",tmp)
    else:
        cv2.imshow("Scorer",img)

    key = cv2.waitKey(100)
    api.ask_button_state()
    if(api.button_state and key == ord(' ')):
        api.click_button()
    elif(key == 27 or key == ord('q')):
        break
print("Stopping thread..")
api.stop()

