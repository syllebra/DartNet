import cv2
from board import Board
from tools import ask_json_url, add_transparent_image
from callback_thread import CommandsAndCallbackThread

granboard_url = "http://192.168.33.75:8822"

board = Board("generator/3D/Boards/canaveral_t520.json")
command_map = {
    "S1": "2.3@",
    "S2": "9.1@",
    "S3": "7.1@",
    "S4": "0.1@",
    "S5": "5.1@",
    "S6": "1.0@",
    "S7": "11.1@",
    "S8": "6.2@",
    "S9": "9.3@",
    "S10": "2.0@",
    "S11": "7.3@",
    "S12": "5.0@",
    "S13": "0.0@",
    "S14": "10.3@",
    "S15": "3.0@",
    "S16": "11.0@",
    "S17": "10.1@",
    "S18": "1.2@",
    "S19": "6.1@",
    "S20": "3.3@",
    "S1OUT": "2.5@",
    "S2OUT": "9.2@",
    "S3OUT": "7.2@",
    "S4OUT": "0.5@",
    "S5OUT": "5.4@",
    "S6OUT": "1.3@",
    "S7OUT": "11.4@",
    "S8OUT": "6.5@",
    "S9OUT": "9.5@",
    "S10OUT": "2.2@",
    "S11OUT": "7.5@",
    "S12OUT": "5.5@",
    "S13OUT": "0.4@",
    "S14OUT": "10.5@",
    "S15OUT": "3.2@",
    "S16OUT": "11.5@",
    "S17OUT": "10.2@",
    "S18OUT": "1.5@",
    "S19OUT": "6.3@",
    "S20OUT": "3.5@",
    "D1": "2.6@",
    "D2": "8.2@",
    "D3": "8.4@",
    "D4": "0.6@",
    "D5": "4.6@",
    "D6": "4.4@",
    "D7": "8.6@",
    "D8": "6.6@",
    "D9": "9.6@",
    "D10": "4.3@",
    "D11": "7.6@",
    "D12": "5.6@",
    "D13": "4.5@",
    "D14": "10.6@",
    "D15": "4.2@",
    "D16": "11.6@",
    "D17": "8.3@",
    "D18": "1.6@",
    "D19": "8.5@",
    "D20": "3.6@",
    "T1": "2.4@",
    "T2": "9.0@",
    "T3": "7.0@",
    "T4": "0.3@",
    "T5": "5.2@",
    "T6": "1.1@",
    "T7": "11.2@",
    "T8": "6.4@",
    "T9": "9.4@",
    "T10": "2.1@",
    "T11": "7.4@",
    "T12": "5.3@",
    "T13": "0.2@",
    "T14": "10.4@",
    "T15": "3.1@",
    "T16": "11.3@",
    "T17": "10.0@",
    "T18": "1.4@",
    "T19": "6.0@",
    "T20": "3.4@",
    "SBULL": "8.0@",
    "DBULL": "4.0@",
    "MISS": "MISS@",
    "OUT": "OUT@"
}

button_state=False

button_overlay = cv2.imread('images/pause.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
button_overlay[:,:,3] = (button_overlay[:,:,2]) *0.65

def send(cmd=""):
    if(cmd == "button_state"):
        return ask_json_url(granboard_url+"/hit?cb")
    elif(cmd == "button_click"):
        return ask_json_url(granboard_url+"/hit?cb=click")
    else:
        return ask_json_url(granboard_url+"/hit?cmd="+cmd)

def cb(r):
    global button_state
    button_state = r["button_state"]
    print(f"callback function called {r}")

# example using BaseThread with callback
thread_poll = CommandsAndCallbackThread(name='poll_thread')
thread_poll.enqueue(send, cb, cmd="button_state", repeat=500)

thread = CommandsAndCallbackThread(name='score_thread')

def drawfunction(event,x,y,flags,param):
    
        
    #if event == cv2.EVENT_MOUSEMOVE:
    if event == cv2.EVENT_LBUTTONUP:
        if(button_state):
            thread.enqueue(send, None, cmd="button_click")
            return
        # if(debug_test is not None):
            # draw_inference_boxes(debug_test,[{"x1":x,"y1":y,"x2":x,"y2":y,"conf":1.0,"cls":0}],filter=[0],detector = detector)
            # cv2.imshow("Dbg", debug_test)
        
        scores = board.get_dart_scores(board.image_cal_pts,([[x,y]]))

        sc = scores[0]
        if(sc == "0"):
            sc = "OUT"
        elif(sc == "DB"):
            sc = "DBULL"
        elif(sc == "B"):
            sc = "SBULL"
        elif("D" not in sc and "T" not in sc):
            sc = f"S{sc}OUT"
        command = command_map[sc]
        print(scores[0],"=>",command)
        thread.enqueue(send, None, cmd=command)

        #cv2.displayOverlay("Dbg",scores[0])
cv2.namedWindow('Scorer', cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow("Scorer", 800, 800) 
cv2.setMouseCallback('Scorer', drawfunction)

img = cv2.imread("generator/3D/Boards/canaveral_t520.jpg")

while(True):
    if(button_state):
        tmp = img.copy()
        offx = 1024-button_overlay.shape[1]//2
        offy = 1024-button_overlay.shape[0]//2
        add_transparent_image(tmp, button_overlay, offx, offy)
        cv2.imshow("Scorer",tmp)
    else:
        cv2.imshow("Scorer",img)

    key = cv2.waitKey(50)
    if(button_state and key == ord(' ')):
        thread.enqueue(send, None, cmd="button_click")
    elif(key == 27 or key == ord('q')):
        break
print("Stopping thread..")
thread.wait_stop()
thread_poll.wait_stop()

