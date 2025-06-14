from PIL import ImageGrab, features, Image
import numpy as np
import cv2
import requests
import json
import asyncio
import aiohttp
import sys
import time

if (sys.platform.startswith('win')
        and sys.version_info[0] == 3
        and sys.version_info[1] >= 8):
    policy = asyncio.WindowsSelectorEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

class IPWebCamVideoCapture():
    ''' cv2 compatible IP Webcam android app  capture class'''
    def __init__(self, ip="127.0.0.1", port=8080, login ="",password="", poll_delay=500) -> None:
        #url="https://92.168.33.35:8080/shot.jpg")
        cred = login if password == "" else f"{login}:{password}"
        if(cred != ""):
            cred = cred+"@"
        self.base_url = f"https://{cred}{ip}:{port}"
        self.stream_url = f"{self.base_url}/video"
        self.status_url = f"{self.base_url}/status.json"
        self.sensors_url = f"{self.base_url}/sensors.json"
        self.status = {}
        self.sensors = {}

        self.poll_delay = poll_delay
        self.last_read = time.time()
        #asyncio.run(self.poll())

        self.capture = cv2.VideoCapture(self.stream_url)
        _, self.img = self.read()


    def _ask_json_url(self, url):
        resp = requests.get(url, verify=False)
        if(resp.status_code < 200 or resp.status_code>=300):
            return None
        data = json.loads(resp.content)
        return data
    
    # async def poll(self):
    #     # r = self._ask_json_url(self.status_url)
    #     # self.status = {} if r is None else r
    #     # print(self.status)

    #     async with aiohttp.ClientSession(trust_env = True) as session:
    #         async with session.get(self.status_url, verify_ssl=False) as response:
    #             r = await response.json()
    #             self.status = {} if r is None else r

                
    #             #print(self.status)
    #     # async with aiohttp.ClientSession(trust_env = True) as session:
    #     #     async with session.get(self.sensors_url, verify_ssl=False) as response:
    #     #         r = await response.json()
    #     #         self.sensors = {} if r is None else r


    def get(self, flag: int):
        if(flag == cv2.CAP_PROP_FRAME_WIDTH):
            return 0 if self.img is None else self.img.shape[1]
        elif(flag == cv2.CAP_PROP_FRAME_HEIGHT):
            return 0 if self.img is None else self.img.shape[0]
        return 0
    
    def set(self, flag, val):
        return True

    def release(self):
        pass

    def read(self, cv_format = True):
        # if(self.poll_delay >=0 and time.time() -self.last_read > self.poll_delay *0.001):
        #     asyncio.run(self.poll())
        #     self.last_read = time.time()
        success, img = self.capture.read()
        return success, img

class ScreenVideoCapture():
    ''' cv2 compatible screen video capture class'''
    def __init__(self, pick=False, box=None, window_name=None) -> None:
        self.box = None
        self.hwnd = None
        if(pick):
            self.pick()


        if(window_name is not None):
            self._init_window_capture(window_name)

    def get(self, flag: int):
        if(flag == cv2.CAP_PROP_FRAME_WIDTH):
            return 0 if self.box is None else abs(self.box[2]-self.box[0])
        elif(flag == cv2.CAP_PROP_FRAME_HEIGHT):
            return 0 if self.box is None else abs(self.box[2]-self.box[0])
        return 0
    
    def set(self, flag, val):
        return True

    def release(self):
        pass



    def pick(self):
        img = ImageGrab.grab() #bbox specifies specific region (bbox= x,y,width,height)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        cv2.imshow("zone", frame)

        box = [0,0,0,0]
        self.dragging = False

        def _update_image():
            global dragging
            img = frame.copy()
            color = (255,0,0) if not self.dragging else (0,255,255)
            cv2.circle(img,(box[0],box[1]),3,color = color,thickness=4)
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color = color,thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow('zone', img)
            cv2.waitKey(1)

        def _Mouse_Event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                box[0]=x
                box[1]=y
                _update_image()
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
                _update_image()
            elif(self.dragging):
                box[2]=x
                box[3]=y
                _update_image()
        
        self.box = box
        # set Mouse Callback method
        cv2.setMouseCallback('zone', _Mouse_Event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def record(self, file="output.avi", fps=21, codec='MJPG'):
        if(self.box is None and self.hwnd is None):
            return
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        size = (self.box[2]-self.box[0], self.box[3]-self.box[1]) 
        vid = cv2.VideoWriter(file, fourcc, fps, size)

        while(True):
            success, img = self.read(True)
            if(success):
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vid.write(frame)
                cv2.imshow("capture", frame)
            key = cv2.waitKey(1000//fps)
            if(key == ord('q')):
                break


    def _init_window_capture(self, window_name):
        import win32gui
        import win32ui
        self.hwnd = win32gui.FindWindow(None, window_name)

        # Uncomment the following line if you use a high DPI display or >100% scaling size
        # windll.user32.SetProcessDPIAware()

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #left, top, right, bot = win32gui.GetClientRect(hwnd)
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        w = right - left
        h = bot - top
        self.box = (left, top, right, bot)

        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC  = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()

        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, w, h)

        self.saveDC.SelectObject(self.saveBitMap)

    def _read_window(self, cv_format):
        from ctypes import windll

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        result = windll.user32.PrintWindow(self.hwnd, self.saveDC.GetSafeHdc(), 3)

        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)
        ret = None
        if(result == 1):
            ret = np.array(im) if cv_format else im
        return result==1, ret
        
    def _clean_window_capture(self):
        import win32gui
        win32gui.DeleteObject(self.saveBitMap.GetHandle())
        self.saveDC.DeleteDC()
        self.mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwndDC)

    # def __del__(self):
    #     if(self.hwnd is not None):
    #         self._clean_window_capture()


    def read(self, cv_format = True):
        if(self.hwnd is not None):
            return self._read_window(cv_format)
        
        if(self.box is None):
            return False, None     
        img = ImageGrab.grab(bbox=self.box) #bbox specifies specific region (bbox= x,y,width,height)
        if(not cv_format):
            return True, img
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return True, frame

if __name__ == "__main__":
    import time, playsound

    cap = IPWebCamVideoCapture("192.168.33.35","8080","BilboX", "testip42")
    while(True):
        success, img = cap.read()
        cv2.putText(img, f'Battery: {cap.status["deviceInfo"]["batteryPercent"]} Charging:{cap.status["deviceInfo"]["batteryCharging"]}',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,0),1)
        cv2.imshow("Capture", img)
        if(cv2.waitKey(1)==27):
            break