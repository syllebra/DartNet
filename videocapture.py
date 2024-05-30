from PIL import ImageGrab, features, Image
import numpy as np
import cv2

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
        pass

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
        import win32ui        
        win32gui.DeleteObject(self.saveBitMap.GetHandle())
        self.saveDC.DeleteDC()
        self.mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwndDC)

    # def __del__(self):
    #     if(self.hwnd is not None):
    #         self._clean_window_capture()


    def read(self, cv_format = False):
        if(self.hwnd is not None):
            return self._read_window(cv_format)
        
        if(self.box is None):
            return False, None     
        img = ImageGrab.grab(bbox=self.box) #bbox specifies specific region (bbox= x,y,width,height)
        if(not cv_format):
            return img
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return True, frame

if __name__ == "__main__":
    cap = ScreenVideoCapture(window_name="(Memu)")
    #cap.pick()
    cap.record()
    # success, frame = cap.read()
    # print(success, frame)
    cv2.destroyAllWindows()