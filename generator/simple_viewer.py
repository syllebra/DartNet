import tkinter as tk
import tkinter.font as TkFont
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import numpy as np
import os
import json
import cv2

class SimpleViewer(tk.Frame):
    def __init__(self, master=None, img = None, menu=True):
        super().__init__(master)

        self.master.geometry("600x400") 

        self.pil_image = None
        self.my_title = "Python Image Viewer"
        if(menu == True):
            self.create_menu()

        self.master.title(self.my_title)
    
        self.create_widget()
        self.reset_transform()

        if(img is not None):
            self.set_pil_image(img)
    
    def menu_open_clicked(self, event=None):

        filename = tk.filedialog.askopenfilename(
            filetypes = [("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = os.getcwd()
            )

        self.set_image(filename)

    def menu_quit_clicked(self):
        self.master.destroy() 

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
 
        self.file_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Open", command = self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command = self.menu_quit_clicked)

        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked)

        self.master.config(menu=self.menu_bar)
 
    def create_widget(self):
        frame_statusbar = tk.Frame(self.master, bd=1, relief = tk.SUNKEN)
        self.label_image_info = tk.Label(frame_statusbar, text="image info", anchor=tk.E, padx = 5)
        self.label_image_pixel = tk.Label(frame_statusbar, text="(x, y)", anchor=tk.W, padx = 5)
        self.label_image_info.pack(side=tk.RIGHT)
        self.label_image_pixel.pack(side=tk.LEFT)
        frame_statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self.master, background="black")
        self.canvas.pack(expand=True,  fill=tk.BOTH)

        self.master.bind("<Button-1>", self.mouse_down_left)                   # MouseDown
        self.master.bind("<B1-Motion>", self.mouse_move_left)                  # MouseDrag
        self.master.bind("<Motion>", self.mouse_move)                          # MouseMove
        self.master.bind("<Double-Button-1>", self.mouse_double_click_left)    # MouseDoubleClick
        self.master.bind("<MouseWheel>", self.mouse_wheel)                     # MouseWheel

    def set_image(self, filename):
        if not filename:
            return
        
        self.filename = filename
        img = Image.open(filename)
        #os.chdir(os.path.dirname(filename))
        self.set_pil_image(img,  os.path.basename(filename))

        self.redraw_image()

    def set_pil_image(self, img, title="Imgviewer"):
        self.pil_image = img
        self.orig_pil_image = img.copy()
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.draw_image(self.pil_image)

        self.master.title(self.my_title + " - " + title)
        self.label_image_info["text"] = f"{self.pil_image.format} : {self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"


    def mouse_down_left(self, event):
        self.__old_event = event

    def mouse_move_left(self, event):
        if (self.pil_image == None):
            return
        self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
        self.redraw_image()
        self.__old_event = event

    def mouse_move(self, event):
        if (self.pil_image == None):
            return
        
        image_point = self.to_image_point(event.x, event.y)
        if image_point is not None:
            self.label_image_pixel["text"] = (f"({image_point[0]:.2f}, {image_point[1]:.2f})")
        else:
            self.label_image_pixel["text"] = ("(--, --)")

    def mouse_double_click_left(self, event):
        if self.pil_image == None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image()

    def mouse_wheel(self, event):
        if self.pil_image == None:
            return

        if event.state != 9:
            if (event.delta < 0):
                self.scale_at(1.25, event.x, event.y)
            else:
                self.scale_at(0.8, event.x, event.y)
        else:
            if (event.delta < 0):
                self.rotate_at(-5, event.x, event.y)
            else:
                self.rotate_at(5, event.x, event.y)     
        self.redraw_image()
        
    def reset_transform(self):
        self.mat_affine = np.eye(3)

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale:float):
        mat = np.eye(3)
        mat[0, 0] = scale
        mat[1, 1] = scale

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale_at(self, scale:float, cx:float, cy:float):
        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def rotate(self, deg:float):
        mat = np.eye(3)
        mat[0, 0] = math.cos(math.pi * deg / 180)
        mat[1, 0] = math.sin(math.pi * deg / 180)
        mat[0, 1] = -mat[1, 0]
        mat[1, 1] = mat[0, 0]

        self.mat_affine = np.dot(mat, self.mat_affine)

    def rotate_at(self, deg:float, cx:float, cy:float):
        self.translate(-cx, -cy)
        self.rotate(deg)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if (canvas_width * image_height) > (image_width * canvas_height):
            scale = canvas_height / image_height
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            scale = canvas_width / image_width
            offsety = (canvas_height - image_height * scale) / 2

        self.scale(scale)
        self.translate(offsetx, offsety)

    def to_image_point(self, x, y):
        if self.pil_image == None:
            return None
        mat_inv = np.linalg.inv(self.mat_affine)
        image_point = np.dot(mat_inv, (x, y, 1.))
        # if  image_point[0] < 0 or image_point[1] < 0 or image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
        #     return None

        return image_point[:2]

    def from_image_point(self, x, y):
        canvas_point = np.dot(self.mat_affine, (x, y, 1.))
        # if  image_point[0] < 0 or image_point[1] < 0 or image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
        #     return None

        return canvas_point[:2]

    def draw_image(self, pil_image):
        self.canvas.delete('all')

        if pil_image == None:
            return
        
        self.pil_image = self.orig_pil_image
        #self.pil_image = self.orig_pil_image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        mat_inv = np.linalg.inv(self.mat_affine)

        affine_inv = (
            mat_inv[0, 0], mat_inv[0, 1], mat_inv[0, 2],
            mat_inv[1, 0], mat_inv[1, 1], mat_inv[1, 2]
            )

        dst = self.pil_image.transform(
                    (canvas_width, canvas_height),
                    Image.AFFINE,
                    affine_inv,
                    Image.NEAREST
                    )

        im = ImageTk.PhotoImage(image=dst)

        item = self.canvas.create_image(
                0, 0,
                anchor='nw',
                image=im
                )

        self.image = im

    def redraw_image(self):
        if self.pil_image == None:
            return
        self.draw_image(self.pil_image)


if __name__ == "__main__":


    import time
    root = tk.Tk()
    app = SimpleViewer(master=root)

    root.mainloop()