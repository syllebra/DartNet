import tkinter as tk
import tkinter.font as TkFont
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import numpy as np
import os
import json
import cv2

from simple_viewer import SimpleViewer
from annotate import get_dart_scores, seg_intersect

colors = {"cal1":"gold","cal2":"hotpink","cal3":"lightgreen","cal4":"salmon", "test":"white", "dart0":"cyan", "dart1":"cyan", "dart2":"cyan"}



class Application(SimpleViewer):
    def __init__(self, master=None, img = None, menu = True, show_straighten = True):
        super().__init__(master, img, menu)

        self.font = TkFont.Font(family='Helvetica', size=36, weight='bold')
        self.straighten_viewer = None

        if(show_straighten):
            second_win = tk.Toplevel(master)
            self.straighten_viewer = SimpleViewer(second_win, menu=False)
        else:
            self.straighten_viewer = None
        self.my_title = "Python Image Viewer"
        self.create_menu()
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.metadata = {"kc": {"cal1":[0,0],"cal2":[0,0],"cal3":[0,0],"cal4":[0,0]},
                        "board":{
                "r_board": 0.2255,  # radius of full board
                "r_double": 0.170,  # center bull to outside double wire edge, in m (BDO standard)
                "r_treble": 0.1064,  # center bull to outside treble wire edge, in m (BDO standard)
                "r_outer_bull": 0.0174,
                "r_inner_bull": 0.007,
                "w_double_treble": 0.01,  # wire apex to apex for double and treble
                "width": 0,
                "height": 0
            }
        }
        self.test_pt = [0,0]
       
        self.kp_size=30

        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label="Add dart marker",
                                    command=self.add_dart_marker)

        self.master.bind("<Button-3>", self.popup) # Button-2 on Aqua
        self.popup_pos = [0,0]

    def popup(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
            self.popup_pos = [event.x, event.y]
        finally:
            self.popup_menu.grab_release()

    def add_dart_marker(self):
        i = 1
        while f"dart{i}" in self.metadata["kc"]:
            i+=1
        p = self.to_image_point(x=self.popup_pos[0],y=self.popup_pos[1])
        self.metadata["kc"][f"dart{i}" ]=p
        self.redraw_image()

    def menu_save_clicked(self, event=None):
        metadata_filepath = os.path.splitext(self.filename)[0]+".json"
        print(self.metadata)

        to_write = self.metadata.copy()
        to_write["kc"] = {k:v for k,v in self.metadata["kc"].items() if k in ["cal1","cal2","cal3","cal4"]}

        with open(metadata_filepath,"w") as f:
            json.dump(self.metadata,f)

    def menu_save_straighten_clicked(self,event=None):
        straighten = self.straighten_image(self.orig_pil_image)
        xy = np.array([p for k,p in self.metadata["kc"].items()])
        xy_dst = self.transform_points(xy, self.M)

        for i,k in enumerate(self.metadata["kc"].keys()):
            self.metadata["kc"][k] = list(xy_dst[i])
        self.metadata["board"]["width"] = straighten.width
        self.metadata["board"]["height"] = straighten.height
        straighten.save(self.filename)
        self.menu_save_clicked(event)

    def create_menu(self):
        super().create_menu()
 
        self.file_menu.add_command(label="Save", command = self.menu_save_clicked, accelerator="Ctrl+S")
        self.file_menu.add_command(label="Save Straighten Image", command = self.menu_save_straighten_clicked, accelerator="Ctrl+E")

        self.menu_bar.bind_all("<Control-s>", self.menu_save_clicked)
        self.menu_bar.bind_all("<Control-e>", self.menu_save_straighten_clicked)
        self.menu_bar.bind_all("<Control-r>", self.reload_metadata)
 
    def create_widget(self):
        super().create_widget()
        self.master.bind("<ButtonRelease-1>", self.drag_stop)

    def drag_stop(self, event):
        """End drag of an object"""
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        self.redraw_image()

    def drag(self, event):
        """Handle dragging of an object"""
        # compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]

        picked = self._drag_data["item"]
        picked_pi= self.metadata["kc"][picked]
        pc = self.from_image_point(picked_pi[0],picked_pi[1])
        
        pc[0] += delta_x
        pc[1] += delta_y
        pi = self.to_image_point(pc[0],pc[1])

        self.metadata["kc"][picked] = list(pi)

        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.redraw_image()

    def reload_metadata(self, event=None):
        metadata_filepath = os.path.splitext(self.filename)[0]+".json"
        if(os.path.exists(metadata_filepath)):
            with open(metadata_filepath,"r") as file:
                self.metadata = json.load(file)
        else:
            mx = self.orig_pil_image.width *0.5
            my = self.orig_pil_image.height *0.5
            
            self.metadata = {"kc": {"cal1":[mx,self.orig_pil_image.height*0.1],"cal2":[mx,self.orig_pil_image.height*0.9],
                                    "cal3":[self.orig_pil_image.width*0.1,my],"cal4":[self.orig_pil_image.width*0.9,my]}}
            print(json.dumps(self.metadata, indent=4))


        if("board_file" in  self.metadata):
            with open(self.metadata["board_file"],"r") as file:
                md = json.load(file)
                self.metadata["board"] = md["board"]
                
        if("board" not in self.metadata):
            self.metadata["board"] = {
                "r_board": 0.2255,  # radius of full board
                "r_double": 0.170,  # center bull to outside double wire edge, in m (BDO standard)
                "r_treble": 0.1074,  # center bull to outside treble wire edge, in m (BDO standard)
                "r_outer_bull": 0.0159,
                "r_inner_bull": 0.00635,
                "w_double_treble": 0.01,  # wire apex to apex for double and treble
            }

        self.metadata["board"]["width"] = self.orig_pil_image.width
        self.metadata["board"]["height"] = self.orig_pil_image.height

        self.metadata["kc"]["test"] = [0,0]

        self.redraw_image()

    def set_image(self, filename):
        super().set_image(filename)
        if not filename:
            return
        
        self.reload_metadata()

    def find_keypoint(self, event):
        md = self.kp_size //2
        for k,p in self.metadata["kc"].items():
            pc = self.from_image_point(p[0],p[1])
            dx = abs(pc[0]-event.x)
            dy = abs(pc[1]-event.y)
            print(dx, dy)
            if(dx < md and dy < md):
                return k
        return None

    def mouse_down_left(self, event):
        super().mouse_down_left(event)
        picked = self.find_keypoint(event)
        print(picked)
        if(picked is not None):
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self._drag_data['item'] = picked
            self.redraw_image()

    def mouse_move_left(self, event):
        #print(self._drag_data['item'])
        if(self._drag_data['item'] is not None):
            self.drag(event)
            self.__old_event = event
            return
        super().mouse_move_left(event)
        
    def compute_transformations(self, compute_inverse, angle=9, sz=2048 , outer_double_to_border_ratio=0.2255/0.17 ):
        xy = np.array([p for k,p in self.metadata["kc"].items()])
        xy = xy[:, :2]
        c = seg_intersect(xy[0],xy[1],xy[2], xy[3])
        cd = [sz*0.5,sz*0.5]
        rd = sz*0.5 / outer_double_to_border_ratio

        src_pts = np.array([xy[0],xy[1],xy[2],xy[3]]).astype(np.float32)
        dst_pts = np.array([
            [cd[0] - rd * np.sin(np.deg2rad(angle)), cd[1] - rd * np.cos(np.deg2rad(angle))],
            [cd[0] + rd * np.sin(np.deg2rad(angle)), cd[1] + rd * np.cos(np.deg2rad(angle))],
            [cd[0] - rd * np.cos(np.deg2rad(angle)), cd[1] + rd * np.sin(np.deg2rad(angle))],
            [cd[0] + rd * np.cos(np.deg2rad(angle)), cd[1] - rd * np.sin(np.deg2rad(angle))]
        ]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Mi = cv2.getPerspectiveTransform(dst_pts,src_pts) if compute_inverse else None
        return M, Mi

    def transform_image(self, img, M, sz = 2048):
        return cv2.warpPerspective(img.copy(), M, (sz, sz))
    
    def transform_points(self, xy, M):
            xyz = np.concatenate((np.array(xy), np.ones((len(xy), 1))), axis=-1).astype(np.float32)
            xyz_dst = np.matmul(M, xyz.T).T
            xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

            return xy_dst
    
    def straighten_image(self, img, draw_circles = False):
        open_cv_image = np.array(img)
        straight = self.transform_image(open_cv_image, self.M)
        return Image.fromarray(straight)

    def update_transforms(self):
        
        dtbr = self.metadata["board"]["r_board"] / self.metadata["board"]["r_double"]
        M, Mi = self.compute_transformations(compute_inverse=True, outer_double_to_border_ratio=dtbr)
        self.M = M
        self.Mi = Mi

    def compute_scores(self):
        scores= []
        xy = np.array([p for k,p in self.metadata["kc"].items()])
        if xy.shape[0] > 4:
            scores = get_dart_scores(self.transform_points(xy,self.M), self.metadata)
        return scores

    def draw_image(self, pil_image):
        self.canvas.delete('all')

        if pil_image == None:
            return
        
        self.update_transforms() # TODO: update only at load and when moving keypoints
                
        if(self.straighten_viewer is not None):
            straighten = self.straighten_image(self.orig_pil_image, draw_circles=self._drag_data["item"] in [None,"test"])
            self.straighten_viewer.set_pil_image(straighten,"Straighten")

        super().draw_image(pil_image)

        def draw_pt(pt, col="white", dragged=False, border=True):
            sz = self.kp_size // 2
            xmin = pt[0]-sz
            xmax = pt[0]+sz
            ymin = pt[1]-sz
            ymax = pt[1]+sz
            self.canvas.create_line(xmin,ymin, xmax, ymax, fill=col)
            self.canvas.create_line(xmax,ymin, xmin, ymax, fill=col)

            if(border):
                self.canvas.create_rectangle(xmin,ymin, xmax, ymax, outline= "yellow" if dragged else col, width=3)

        def draw_poly(poly, col="white"):
            last = poly[-1]
            for p in poly:
                self.canvas.create_line(last[0],last[1], p[0], p[1], fill=col, width=3)
                last = p

        scores = self.compute_scores()

        xy_cal =  np.array([p for k,p in self.metadata["kc"].items( )if k in ["cal1","cal2","cal3","cal4"] ] )

        center = seg_intersect(xy_cal[0],xy_cal[1],xy_cal[2],xy_cal[3])
        draw_pt(self.from_image_point(x=center[0], y=center[1]), border=False)


        def _draw_circle(center, radius_real, color="red"):
            #center_str = self.transform_points(np.array([center]), self.M)
            center_str = np.array([1024,1024])
            xy_cal_str = self.transform_points(xy_cal, self.M)
            radius_str_dbl = np.mean(np.linalg.norm(xy_cal_str[:4] - center_str, axis=-1))

            def _circle(r_real, center_str, segments = None):
                ratio = (r_real/ self.metadata["board"]["r_double"])
                if(segments is None):
                    segments = max(int(ratio * 250),0)
                a = np.arange(0,np.pi*2,np.pi*2/segments)
                pts = np.array([np.cos(a),np.sin(a)]).T
                pts = center_str + pts*radius_str_dbl * ratio
                return pts

            pts = _circle(radius_real, center_str)
            pts = self.transform_points(pts, self.Mi)
            pt = self.from_image_point(pts[0][0],pts[0][1])
            for p in pts[1:]:
                p = self.from_image_point(p[0], p[1])
                self.canvas.create_line(pt[0],pt[1],p[0],p[1], fill=color)
                pt = p

        _draw_circle(center, self.metadata["board"]["r_board"])
        _draw_circle(center, self.metadata["board"]["r_double"])
        _draw_circle(center, self.metadata["board"]["r_double"]-self.metadata["board"]["w_double_treble"])
        _draw_circle(center, self.metadata["board"]["r_treble"])
        _draw_circle(center, self.metadata["board"]["r_treble"]-self.metadata["board"]["w_double_treble"])
        _draw_circle(center, self.metadata["board"]["r_outer_bull"])
        _draw_circle(center, self.metadata["board"]["r_inner_bull"])

        for k, pti in self.metadata["kc"].items():
            pt = self.from_image_point(x=pti[0], y=pti[1])
            pt_col = colors.get(k,"cyan")
            draw_pt(pt, pt_col, self._drag_data["item"] == k)
        
        if("bbox" in self.metadata):
            for k, box in self.metadata["bbox"].items():
                bx = [
                    [box[0][0],box[0][1]],
                    [box[0][0],box[1][1]],
                    [box[1][0],box[1][1]],
                    [box[1][0],box[0][1]]
                ]
                pts = [self.from_image_point(x=pti[0], y=pti[1]) for pti in bx]
                pt_col = colors.get(k,"cyan")
                draw_poly(pts, pt_col)#, self._drag_data["item"] == k)

        if(len(scores)>0):
            for i,k in enumerate(self.metadata["kc"].keys()):
                if(i>3):
                        pt = self.metadata["kc"][k]
                        score = scores[i-4]
                        ptc = self.from_image_point(pt[0], pt[1])
                        pt_col = colors.get(k,"cyan")
                        self.canvas.create_text(ptc[0]+35,ptc[1], text= score, fill = pt_col, font=self.font)
                        #print(scores)



if __name__ == "__main__":
    import time
    root = tk.Tk()
    app = Application(master=root)

    root.mainloop()