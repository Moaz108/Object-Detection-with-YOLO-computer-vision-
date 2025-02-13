import sys
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import *

import cv2
import cvzone
import pandas as pd
from PIL import Image, ImageTk
from ultralytics import YOLO
import win32api
import win32event
from winerror import ERROR_ALREADY_EXISTS

mutex = win32event.CreateMutex(None, False, 'name')
if win32api.GetLastError() == ERROR_ALREADY_EXISTS:
    sys.exit(0)

cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
video_paused = False
model = YOLO('yolov8s.pt')


def read_classes(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


def start_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)
        is_camera_on = True
        video_paused = False
        update_canvas()


def stop_webcam():
    global cap, is_camera_on
    if cap:
        cap.release()
        is_camera_on = False


def pause_resume_video():
    global video_paused
    video_paused = not video_paused


def select_file():
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()


def update_canvas():
    global frame_count
    if is_camera_on and not video_paused:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % frame_skip_threshold == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1020, 500))
                selected_class = class_selection.get()
                results = model.predict(frame)
                for row in pd.DataFrame(results[0].boxes.data).astype("float").itertuples():
                    x1, y1, x2, y2, _, d = map(int, row[1:])
                    if selected_class == "All" or class_list[d] == selected_class:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f'{class_list[d]}', (x1, y1), 1, 1)
                photo = ImageTk.PhotoImage(Image.fromarray(frame))
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.img = photo
    canvas.after(10, update_canvas)


def quit_app():
    stop_webcam()
    root.quit()
    root.destroy()


root = tk.Tk()
root.title("YOLO v8 My App")
canvas = tk.Canvas(root, width=1020, height=500)
canvas.pack(fill='both', expand=True)

class_list = read_classes('coco.txt')
class_selection = tk.StringVar(value="All")

frame = tk.Frame(root)
frame.pack(fill='x')

tk.Label(root, text="Select Class:").pack(side='left')
OptionMenu(root, class_selection, "All", *class_list).pack(side='left')

for text, command in zip(["Play", "Stop", "Select File", "Pause/Resume", "Quit"],
                         [start_webcam, stop_webcam, select_file, pause_resume_video, quit_app]):
    tk.Button(frame, text=text, command=command).pack(side='left')

initial_photo = ImageTk.PhotoImage(Image.open('yolo.jpg'))
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)
canvas.img = initial_photo

root.mainloop()
