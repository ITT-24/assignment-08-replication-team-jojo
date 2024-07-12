from tkinter import Tk, Canvas
from threading import Thread
import win32gui
import win32con
import mediapipe as mp
import cv2
import time
import numpy as np
import pyautogui
import math
import sys

screen_width, screen_height = pyautogui.size()

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

START_DRAW_THRESHOLD = 40
END_DRAW_THRESHOLD = 80
TIME_INTERVAL = 0.05

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

class LetterDrawer():
    def __init__(self):
        self.cap = cv2.VideoCapture(video_id) 
        self.gesture_being_drawn = False
        self.line_points = []
        self.running = True
        self.dimensions = (screen_height, screen_width, 3) #set size to screen size to avoid mapping
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result)

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        out = np.zeros(self.dimensions)
        self.annotated_image = self.draw_landmarks_on_image(out, result)


    def run(self):
        with HandLandmarker.create_from_options(self.options) as landmarker:
            while self.running:
                _, frame = self.cap.read()

                frame = cv2.flip(frame, 1)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_result = landmarker.detect_async(mp_image, int(time.time() * 1000))

                time.sleep(TIME_INTERVAL)

            self.cap.release()

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def check_for_gesture(self, thumb_landmark, index_landmark, hand_landmark):
        x_position_thumb = thumb_landmark.x * screen_width
        y_position_thumb = thumb_landmark.y * screen_height
        x_position_index = index_landmark.x * screen_width
        y_position_index = index_landmark.y * screen_height
        x_position_hand = hand_landmark.x * screen_width
        y_position_hand = hand_landmark.y * screen_height
        distance = self.get_distance(x_position_thumb, y_position_thumb, x_position_index, y_position_index)
        if(not self.gesture_being_drawn and distance <= START_DRAW_THRESHOLD):
            (x_position_hand, y_position_hand)
            self.gesture_being_drawn = True
            print("Gesture started")
        elif self.gesture_being_drawn and distance > END_DRAW_THRESHOLD:
            canvas.delete('all')
            self.line_points = []
            self.gesture_being_drawn = False

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        global gesture_being_drawn, line_points
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        if len(hand_landmarks_list) == 0:
            canvas.delete('all')
            line_points = []

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            thumb_landmark = hand_landmarks[4] #https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index?hl=de#models
            index_landmark = hand_landmarks[8]
            hand_landmark = hand_landmarks[9]

            self.check_for_gesture(thumb_landmark, index_landmark, hand_landmark)
            self.try_add_line_point(hand_landmark)
            self.draw_line()
        return annotated_image

    def try_add_line_point(self, hand_landmark):
        if self.gesture_being_drawn:
            x_position_hand = hand_landmark.x * screen_width
            y_position_hand = hand_landmark.y * screen_height
            self.line_points.append((x_position_hand, y_position_hand))

    def draw_line(self):
        if len(self.line_points) > 1:
            for i in range(1, len(self.line_points)):
                x1, y1 = self.line_points[i - 1]
                x2, y2 = self.line_points[i]
                canvas.create_line(x1, y1, x2, y2, fill='black', width=5)

def make_window_click_through(hwnd):
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 5, win32con.LWA_ALPHA)

# Create a full-screen window
root = Tk()
root.attributes('-fullscreen', True)  # Full-screen mode
root.attributes('-topmost', True)  # Always on top

# Create a canvas for drawing
canvas = Canvas(root, bg='white', highlightthickness=0)
canvas.pack(fill='both', expand=True)

# Make the window click-/see-through
root.update_idletasks()
hwnd = win32gui.FindWindow(None, root.title())
make_window_click_through(hwnd)

detector = LetterDrawer()
detector_thread = Thread(target=detector.run)
detector_thread.start()

root.mainloop()