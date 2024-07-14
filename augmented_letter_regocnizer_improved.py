from threading import Thread
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow
from PyQt6.QtGui import QPainter, QPen, QFont
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal
import mediapipe as mp
import cv2
import time
import numpy as np
import pyautogui
import math
import sys
import json
from dollar_recognizer import DollarRecognizer, Point  # Import DollarRecognizer and Point
screen_width, screen_height = pyautogui.size()

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

START_DRAW_THRESHOLD = 40
END_DRAW_THRESHOLD = 120
TIME_INTERVAL = 0.01
TOLERANCE = 50
COMMANDS = json.load(open('commands.json')) 

video_id = 1
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])


class HandDetector(QThread):
    finished = pyqtSignal()
    clear_scene_signal = pyqtSignal()
    add_line_point_signal = pyqtSignal(object)  # Assuming hand_landmark is an object
    draw_line_signal = pyqtSignal()
    set_gesture_being_drawn_signal = pyqtSignal(bool)
    gesture_recognized_signal = pyqtSignal(str, float)  # signal for gesture recognition
    def __init__(self):
        super().__init__()
        self.options = HandLandmarkerOptions(
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.handle_result)
        self.cap = cv2.VideoCapture(video_id) 
        self.dimensions = (screen_height, screen_width, 3) #set size to screen size to avoid mapping
        self.gesture_being_drawn = False
        self.time_since_last_detection = np.inf

    def run(self):
        with HandLandmarker.create_from_options(self.options) as landmarker:
            while True:
                _, frame = self.cap.read()

                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_result = landmarker.detect_async(mp_image, int(time.time() * 1000))
                time.sleep(TIME_INTERVAL)

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def check_for_gesture(self, thumb_landmark, index_landmark, hand_landmark):
    # Initialize the counter if it doesn't exist
        if not hasattr(self, 'over_threshold_counter'):
            self.over_threshold_counter = 0

        x_position_thumb = thumb_landmark.x * screen_width
        y_position_thumb = (1 - thumb_landmark.y) * screen_height  # Flip y-coordinate
        x_position_index = index_landmark.x * screen_width
        y_position_index = (1 - index_landmark.y) * screen_height  # Flip y-coordinate
        distance = self.get_distance(x_position_thumb, y_position_thumb, x_position_index, y_position_index)
        if not self.gesture_being_drawn and distance <= START_DRAW_THRESHOLD:
            print("Gesture started")
            self.gesture_being_drawn = True
            self.over_threshold_counter = 0  # Reset counter when gesture starts
        elif self.gesture_being_drawn:
            if distance > END_DRAW_THRESHOLD:
                self.over_threshold_counter += 1  # Increment counter if over threshold
                if self.over_threshold_counter >= 5:  # Check if counter has reached 5
                    print("Gesture ended")
                    self.gesture_being_drawn = False
                    self.over_threshold_counter = 0  # Reset counter after gesture ends
                    self.clear_scene_signal.emit()
            else:
                self.over_threshold_counter = 0  # Reset counter if distance is not over threshold

        if self.gesture_being_drawn:
            self.add_line_point_signal.emit(hand_landmark)
            self.draw_line_signal.emit()

    def handle_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        hand_landmarks_list = result.hand_landmarks

        if time.time() - self.time_since_last_detection > 1:
            self.clear_scene_signal.emit()

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            self.time_since_last_detection = time.time()
            hand_landmarks = hand_landmarks_list[idx]
            
            thumb_landmark = hand_landmarks[4] #https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index?hl=de#models
            index_landmark = hand_landmarks[8]
            hand_landmark = hand_landmarks[9]

            self.check_for_gesture(thumb_landmark, index_landmark, hand_landmark)
                
class LetterDrawer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.line_points = []
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.time_since_last_point = np.inf
        self.menu_opened = False
        self.dimensions = (screen_height, screen_width, 3) #set size to screen size to avoid mapping
        self.startLongRunning()

         # Init Dollar Recognizer
        self.recognizer = DollarRecognizer()
        self.recognizer.load_unistrokes_from_xml()

    def initUI(self):
        self.setWindowTitle('Drawing Application')
        self.setGeometry(0, 0, screen_width, screen_height)
        
        # Make the window transparent and always on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Create a scene and a view for drawing
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, screen_width, screen_height)
        
        # Set the view to be transparent
        self.view.setStyleSheet("background: transparent")
        self.view.setFrameStyle(0)
        
        # Initialize drawing state
        self.lastPoint = None
        self.pen = QPen(Qt.GlobalColor.black, 5, Qt.PenStyle.SolidLine)

    def startLongRunning(self):
        # Setup the thread and worker
        self.hand_detector = HandDetector()
        self.hand_detector.clear_scene_signal.connect(self.clear_scene)
        self.hand_detector.add_line_point_signal.connect(self.try_add_line_point)
        self.hand_detector.draw_line_signal.connect(self.draw_line)
        self.hand_detector.finished.connect(self.hand_detector.deleteLater)
        self.hand_detector.start()

    def clear_scene(self):
        self.scene.clear()
        self.view.update()
        self.line_points = []

    def try_add_line_point(self, hand_landmark):
        x_position_hand = hand_landmark.x * screen_width
        y_position_hand = hand_landmark.y * screen_height
        # Check if the hand is moving
        # If the hand is not moving with a small tolerance and a timer, open the command menu
        if (abs(self.last_mouse_x - x_position_hand) < TOLERANCE and abs(self.last_mouse_y - y_position_hand) < TOLERANCE):
            if time.time() - self.time_since_last_point > 0.5:
                if not self.menu_opened:
                    recognized_result = self.recognizer.Recognize([Point(p[0], p[1]) for p in self.line_points])
                    self.on_gesture_recognized(recognized_result.Name, recognized_result.Score)
        # If the hand is moving, add the new point to the line
        else:
            self.menu_opened = False
            self.last_mouse_x = x_position_hand
            self.last_mouse_y = y_position_hand
            self.line_points.append((x_position_hand, y_position_hand))
            self.time_since_last_point = time.time()

    def draw_line(self):
        if len(self.line_points) > 1:
            for i in range(1, len(self.line_points)):
                x1, y1 = self.line_points[i - 1]
                x2, y2 = self.line_points[i]
                self.scene.addLine(x1, y1, x2, y2, self.pen)
                self.view.update()

    def on_gesture_recognized(self, gesture_name, score):
        print(f"Gesture recognized: {gesture_name} with score: {score}")
        self.open_command_menu(gesture_name, self.last_mouse_x, self.last_mouse_y)


    def open_command_menu(self, detected_character, x, y):
        print(detected_character)
        #print(COMMANDS["commands"][detected_character])
        #commands = COMMANDS["commands"][detected_character]
        self.menu_opened = True
        #for i in range(len(commands)): 
        #    self.scene.addText(commands[i], QFont('Helvetica', 12)).setPos(x, y + i * 20)

    def close_command_menu(self):
        self.menu_opened = False
        self.clear_scene()
        self.draw_line()

    def execute_command(self, command):
        print("Command executed:", command)
        # Add code to execute the command here


app = QApplication(sys.argv)
detector = LetterDrawer()

detector.show()
sys.exit(app.exec())