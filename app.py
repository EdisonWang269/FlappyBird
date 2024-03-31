import time
import cv2
import numpy as np
import random

face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('./model/haarcascade_smile.xml')

bird_x = 100
bird_y = 300
bird_velocity = 0

# 鳥的下墜速度
gravity = 1  

# 鳥的跳躍高度
jump_strength = -8

# 控制障礙物生成的時間間隔
obstacle_generation_interval = 1

def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        for (sx, sy, sw, sh) in smiles:
            return True

    return False

class Obstacle:
    def __init__(self):
        self.width = 40
        self.height = random.randint(50, 200)
        self.x = 800 
        self.speed = 10 
        self.gap = 80 
        self.generate_position()

    def generate_position(self):
        self.upper_y = random.randint(50, 200)
        self.lower_y = self.upper_y + self.gap + self.height

    def move(self):
        self.x -= self.speed
        if self.x < -self.width:
            self.x = 800
            self.generate_position()

cap = cv2.VideoCapture(0)

obstacle = Obstacle()

score = 0

last_obstacle_generation_time = time.time()

game_over = False  
game_over_time = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if not game_over:
      
        bird_width = 30
        bird_height = 30
       
        cv2.rectangle(frame, (bird_x, bird_y), (bird_x + bird_width, bird_y + bird_height), (255, 0, 0), -1)

       
        bird_velocity += gravity
        bird_y += bird_velocity

        bird_y = max(0, min(bird_y, frame.shape[0] - bird_height))

        obstacle.move()
        obstacle_width = obstacle.width 
        upper_obstacle_height = obstacle.upper_y 
        lower_obstacle_height = obstacle.lower_y 
        obstacle_x = obstacle.x  
        obstacle_color = (0, 255, 0)  

        cv2.rectangle(frame, (obstacle_x, 0), (obstacle_x + obstacle_width, upper_obstacle_height), obstacle_color, -1)
        cv2.rectangle(frame, (obstacle_x, obstacle.lower_y), (obstacle_x + obstacle_width, frame.shape[0]), obstacle_color, -1)

        if bird_x + bird_width > obstacle_x and bird_x < obstacle_x + obstacle_width:
            if bird_y < upper_obstacle_height or bird_y + bird_height > lower_obstacle_height:
                game_over = True
                game_over_time = time.time()

        if obstacle_x < -obstacle_width:
            obstacle = Obstacle()

        smile_detected = detect_smile(frame)

        if smile_detected:
            bird_velocity = jump_strength

        if bird_x > obstacle_x + obstacle_width:
            score += 1 // 10
    else:
        if game_over_time is not None and time.time() - game_over_time >= 2:
            break
        game_over_text = "GAME OVER"
        text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(frame, game_over_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow('Game', frame)

    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()