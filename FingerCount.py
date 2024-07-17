import cv2
import mediapipe as mp
import time
import os
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        if img is None:
            return img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = [0] * 5  # Initialize a list to track fingers up
        if len(self.lmList) == 0:
            return fingers

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  # Comparing x values for thumb
            fingers[0] = 1

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  # Comparing y values for other fingers
                fingers[id] = 1

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) == 0:
            return 0, img, [0, 0, 0, 0, 0, 0]
        
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  # Change to 0 for built-in camera
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Initialize hand detector
    detector = handDetector(detectionCon=0.75)

    # Define the path to the directory containing overlay images
    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    if not myList:
        print(f"No images found in '{folderPath}'. Please add images and rerun the script.")
        return

    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        image = cv2.resize(image, (200, 200))  # Resize the images to fit the video frame
        overlayList.append(image)
        print(f'Loaded and resized image: {folderPath}/{imPath}, Shape: {image.shape}')
    print(f'Total overlay images loaded: {len(overlayList)}')

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            totalFingers = sum(fingers)
            print(f'Number of fingers detected: {totalFingers}')

            # Overlay the appropriate image based on the number of fingers
            if totalFingers <= 5:
                overlayImg = overlayList[totalFingers - 1]  # 1 to 5 fingers images
            else:
                overlayImg = overlayList[5]  # Default to closed fist image for more than 5 fingers

            h, w, c = overlayImg.shape
            img[0:h, 0:w] = overlayImg

            # Draw a rectangle and text for finger count
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)
        else:
            print("No hand detected or not enough landmarks found.")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
