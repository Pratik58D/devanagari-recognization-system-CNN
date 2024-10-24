import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import pandas as pd

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Ask user for the alphabet or gesture to capture
label = input("Enter the label for the gesture: ").strip().upper()

# Set maximum number of data samples to capture
max_samples = 200
samples_count = 0  # Counter to track the number of samples saved

# Create the "images" directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def capture_hand_data():
    global samples_count  # Track samples across function calls

    while True:
        success, img = cap.read()  # Step 1: Capture video frame
        hands, img = detector.findHands(img)  # Step 2: Detect hand landmarks

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Step 3: Crop the hand region with some offset (ROI)
            imgCrop = img[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]

            # Step 4: Resize cropped image while maintaining aspect ratio
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:  # Height greater than width
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = (300 - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  # Width greater than height
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = (300 - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Step 6: Convert to grayscale (after Step 4 resize)
            imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)

            # Step 5: Normalize the grayscale image (values between 0 and 1)
            imgGrayNormalized = imgGray / 255.0

            # Display the cropped, resized, and processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgGrayNormalized)

            key = cv2.waitKey(1)
            if key == ord('s'):  # Save data if 's' is pressed
                existing_samples = count_existing_samples(label)

                if existing_samples < max_samples:  # Check against max_samples
                    samples_count += 1
                    img_filename = f"images/img_{time.time()}.jpg"
                    cv2.imwrite(img_filename, imgGrayNormalized * 255)  # Save as uint8 after rescaling

                    # Collect hand landmarks
                    hand_data = []
                    for landmark in hand['lmList']:
                        hand_data.extend(landmark)  # Append x, y coordinates of each landmark
                    hand_data.append(label)  # Add the label to the data

                    # Save data to CSV
                    save_to_csv(hand_data)
                    print(f"Data saved for label '{label}' ({samples_count}/{max_samples}).")
                else:
                    print(f"Maximum samples ({max_samples}) reached for label '{label}'. No more data will be saved.")

            elif key == ord('q'):  # If 'q' is pressed, quit the application
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

# Save data to CSV
def save_to_csv(data):
    df = pd.DataFrame([data])
    df.to_csv('hand_data.csv', mode='a', header=False, index=False)

# Count existing samples for the label
def count_existing_samples(label):
    try:
        df = pd.read_csv('hand_data.csv', header=None)
        return (df.iloc[:, -1] == label).sum()  # Assuming the label is in the last column
    except FileNotFoundError:
        return 0  # Return 0 if the file doesn't exist

# Run the capture function
capture_hand_data()
