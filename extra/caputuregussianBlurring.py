import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import pandas as pd
import os

# Create a folder for images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Ask user for the alphabet or gesture to capture
label = input("Enter the label for the gesture: ").strip().upper()

# Set maximum number of data samples to capture
max_samples = 200
samples_count = 0  # Counter to track the number of samples saved

# Create a function to capture hand landmarks
def capture_hand_data():
    global samples_count  # Track samples across function calls

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop the hand region with some offset
            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

            # Normalize and apply Gaussian Blurring
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            
            # Resize cropped image while maintaining aspect ratio
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Convert to grayscale and apply Gaussian Blurring
            imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

            # Display the processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgBlurred)  # Show the blurred grayscale image

            key = cv2.waitKey(1)
            if key == ord('s'):  # Save data if 's' is pressed
                # Check the number of samples already saved for the label
                existing_samples = count_existing_samples(label)

                if existing_samples < max_samples:  # Check against max_samples
                    samples_count += 1

                    # Save the blurred image as a JPEG in the images folder
                    img_filename = f"gussianimages/img_{time.time()}.jpg"
                    cv2.imwrite(img_filename, imgBlurred)  # Save the preprocessed image

                    # Collect hand landmarks
                    hand_data = []
                    for landmark in hand['lmList']:  # Assuming lmList contains landmarks
                        hand_data.extend(landmark)  # Append x, y coordinates of each landmark
                    hand_data.append(label)  # Add the label to the data

                    # Save data to CSV without the image filename
                    save_to_csv(hand_data)
                    print(f"Data saved for label '{label}' ({samples_count}/{max_samples}).")
                else:
                    print(f"Maximum samples ({max_samples}) reached for label '{label}'. No more data will be saved.")

            elif key == ord('q'):  # If 'q' is pressed, quit the application
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to save data into a CSV file
def save_to_csv(data):
    df = pd.DataFrame([data])
    # Save with the label in the last column
    df.to_csv('hand_data.csv', mode='a', header=False, index=False)

# Function to count existing samples for a specific label
def count_existing_samples(label):
    try:
        df = pd.read_csv('hand_data.csv', header=None)
        # Count the number of times the label appears in the last column
        return (df.iloc[:, -1] == label).sum()  # Assuming the label is the last column
    except FileNotFoundError:
        return 0  # Return 0 if the file doesn't exist

# Run the capture function
capture_hand_data()
