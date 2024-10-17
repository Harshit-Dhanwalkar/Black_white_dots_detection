import cv2
import numpy as np

# Set the minimum and maximum size for the black dots
s1 = 10
s2 = 20

# List to keep track of dot positions for tracing
positions = []

# Open the video capture (0 for the default camera, or use a video file path)
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Read the current frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to binary (detect black dots)
    th, threshed = cv2.threshold(
        gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Find contours of the dots
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Filter the contours by area (to find small black dots)
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)

    # Draw contours and trace path
    for cnt in xcnts:
        # Get the center of the dot (centroid)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Add the current position to the list of tracked positions
            positions.append((cX, cY))

            # Draw the dot's current position on the frame
            cv2.circle(frame, (cX, cY), 3, (0, 0, 255), -1)

    # Draw the tracing lines connecting the previous dot positions
    for i in range(1, len(positions)):
        cv2.line(frame, positions[i - 1], positions[i], (0, 255, 0), 2)

    # Display the resulting frame with dots and tracing path
    cv2.imshow("Tracking Black Dots", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
