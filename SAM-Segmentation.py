import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File Selection: Please set VideoCapture value to Video1 or Video2
vidcap = cv2.VideoCapture("Video1.mp4")

# Initialize an empty list to store video frames
frames_array = []

while vidcap.isOpened():
    # Read the next frame from the video
    return_value, frame = vidcap.read()

    # Check if the frame was successfully read
    if return_value:
        frames_array.append(frame)
    else:
        break

# Convert the list of frames into a NumPy array
frames_array = np.array(frames_array)
vidcap.release()

# Close all windows opened by OpenCV
cv2.destroyAllWindows()

print("Reading Part Completed!")

# Initialize an empty NumPy array to store bounding boxes
bounding_boxes = np.zeros((len(frames_array), 4))

# Create an object tracker using KCF algorithm
object_tracker = cv2.TrackerKCF_create()

# Select the initial bounding box for the object of interest
current_bounding_box = cv2.selectROI('Frames', frames_array[0], False)

# Update the bounding boxes array with the initial bounding box
bounding_boxes[0][:4] = current_bounding_box[:4]

# Adjust the bounding box coordinates
bounding_boxes[0][2] += current_bounding_box[0]
bounding_boxes[0][3] += current_bounding_box[1]

# Initialize the object tracker with the initial bounding box and the first frame
object_tracker.init(frames_array[0], current_bounding_box)

# Iterate through all frames and track the object using the KCF tracker
for i in range(1, len(frames_array)):
    tracked, current_bounding_box = object_tracker.update(frames_array[i])
    if tracked:
        bounding_boxes[i][:4] = current_bounding_box[:4]
        bounding_boxes[i][2] += current_bounding_box[0]
        bounding_boxes[i][3] += current_bounding_box[1]
    else:
        print("Unable to locate the object in the current frame.")

print("Bounding boxes successfully obtained for all frames.")

# Load the SAM segmentation model from the checkpoint file
sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device=device)

# Create a SAM predictor object using the loaded model
segmentation_predictor = SamPredictor(sam_model)

# Convert the bounding boxes array to a Torch tensor
tracked_object_boxes = torch.tensor(bounding_boxes, device=device)

# Initialize an empty list to store frames with contours
frames_with_contours = []

for i in range(0, len(frames_array), 5):  # Selecting 1 frame out of every 5 frames
    segmentation_predictor.set_image(frames_array[i])
    transformed_object_boxes = segmentation_predictor.transform.apply_boxes_torch(tracked_object_boxes[i], frames_array[i].shape[:2])
    masks, _, _ = segmentation_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_object_boxes, multimask_output=False)

    contours, _ = cv2.findContours(masks[0][0].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_contours = frames_array[i].copy()
    cv2.drawContours(frame_with_contours, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=5)
    frames_with_contours.append(frame_with_contours)

# This final part will generate a video with the object's contour drawn in each frame.
height, width, _ = frames_with_contours[0].shape
output_frames_per_second = 10
segmented_video = cv2.VideoWriter('Segmented_Video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), output_frames_per_second, (width, height))

# Write each frame with contours to the video file
for image in frames_with_contours:
    segmented_video.write(image)

# Release the video writer object
segmented_video.release()
print("Operation Successfully Completed!")
