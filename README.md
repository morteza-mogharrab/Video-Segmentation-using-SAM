**README.md**

### Introduction
This Python application utilizes the SAM (Segment Anything) model for image and video segmentation. The code allows for tracking an object in a video and generating a segmented video with the object's contour drawn in each frame.

### Prerequisites
Ensure you have the following installed:
- Python (version 3.8 or higher)
- PyTorch (version 1.7 or higher)
- TorchVision (version 0.8)
- OpenCV contrib-python (version 4.8.1.78)
- OpenCV (version 4.8.1.78)

### SAM Model
The application utilizes the `sam_vit_h_4b8939.pth` model for segmentation. This model file is approximately 2.4 GB in size. Download the model file from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it in the same directory as the Python code.

### Usage
1. **Set Up Environment**: Ensure all required libraries and the SAM model file are correctly installed and placed in the appropriate directory.
2. **Place Videos**: Put the videos you want to process in the same directory as the Python code.
3. **Execution**:
   - Run the Python script.
   - Upon execution, the initial frame of the video will be displayed.
   - Use the mouse to draw a bounding box around the target object for tracking.
   - Press Enter to initiate the video generation process.
4. **Output**: A segmented video will be generated with the object's contour drawn in each frame.

### Miscellaneous Tips
1. **Frame Rate Adjustment**: By default, the code selects one frame out of every five frames to process due to hardware limitations. You can increase the frame rate if CUDA (GPU) is available on other computer machines.
2. **Object Trackers**: If there are issues with the KCF tracker, alternative trackers like `cv2.TrackerMOSSE_create()` and `cv2.TrackerCSRT_create()` are available for use.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- This project utilizes the SAM model, developed by [Facebook AI](https://ai.facebook.com/).
- Thanks to the contributors of PyTorch, TorchVision, and OpenCV for their valuable libraries.
- Special thanks to the authors of relevant papers and resources used in the development of this application.
