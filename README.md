# Face Recognition Project

## Requirements
Before running the code, you need to install the required Python libraries. 
Open your terminal or command prompt and run:

pip install opencv-python numpy

*Note: You also need to download the 'haarcascade_frontalface_alt.xml' file from the OpenCV GitHub repository and save it in your project folder.*

---

## Step 1: Collect Face Data
1. Open the file named `Faces_detection.py`.
2. **Crucial:** Update the `dataset_path` variable to the folder where you want to save the images.
3. Update the `xml path` to where your `haarcascade_frontalface_alt.xml` file is located.
4. Run the code. It will open your webcam.
5. Enter your name when prompted in the console.
6. The camera will begin taking snapshots of your face.
   * *Note: You must allow your code editor to access the camera.*
   * *Tip: Move your head slightly to capture different angles. The more pictures you take, the more accurate the model will be.*
7. Press **'q'** to save the data and exit.

## Step 2: Recognize Faces
1. Open the second file (`face_Detection_2.py`).
2. Update the `dataset_path` and `xml path` in this file as well (make sure they match the paths in Step 1).
3. Run the code.
4. The webcam will open.
5. The system will detect your face and draw a **Green box** around it with your name displayed.

## How it works
* **Face Detection:** The code uses `haarcascade.xml` to identify facial features (eyes, nose, mouth) and isolate the face from the background.
* **Recognition:** It compares the live video feed against the data saved in Step 1 to predict who you are.
