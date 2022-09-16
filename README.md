# Superimposing a Virtual Cube on an AR Tag

Project-01 for the course _ENPM673: Perception for Autonomous Robots (Spring 2022)_.

* [Tag Detection Video](https://drive.google.com/file/d/1b3BwREXBDh4NX7IK18lkpWzGDRhTIOKu/view)  
* [Superimposing an Image on the Tag Video](https://drive.google.com/file/d/1t0cBmiBT7ZfrKwVpiUB6KkpTy66exQHG/view)
* [Placing a Virtual Cube onto the Tag Video](https://drive.google.com/file/d/1jZkY2Ptfy8n9z-PoPUe6jMRn7E0lAndq/view)

## AR Tag Detection

<!-- p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190576008-6ef15b41-68dc-401e-ae69-06a554e4d8a9.png" width="600" height="400">
</p -->

### Flowchart
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190575327-45e55da9-9828-4041-b77c-a54f6546acfa.jpeg" width="600" height="300">
</p>

### Edge Detection using FFT
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190576475-5c54a7cb-ee01-4655-bb38-ca0e3c9ddd09.png" width="300" height="200">
</p>

### Extracted Corners and Edges
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190576654-6b7252dd-982e-4d6c-811c-73dabbf571f8.png" width="600" height="300">
</p>


## AR Tag Decoding

### Flowchart
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190577433-c7126ea5-f5e1-4790-927c-83ade84b5d7d.png" width="600" height="300">
</p>

### Implementation
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190578219-7db39015-9531-4a24-bb4d-f58149d7437a.png" width="300" height="250">
</p>

### Decoding Tag ID
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190578343-98258015-c3d8-41a7-8ad6-7b1c5a50d8c7.png" width="600" height="300">
</p>


## AR Tag Tracking

### Flowchart for superimposing an image on Tag
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190579335-f450dd40-b79c-4d3b-87d4-4349a11508b5.png" width="600" height="300">
</p>

### Superimposed Testudo image on Tag
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190579882-10b4e495-af85-4e90-887b-8b263171d3d9.png" width="600" height="300">
</p>

### Flowchart for placing a virtual cube onto Tag
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190580118-4bebbb84-bd39-44a3-bd7c-3648c1707dfd.png" width="600" height="300">
</p>

### Placed virtual cube on Tag
<p align="center">
  <img src="https://user-images.githubusercontent.com/40534801/190580167-ed47fa72-bc56-4e59-8fd1-02986c4ea194.png" width="600" height="300">
</p>


### Media Files \& Codes

    ├── 1tagvideo.mp4                     # Main video file for Q2.
    ├── tag_single_frame.jpg              # Single frame image for Q1(a).
    ├── tag_ref_image.png                 # Reference tag image for Q1(b).
    └── testudo.png                       # Testudo template image for Q2(a).


    ├── ARTag_detection.py                # For detecting the AR tag from tag_single_frame.jpg using FFT.
    ├── ARTag_decode_reference.py         # For decoding the AR tag info from tag_ref_image.png.
    ├── ARTag_decode.py                   # For decoding the AR tag info from a single frame of 1tagvideo.mp4.
    ├── ARTag_testudo_video.py            # For superimposing testudo.png on the AR tag in 1tagvideo.mp4.
    └── ARTag_cube_video.py               # For constructing a virtual cube on the AR tag in 1tagvideo.mp4.

### Dependencies 
* cv2
* numpy
* scipy
* imutils
* matplotlib

### Running the Codes

Ensure all the media files are in the same directory as that of the Python scripts. 

1. For detecting the tag using FFT:

			`python3 ARTag_detection.py`


2. For decoding the reference image:


			`python3 ARTag_decode_reference.py`
 

3.  For decoding a single frame from the video:


			`python3 ARTag_decode.py`


4. For superimposing the Testudo image on the tag:


		 	`python3 ARTag_testudo_video.py`


5. For constructing the cube on the tag:


		 	`python3 ARTag_cube_video.py`
