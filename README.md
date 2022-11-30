# Final Project
### Description
In this project, we use motion vector of each macroblock to sperate foreground and background.
The first application is composing panorama using processed background images.
The second application is producing a interesting video by apply forground images on panorama.
The third application is to eliminate one of forgrounds from second application.

### Progress
1. Read Video [Done]
2. Get Motion Vector [Done]
3. Get Foreground and Background [middle piont 1] [Done]

https://user-images.githubusercontent.com/20626329/204932300-3b7bc119-16ff-4071-a4ac-b2dc8e15bf2a.mp4

https://user-images.githubusercontent.com/20626329/204932308-2471b5c1-b16e-43ab-8e91-ef1d7492bf17.mp4

4. Stick Background to Parorama [middle piont 2] [Done]
![panorama_SAL](https://user-images.githubusercontent.com/20626329/204932333-54b48527-be3c-4f9f-ab39-950fce8facf3.jpg)

5. Application Outputs 1:  Panorama with foreground motion trail
![App1_SAL](https://user-images.githubusercontent.com/20626329/204932593-a217a62b-43bd-488f-a97d-543b7fa296c1.jpg)

6. Application Outputs 2:  Panorama Video with specified path


7. Application Outputs 3:  Panorama Video by removing one of foreground object

### Code Structure

    main
    ├── ioVideo                 # handle read video and play video
    ├── motionVector            # derive motion vector for each macroblock
    ├── fgextract               # seperate foreground and background for each frame
    ├── cache                   # store precessed video and motion vector np array here to accelerate process
    ├── result                  # parorama image and output video
    ├── video                   # sample video to be processed
    └── README.md

### Setup
    git clone https://github.com/wallinslax/576Fall22FinalProject.git
    pip install -r requirements.txt

### Uasage Example
    python main.py -f video/Stairs.mp4
    python main.py -d "C:\\video_rgb\\video2_240_424_383"
    
### Member
- [Chia-Hao Chang](https://www.linkedin.com/in/chia-hao-chang/)
- [Chih-Ken Yao](https://www.linkedin.com/in/chih-ken-yao/)
- [Sung-Fu Han](linkedin.com/in/sungfuhan/)
