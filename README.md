# Final Project
### Description
In this project, we use motion vector of each macroblock to sperate foreground and background.
The first application is composing panorama using processed background images.
The second application is producing a interesting video by apply forground images on panorama.
The third application is to eliminate one of forgrounds from second application.

### Progress
1. Read Video [Done]
2. Get Motion Vector [Done]
3. Get Foreground and Background [middle piont 1]
4. Stick Background to Parorama [middle piont 2]
5. Application Outputs 1:  Panorama Video
6. Application Outputs 2:  Panorama Video with specified path
7. Application Outputs 3:  Panorama Video by removing one of foreground object



### Code structure

    main
    ├── ioVideo                 # handle read video and play video
    ├── motionVector            # derive motion vector for each macroblock
    ├── fgextract               # seperate foreground and background for each frame
    ├── cache                   # store precessed video and motion vector np array here to accelerate process
    ├── result                  # parorama image and output video
    ├── video                   # sample video to be processed
    └── README.md

### Uasage Example
    java foregroundSplit.java "C:\\video_rgb\\Stairs_490_270_346"
    python opencv/MP4toRGB.py -f video/Stairs.mp4
    python opencv/MP4toRGB.py -d "C:\\video_rgb\\video2_240_424_383"
    
### Member
- [Chia-Hao Chang](https://www.linkedin.com/in/chia-hao-chang/)
- [Chih-Ken Yao](https://www.linkedin.com/in/chih-ken-yao/)
- [Sung-Fu Han](linkedin.com/in/sungfuhan/)
