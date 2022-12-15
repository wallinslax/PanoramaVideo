# video2Panorama
### Description
In this project, we use motion vector of each macroblock to sperate foreground and background.
The first application is composing panorama with foreground motion trail.
The second application is producing a panorama video by apply forground on panorama.
The third applic



ation is to create original video without foregound.

### Result
1. Original Video

https://user-images.githubusercontent.com/20626329/207740243-4c6ee809-a124-4ce6-8c4b-d7cc1218e937.mp4

2. Prcessed Foreground [middle output 1]


3. Prcessed Backgound [middle output 2]

4. Panorama without foregound

![1  SAL_panorama](https://user-images.githubusercontent.com/20626329/207740803-495cced0-8cf7-47df-a1c3-5cb194c54f6a.jpg)


5. Application Outputs 1:  Panorama with foreground motion trail

![3  motion_trail_SAL_06](https://user-images.githubusercontent.com/20626329/207740950-bd4d3ac5-32c7-42f5-9d20-5fc53eddadab.jpg)

6. Application Outputs 2:  Panorama Video



https://user-images.githubusercontent.com/20626329/207741026-7eed6994-5c4e-4e81-aa9d-1c8ec45ea3a6.mp4



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
