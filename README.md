# video2Panorama
### Description
In this project, we use motion vector of each macroblock to sperate foreground and background. 
The first application is composing panorama with foreground motion trail. 
The second application is producing a panorama video by apply forground on panorama. 
The third application is to create original video without foregound.

### Result
1. Original Video

https://user-images.githubusercontent.com/20626329/207740243-4c6ee809-a124-4ce6-8c4b-d7cc1218e937.mp4

2. Prcessed Foreground [middle output 1]

[![extracted_foreground](https://img.youtube.com/vi/E1ky-vNfdNk/0.jpg)](https://www.youtube.com/watch?v=E1ky-vNfdNk)

3. Prcessed Backgound [middle output 2]

[![extracted_background](https://img.youtube.com/vi/y3mnn0xp6GE/0.jpg)](https://www.youtube.com/watch?v=y3mnn0xp6GE)

4. Panorama without foregound

![panorama_sal_full](https://user-images.githubusercontent.com/20626329/207943440-bdeb8b32-f919-496e-815b-1801fef21a47.jpg)

5. Application Outputs 1:  Panorama with foreground motion trail

![3  motion_trail_SAL_06](https://user-images.githubusercontent.com/20626329/207740950-bd4d3ac5-32c7-42f5-9d20-5fc53eddadab.jpg)

6. Application Outputs 2:  Panorama Video

[![Panorama Video](https://img.youtube.com/vi/JSElgxdzX44/0.jpg)](https://www.youtube.com/watch?v=JSElgxdzX44)

7. Application Outputs 3:  Panorama Video by removing one of foreground object

[![Panorama Video without foreground](https://img.youtube.com/vi/GoLkilTI9Zg/0.jpg)](https://www.youtube.com/watch?v=GoLkilTI9Zg)


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
    git clone https://github.com/wallinslax/PanoramaVideo.git
    pip install -r requirements.txt

### Generate the result above
    python main.py -f video/SAL.mp4
    
### Member
- [Chia-Hao Chang](https://www.linkedin.com/in/chia-hao-chang/)
- [Chih-Ken Yao](https://www.linkedin.com/in/chih-ken-yao/)
- [Sung-Fu Han](linkedin.com/in/sungfuhan/)
