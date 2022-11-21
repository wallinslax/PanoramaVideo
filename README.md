# Final Project
### Description
In this project, we use motion vector of each macroblock to sperate foreground and background.
The first application is composing panorama using processed background images.
The second application is producing a interesting video by apply forground images on panorama.
The third application is to eliminate one of forgrounds from second application.

### Member
- [Chia-Hao Chang](https://www.linkedin.com/in/chia-hao-chang/)
- [Chih-Ken Yao](https://www.linkedin.com/in/chih-ken-yao/)
- [Sung-Fu Han](linkedin.com/in/sungfuhan/)

### Uasage Example
    java foregroundSplit.java "C:\\video_rgb\\Stairs_490_270_346"
    python opencv/MP4toRGB.py -f video/Stairs.mp4
    python opencv/MP4toRGB.py -d "C:\\video_rgb\\video2_240_424_383"
