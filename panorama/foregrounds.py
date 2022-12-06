import argparse
import logging
import cv2
import stitcher


# TODO: rename function name
def main(filepath):
    video = cv2.VideoCapture(filepath)
    curr_frame_index = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = stitcher.crop_black(frame)
            # TODO: use constant
            cv2.imwrite('./out/foregrounds/fg' + str(curr_frame_index) + '.jpg', frame)
            curr_frame_index += 1
        else:
            break
    video.release()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('filepath', nargs='?', default='./data/video/SAL_fg1s.mp4',
                        help='path of the video file (default: ./data/video/SAL_fg1s.mp4)')
    args = parser.parse_args()

    main(args.filepath)
