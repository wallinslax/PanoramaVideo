import argparse
import logging
import cv2

# TODO: update FRAME_SPACE
FRAME_SPACE = 10


# TODO: rename function name
def main(filepath):
    video = cv2.VideoCapture(filepath)
    curr_frame_index, key_frame_idx = 0, 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if curr_frame_index % FRAME_SPACE == 0:
                cv2.imwrite('./out/key_frames/frame' + str(key_frame_idx) + '.jpg', frame)
                key_frame_idx += 1
            curr_frame_index += 1
        else:
            break
    video.release()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('filepath', nargs='?', default='./data/video/test3.mp4',
                        help='path of the video file (default: ./data/video/test3.mp4)')
    args = parser.parse_args()

    main(args.filepath)
