import argparse
import logging
import cv2
import numpy as np

VIDEO_BASE_PATH = './out/motion_video_frames/frame{}.jpg'
FILE_PATH = './out/motion_video/motion_video.mp4'


# TODO: rename function name
def main(video_frame_num):
    frames = []
    for i in range(0, video_frame_num):
        logging.info('Reading frame{} to buffer...'.format(i))
        frames.append(cv2.imread(VIDEO_BASE_PATH.format(i)))

    save_video(frames)
    logging.info("motion video saved!")


def save_video(frames):
    height, width, _ = np.shape(frames[0])
    # TODO: use constant
    out = cv2.VideoWriter(FILE_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, frameSize=(width, height))
    for frame in frames:
        out.write(frame)
    out.release()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=437, type=int,
                        help='the number of video frames (default: 437)')
    args = parser.parse_args()

    main(args.num)
