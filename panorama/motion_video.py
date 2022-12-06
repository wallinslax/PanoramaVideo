import argparse
import logging
import cv2
import motion_trail_stitcher

PANORAMA_PATH = './out/panorama/panorama_SAL.jpg'
FG_BASE_PATH = './out/foregrounds/SAL_fg1_{}.jpg'


# TODO: rename function name
def main(foreground_num):
    for i in range(0, foreground_num):
        motion_trail = PANORAMA_PATH
        fg_img = FG_BASE_PATH.format(i)
        logging.info('Stitching fg{} to panorama...'.format(i))
        motion_trail = motion_trail_stitcher.stitch_fg_bg(fg_img, motion_trail)
        # TODO: use constant
        cv2.imwrite('./out/motion_video/frame' + str(i) + '.jpg', motion_trail)

    logging.info("motion video frames complete!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=437, type=int,
                        help='the number of foreground frames (default: 437)')
    args = parser.parse_args()

    main(args.num)
