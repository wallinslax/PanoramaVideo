import argparse
import logging
import cv2
import motion_trail_stitcher

PANORAMA_PATH = './out/panorama/test3_panorama.jpg'


# TODO: rename function name
def main(foreground_num):
    motion_trail = PANORAMA_PATH

    for i in range(0, foreground_num):
        # TODO: use constant
        fg_img = './out/foregrounds/fg{}.jpg'.format(i)
        logging.info('Stitching fg{} to panorama...'.format(i))
        motion_trail = motion_trail_stitcher.stitch_fg_bg(fg_img, motion_trail)
        # TODO: use constant
        cv2.imwrite('./out/temp/motion_temp' + str(i) + '.jpg', motion_trail)

    result = motion_trail

    # TODO: use constant
    cv2.imwrite('./out/motion_trail.jpg', result)
    logging.info("motion_trail.jpg complete!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=2, type=int,
                        help='the number of foreground frames (default: 2)')
    args = parser.parse_args()

    main(args.num)
