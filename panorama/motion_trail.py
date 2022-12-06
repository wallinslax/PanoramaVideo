import argparse
import logging
import cv2
import motion_trail_stitcher

FRAME_SPACE = 50
PANORAMA_PATH = './out/panorama/panorama_SAL.jpg'
FG_BASE_PATH = './out/foregrounds/SAL_fg1_'


# TODO: rename function name
def main(foreground_num):
    # motion_trail = PANORAMA_PATH
    #
    # for i in range(0, foreground_num):
    #     # TODO: use constant
    #     fg_img = './out/foregrounds/fg{}.jpg'.format(i)
    #     logging.info('Stitching fg{} to panorama...'.format(i))
    #     motion_trail = motion_trail_stitcher.stitch_fg_bg(fg_img, motion_trail)
    #     # TODO: use constant
    #     cv2.imwrite('./out/temp/motion_temp' + str(i) + '.jpg', motion_trail)
    #
    # result = motion_trail

    foreground_paths = []
    for i in range(0, foreground_num):
        if i % FRAME_SPACE == 0:
            foreground_paths.append(FG_BASE_PATH + str(i) + '.jpg')
    result = motion_trail_stitcher.generate_motion_trails(PANORAMA_PATH, foreground_paths)

    # TODO: use constant
    cv2.imwrite('./out/motion_trail.jpg', result)
    logging.info("motion_trail.jpg complete!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=437, type=int,
                        help='the number of foreground frames (default: 437)')
    args = parser.parse_args()

    main(args.num)
