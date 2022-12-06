import logging
import cv2
import argparse
import stitcher


# TODO: rename function name
def main_reverse(key_frame_num):
    # TODO: use constant
    curr_img = './out/key_frames/frame' + str(key_frame_num - 1) + '.jpg'

    for i in range(key_frame_num - 2, -1, -1):
        # TODO: use constant
        next_img = './out/key_frames/frame{}.jpg'.format(i)
        logging.info('Stitching frame{} and frame{}...'.format(i + 1, i))
        curr_img = stitcher.stitch_image(curr_img, next_img)
        # TODO: use constant
        cv2.imwrite('./out/temp/temp' + str(key_frame_num - i - 1) + '.jpg', curr_img)

    result = curr_img

    # TODO: use constant
    cv2.imwrite('./out/panorama.jpg', result)
    logging.info("panorama.jpg complete!")


# TODO: rename function name
def main(key_frame_num):
    # TODO: use constant
    curr_img = './out/key_frames/frame0.jpg'

    # TODO: handle left right parts
    # mid_frame = key_frame_num // 2
    # left_panorama = None
    # for i in range(1, key_frame_num):
    #     if i == mid_frame:
    #         left_panorama = curr_img
    #         # TODO: update path
    #         curr_img = './out/key_frames/frame{}.jpg'.format(i)
    #         continue
    #     # TODO: update path
    #     next_img = './out/key_frames/frame{}.jpg'.format(i)
    #     logging.info('Stitching frame{} and frame{}...'.format(i - 1, i))
    #     curr_img = stitcher.stitch_image(curr_img, next_img)
    #     cv2.imwrite('./out/temp/temp' + str(i) + '.jpg', curr_img)
    #
    # if key_frame_num > 1:
    #     right_panorama = curr_img
    #     result = stitcher.stitch_image(left_panorama, right_panorama)
    #     logging.info("Join left and right parts.")
    # else:
    #     result = curr_img

    for i in range(1, key_frame_num):
        # TODO: use constant
        next_img = './out/key_frames/frame{}.jpg'.format(i)
        logging.info('Stitching frame{} and frame{}...'.format(i - 1, i))
        curr_img = stitcher.stitch_image(curr_img, next_img)
        # TODO: use constant
        cv2.imwrite('./out/temp/temp' + str(i) + '.jpg', curr_img)

    result = curr_img

    # TODO: use constant
    cv2.imwrite('./out/panorama.jpg', result)
    logging.info("panorama.jpg complete!")


def generate_panorama(frames):
    # TODO: use constant
    if len(frames) < 2:
        logging.warning('Length of frames must be greater than 2.')
        exit()

    curr_img = None

    for i, frame in enumerate(frames):
        if i == 0:
            curr_img = frame
        else:
            next_img = frame
            logging.info('Stitching frame{} and frame{}...'.format(i - 1, i))
            curr_img = stitcher.stitch_image(curr_img, next_img)
            # TODO: use constant
            cv2.imwrite('./out/temp/temp' + str(i) + '.jpg', curr_img)

    result = curr_img

    # TODO: use constant
    cv2.imwrite('./out/panorama/panorama.jpg', result)
    logging.info("panorama.jpg complete!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=41, type=int,
                        help='the number of key frames (default: 41)')
    args = parser.parse_args()

    main(args.num)
