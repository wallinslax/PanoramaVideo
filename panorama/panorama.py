# TODO: extract code

import logging
import cv2
import argparse
import stitcher


# TODO: update function name
def main(key_frame_num):
    curr_img = './out/key_frames/frame0.jpg'

    # TODO: handle left right part
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
        next_img = './out/key_frames/frame{}.jpg'.format(i)
        logging.info('Stitching frame{} and frame{}...'.format(i - 1, i))
        curr_img = stitcher.stitch_image(curr_img, next_img)
        cv2.imwrite('./out/temp/temp' + str(i) + '.jpg', curr_img)

    result = curr_img

    cv2.imwrite('./out/panorama.jpg', result)
    logging.info("panorama.jpg complete!")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # TODO: update default value
    parser.add_argument('num', nargs='?', default=9, type=int,
                        help='the number of key frames (default: 2)')
    args = parser.parse_args()

    main(args.num)