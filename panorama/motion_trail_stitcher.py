import logging
import cv2
import numpy as np
import stitcher


logging.getLogger().setLevel(logging.INFO)


def generate_motion_trails(panorama_path, foreground_paths):
    logging.info('Reading foreground images...')
    fg_imgs = get_fg_imgs(foreground_paths)
    bg_img = cv2.imread(panorama_path)

    logging.info('Calculating homograpies...')
    hs = compute_homographies(fg_imgs, bg_img)
    stitched_image = stitch_imgs(fg_imgs, bg_img, hs)

    return stitched_image


def get_fg_imgs(foreground_paths):
    fg_imgs = []
    for foreground_path in foreground_paths:
        fg_img = cv2.imread(foreground_path)
        # TODO: remove crop_black
        fg_img = stitcher.crop_black(stitcher.cylindrical_project(stitcher.crop_black(fg_img)))
        fg_imgs.append(fg_img)
    return fg_imgs


def compute_homographies(fg_imgs, bg_img):
    hs = []
    for fg_img in fg_imgs:
        hs.append(compute_homography(fg_img, bg_img))
    return hs


def stitch_imgs(fg_imgs, bg_img, hs):
    img_out = bg_img
    for i, fg_img in enumerate(fg_imgs):
        logging.info('Stitching fg{} to panorama...'.format(i))
        img_out = stitch(fg_img, img_out, hs[i])

    return img_out


def stitch_fg_bg(fg_img_path, bg_img_path):
    if isinstance(fg_img_path, str):
        fg_img = cv2.imread(fg_img_path)
    else:
        fg_img = fg_img_path

    fg_img = stitcher.crop_black(stitcher.cylindrical_project(fg_img))

    if isinstance(bg_img_path, str):
        bg_img = cv2.imread(bg_img_path)
    else:
        bg_img = bg_img_path

    h = compute_homography(fg_img, bg_img)
    stitched_image = stitch(fg_img, bg_img, h)

    return stitched_image


# TODO: extract function
def stitch(fg_img, bg_img, h):
    h1, w1 = fg_img.shape[0:2]
    h2, w2 = bg_img.shape[0:2]

    fg_img_corners = np.float32([[0, 0], [0, h1],
                                 [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    fg_img_corners = cv2.perspectiveTransform(fg_img_corners, h)

    [x_min, y_min] = np.int32(fg_img_corners.min(axis=0, initial=None).ravel() - 0.5)
    [x_max, y_max] = np.int32(fg_img_corners.max(axis=0, initial=None).ravel() + 0.5)

    img_out = cv2.warpPerspective(fg_img, h, (w2, h2))

    img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    # TODO: use constant
    img_out_mask = cv2.threshold(img_out_gray, 5, 255, cv2.THRESH_BINARY)[1]

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if x < bg_img.shape[1] and y < bg_img.shape[0] and np.any(img_out_mask[y][x]):
                bg_img[y][x] = img_out[y][x]

    return bg_img


# # TODO: extract function
# def stitch(fg_img, bg_img, h):
#     h1, w1 = fg_img.shape[0:2]
#     h2, w2 = bg_img.shape[0:2]
#
#     fg_img_corners = np.float32([[0, 0], [0, h1],
#                                  [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     fg_img_corners = cv2.perspectiveTransform(fg_img_corners, h)
#     bg_img_corners = np.float32([[0, 0], [0, h2],
#                                  [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
#     output_corners = np.concatenate((fg_img_corners, bg_img_corners), axis=0)
#
#     [x_min, y_min] = np.int32(output_corners.min(axis=0, initial=None).ravel() - 0.5)
#     [x_max, y_max] = np.int32(output_corners.max(axis=0, initial=None).ravel() + 0.5)
#
#     img_out = cv2.warpPerspective(fg_img, h, (x_max - x_min, y_max - y_min))
#
#     for y in range(y_min, y_max):
#         for x in range(x_min, x_max):
#             if not np.any(img_out[y][x]) and x < bg_img.shape[1] and y < bg_img.shape[0]:
#                 img_out[y][x] = bg_img[y][x]
#
#     return img_out


# TODO: change function name
# TODO: extract function
def compute_homography(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # TODO: update cv2.NORM_L2
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    # TODO: use constant
    matches = matcher.knnMatch(des1, des2, k=2)

    # TODO: update match_ratio
    # TODO: use constant
    match_ratio = 0.4
    good_matches = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance:
            good_matches.append(m)

    # TODO: update min_match_count
    # TODO: use constant
    min_match_count = 4
    if len(good_matches) > min_match_count:
        img1_pts = []
        img2_pts = []
        for match in good_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)

        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # TODO: use constant
        h, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        for i in range(mask.shape[0] - 1, -1, -1):
            if mask[i] == 0:
                np.delete(img1_pts, [i], axis=0)
                np.delete(img2_pts, [i], axis=0)

        # TODO: use constant
        h, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        return h
    else:
        logging.warning("Not enough matches are found.")
        exit()
