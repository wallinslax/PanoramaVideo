import logging
import cv2
import numpy as np
import stitcher


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
    bg_img_corners = np.float32([[0, 0], [0, h2],
                                 [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    output_corners = np.concatenate((fg_img_corners, bg_img_corners), axis=0)

    [x_min, y_min] = np.int32(output_corners.min(axis=0, initial=None).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_corners.max(axis=0, initial=None).ravel() + 0.5)

    img_out = cv2.warpPerspective(fg_img, h, (x_max - x_min, y_max - y_min))

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if not np.any(img_out[y][x]) and x < bg_img.shape[1] and y < bg_img.shape[0]:
                img_out[y][x] = bg_img[y][x]

    return img_out


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
