import logging
import cv2
import numpy as np

# TODO: update F value
F = 550.2562584220408 * 3
# TODO: update resize
RESIZE_W = 1920
RESIZE_H = 1080


def stitch_image(img1_path, img2_path):
    if isinstance(img1_path, str):
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (RESIZE_W, RESIZE_H))
        img1 = crop_black(cylindrical_project(img1))
    else:
        img1 = img1_path

    if isinstance(img2_path, str):
        img2 = cv2.imread(img2_path)
    else:
        img2 = img2_path
    img2 = cv2.resize(img2, (RESIZE_W, RESIZE_H))
    img2 = crop_black(cylindrical_project(img2))

    h = compute_homography(img1, img2)
    stitched_image = stitch(img1, img2, h)
    stitched_image = crop_black(stitched_image)

    return stitched_image


def cylindrical_project(img):
    cylindrical_img = np.zeros_like(img)
    height, width, depth = cylindrical_img.shape

    center_x = width / 2
    center_y = height / 2

    for i in range(width):
        for j in range(height):
            theta = (i - center_x) / F
            point_x = int(F * np.tan((i - center_x) / F) + center_x)
            point_y = int((j - center_y) / np.cos(theta) + center_y)

            for k in range(depth):
                if 0 <= point_x < width and 0 <= point_y < height:
                    cylindrical_img[j, i, k] = img[point_y, point_x, k]
                else:
                    cylindrical_img[j, i, k] = 0

    return cylindrical_img


def crop_black(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: use constant
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        delta_height = h - y
        delta_width = w - x

        area = delta_height * delta_width

        if area > max_area and delta_height > 0 and delta_width > 0:
            max_area = area
            best_rect = (x, y, w, h)

    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
                       best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        img_crop = img

    return img_crop


# TODO: change function name
# TODO: extract function
def compute_homography(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    img1_cropped = img1[:, w1 - w2:]
    diff = np.size(img1, axis=1) - np.size(img1_cropped, axis=1)

    kp1, des1 = sift.detectAndCompute(img1_cropped, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # TODO: update cv2.NORM_L2
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    # TODO: use constant
    matches = matcher.knnMatch(des1, des2, k=2)

    # TODO: update match_ratio
    # TODO: use constant
    match_ratio = 0.6
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
        img1_pts[:, :, 0] += diff
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


# TODO: extract function
def stitch(img1, img2, h):
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    img1_corners = np.float32([[0, 0], [0, h1],
                               [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img1_corners = cv2.perspectiveTransform(img1_corners, h)
    img2_corners = np.float32([[0, 0], [0, h2],
                               [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    output_corners = np.concatenate((img1_corners, img2_corners), axis=0)

    [x_min, y_min] = np.int32(output_corners.min(axis=0, initial=None).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_corners.max(axis=0, initial=None).ravel() + 0.5)
    t = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]])

    img_out = cv2.warpPerspective(img1, t.dot(h), (x_max - x_min, y_max - y_min))

    for y in range(-y_min, h2 - y_min):
        for x in range(-x_min, w2 - x_min):
            # TODO: add foreground black pixels filling logic
            if np.any(img2[y + y_min][x + x_min]):
                img_out[y][x] = img2[y + y_min][x + x_min]

    return img_out
