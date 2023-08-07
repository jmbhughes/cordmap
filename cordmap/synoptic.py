import numpy as np
import cv2
from skimage.draw import disk
import matplotlib.pyplot as plt


def align_images(img_test, img_ref, show_plots=False, use_mask=False, 
                 num_orb_points=10_000):
    # based on https://thinkinfi.com/image-alignment-and-registration-with-opencv/
    # TODO: remove hard coded numbers
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_test_grey = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    if use_mask:
        center = (img_test_grey.shape[1] // 2, img_test_grey.shape[0] // 2)
        rr, cc = disk(center, 1000)
        img_test_grey[cc, rr] = 255

    height, width = img_ref_grey.shape

    detector = cv2.ORB_create(num_orb_points)
    keypoints_test, descriptors_test = detector.detectAndCompute(img_test_grey, None)
    keypoints_ref, descriptors_ref = detector.detectAndCompute(img_ref_grey, None)

    if show_plots:
        visual_keypoints = cv2.drawKeypoints(img_ref, keypoints_test, 0, (0, 222, 0))
        visual_keypoints = cv2.resize(visual_keypoints, (width // 2, height // 2))
        fig, ax = plt.subplots()
        ax.imshow(visual_keypoints)

    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(descriptors_test, descriptors_ref))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # # Display only 100 best matches {good[:100}
    if show_plots:
        img_match = cv2.drawMatches(img_test, keypoints_test, img_ref, keypoints_ref, matches[:100], None, flags=2)
        img_match = cv2.resize(img_match, (width // 3, height // 3))
        fig, ax = plt.subplots()
        ax.imshow(img_match)

    # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keypoints_test[matches[i].queryIdx].pt
        p2[i, :] = keypoints_ref[matches[i].trainIdx].pt

    # only keep matches that are within a 50 pixel change
    good_matches = np.sqrt(np.sum(np.square(p1 - p2), axis=1)) < 50
    matches = np.array(matches)[good_matches]
    no_of_matches = len(matches)

    # # Display only 100 best matches {good[:100}
    if show_plots:
        img_match = cv2.drawMatches(img_test, keypoints_test, img_ref, keypoints_ref, matches[:100], None, flags=2)
        img_match = cv2.resize(img_match, (width // 3, height // 3))
        fig, ax = plt.subplots()
        ax.imshow(img_match)

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv2.warpPerspective(img_test, homography, (width, height))

    return homography, aligned_img
