import os
import cv2
import numpy as np
from homography import compute_homography, transplane_image


if __name__ == '__main__':
    # Correspondences between original image and
    # image with projectivity removed
    x = [(514, 252), (714, 252), (514, 402), (714, 402)]
    x_new = [(514, 252), (739, 229), (523, 458), (747, 485)]
    # Compute the homography
    H = compute_homography(x, x_new)

    # Load the original image and Create the destination image
    original_image = cv2.imread(os.path.dirname(__file__).join('picture.jpg'))
    corrected_image = np.zeros(
        (original_image.shape[0], original_image.shape[1], 3), np.uint8)

    # Transplane image to remove projectivity and write to file
    # (without interpolation)
    cv2.imwrite('transpalaned.png', transplane_image(
        original_image, corrected_image, H))
    # Transplane image to remove projectivity and write to file
    # (with interpolation)
    cv2.imwrite(
        'transplaned_interpolated.png',
        transplane_image(original_image, corrected_image, H, True))

    # Correspondences between picture of paper, and scanner simulated one
    x = [(0, 0), (773, 0), (0, 1000), (773, 1000)]
    x_new = [(156, 656), (2072, 656), (156, 3176), (2072, 3176)]
    # Compute the homography
    H = compute_homography(x, x_new)

    # Load the paper image and Create the destination image
    paper_image = cv2.imread(os.path.dirname(__file__).join('paper.jpg'))
    scanned_paper = np.zeros((1000, 773, 3), np.uint8)

    # Transplane image to simulate scanner and write to file
    cv2.imwrite(
        'scanned_paper.png',
        transplane_image(paper_image, scanned_paper, H, True))

    # Correspondences between straight image and original image
    x = [(535, 275), (712, 262), (542, 435), (722, 452)]
    x_new = [(0, 0), (546, 0), (0, 489), (546, 489)]
    # Compute the homography
    H = compute_homography(x, x_new)

    # Load staright image
    sign_image = cv2.imread(os.path.dirname(__file__).join('new_sign.png'))

    # Transplane straight image to add projectivity and write to file
    cv2.imwrite(
        'new_sign_transplaned.png',
        transplane_image(sign_image, original_image, H, True))
