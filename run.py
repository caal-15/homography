import os
import cv2
import numpy as np
from homography import compute_homography, transplane_image


if __name__ == '__main__':
    x = [(514, 252), (714, 252), (514, 402), (714, 402)]
    x_new = [(514, 252), (739, 229), (523, 458), (747, 485)]
    H = compute_homography(x, x_new)

    original_image = cv2.imread(os.path.dirname(__file__).join('picture.jpg'))
    corrected_image = np.zeros(
        (original_image.shape[0], original_image.shape[1], 3), np.uint8)

    cv2.imwrite('transpalaned.png', transplane_image(
        original_image, corrected_image, H))
    cv2.imwrite(
        'transplaned_interpoplated.png',
        transplane_image(original_image, corrected_image, H, True))

    x = [(535, 275), (712, 262), (542, 435), (722, 452)]
    x_new = [(0, 0), (546, 0), (0, 489), (546, 489)]
    H = compute_homography(x, x_new)

    sign_image = cv2.imread(os.path.dirname(__file__).join('new_sign.png'))

    cv2.imwrite(
        'new_sign_transplaned.png',
        transplane_image(sign_image, original_image, H, True))
