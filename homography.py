import numpy as np
import cv2
import os


def compute_homography(x, x_new):
    input_mat = []
    output_vec = []
    for i in range(0, 4):
        input_mat.append([
            x[i][0], x[i][1], 1, 0, 0, 0, -(x[i][0] * x_new[i][0]),
            -(x[i][1] * x_new[i][0])])
        input_mat.append([
            0, 0, 0, x[i][0], x[i][1], 1, -(x[i][0] * x_new[i][1]),
            -(x[i][1] * x_new[i][1])])
        output_vec.append(x_new[i][0])
        output_vec.append(x_new[i][1])
    np_input_mat = np.array(input_mat)
    np_output_vec = np.array(output_vec)
    h_vec = np.append(np.linalg.solve(np_input_mat, np_output_vec), [1])
    return h_vec.reshape(3, 3)


def transplane_point(point, homography):
    preliminary = homography.dot(point)
    return preliminary / preliminary[2]


def transplane_image(from_image, to_image, homography, outpath):
    to_height, to_width, to_channels = to_image.shape
    from_height, from_width, from_channels = from_image.shape
    for i in range(0, to_width):
        for j in range(0, to_height):
            trans_point = transplane_point(np.array([i, j, 1]), homography)
            new_i = int(round(trans_point[0]))
            new_j = int(round(trans_point[1]))
            cond_1 = new_i < from_width and new_i > 0
            cond_2 = new_j < from_height and new_j > 0
            if cond_1 and cond_2:
                to_image[j][i] = from_image[new_j][new_i]
    cv2.imwrite(outpath, to_image)


x = [(514, 252), (714, 252), (514, 402), (714, 402)]
x_new = [(514, 252), (739, 229), (523, 458), (747, 485)]

H = compute_homography(x, x_new)
from_image = cv2.imread(os.path.dirname(__file__).join('picture.jpg'))
to_image = np.zeros((from_image.shape[0], from_image.shape[1], 3), np.uint8)

transplane_image(from_image, to_image, H, 'transplaned.png')
