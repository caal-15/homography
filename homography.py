import numpy as np
import cv2
import os
from math import floor


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


def rounded_approximation(i, j, from_image):
    from_height, from_width, from_channels = from_image.shape
    new_i = int(round(i))
    new_j = int(round(j))
    cond_1 = new_i < from_width and new_i > 0
    cond_2 = new_j < from_height and new_j > 0
    if cond_1 and cond_2:
        return from_image[new_j][new_i]
    else:
        return None


def interpolated_approximation(i, j, from_image):
    from_height, from_width, from_channels = from_image.shape
    if i.is_integer() and j.is_integer():
        return from_image[int(j)][int(i)]
    else:
        i_1 = floor(i)
        j_1 = floor(j)
        i_2 = i_1 + 1
        j_2 = j_1 + 1
        cond_1 = 0 <= i_1 < from_width and 0 <= i_2 < from_width
        cond_2 = 0 <= j_1 < from_height and 0 <= j_2 < from_height
        if cond_1 and cond_2:
            new_pixel = []
            for c in range(0, 3):
                term_1 = 1 / ((i_2 - i_1) * (j_2 - j_1))
                vec_1 = np.array([i_2 - i, i - i_1])
                mat = np.array([
                    [from_image[j_1][i_1][c], from_image[j_2][i_1][c]],
                    [from_image[j_2][i_1][c], from_image[j_2][i_2][c]]
                ])
                vec_2 = [[j_2 - j], [j - j_1]]
                new_pixel.append(int(term_1 * vec_1.dot(mat).dot(vec_2)[0]))
            return new_pixel
        else:
            return None


def transplane_image(from_image, to_image, homography, outpath,
                     interpolate=False):
    to_height, to_width, to_channels = to_image.shape
    from_height, from_width, from_channels = from_image.shape
    for i in range(0, to_width):
        for j in range(0, to_height):
            trans_point = transplane_point(np.array([i, j, 1]), homography)
            if interpolate:
                new_point = interpolated_approximation(
                    trans_point[0], trans_point[1], from_image)
            else:
                new_point = rounded_approximation(
                    trans_point[0], trans_point[1], from_image)
            if new_point is not None:
                to_image[j][i] = new_point
    cv2.imwrite(outpath, to_image)


x = [(514, 252), (714, 252), (514, 402), (714, 402)]
x_new = [(514, 252), (739, 229), (523, 458), (747, 485)]

H = compute_homography(x, x_new)
from_image = cv2.imread(os.path.dirname(__file__).join('picture.jpg'))
to_image = np.zeros((from_image.shape[0], from_image.shape[1], 3), np.uint8)

transplane_image(from_image, to_image, H, 'transplaned.png')
transplane_image(from_image, to_image, H, 'transplaned_interpol.png', True)
