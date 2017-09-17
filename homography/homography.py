import numpy as np
from math import floor


def compute_homography(x, x_new):
    '''
        Computes the homography matrix between two planes given two arrays
        of points that correspond between them in both planes.

        x: Array of points on the original plane.
        x_new: Array of points that x map to on the destiantion plane.
    '''
    input_mat = []
    output_vec = []
    # Organize the input points into the matrix and the vector necessary to
    # solve the system
    for i in range(0, 4):
        input_mat.append([
            x[i][0], x[i][1], 1, 0, 0, 0, -(x[i][0] * x_new[i][0]),
            -(x[i][1] * x_new[i][0])])
        input_mat.append([
            0, 0, 0, x[i][0], x[i][1], 1, -(x[i][0] * x_new[i][1]),
            -(x[i][1] * x_new[i][1])])
        output_vec.append(x_new[i][0])
        output_vec.append(x_new[i][1])
    # Create numpy arrays from the organized data
    np_input_mat = np.array(input_mat)
    np_output_vec = np.array(output_vec)
    # Obtain values for h11 through h32, append a 1 at the end
    h_vec = np.append(np.linalg.solve(np_input_mat, np_output_vec), [1])
    # Reshape into 3x3 matrix and return
    return h_vec.reshape(3, 3)


def transplane_point(point, homography):
    '''
        Transplanes a point given the an homography matrix.

        point: array of the form [x, y, 1].
        homography: 3x3 numpy array.
    '''
    # Transplane point: x' = xH
    preliminary = homography.dot(point)
    # Divide by the last component to ensure the form [x', y', 1]
    return preliminary / preliminary[2]


def rounded_approximation(i, j, from_image):
    '''
        Returns the pixel on the position round(i), round(j), from the provided
        image, if the pixels are negative or exceed the boundaries of the image
        it returns None.

        i: float column
        j: float row
        from_image: image to take the pixels from
    '''
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
    '''
        Returns a 3 position array containing the interpolated_approximation
        of the colors for float coordinates i and j using the 4 surrounding
        pixels, if either i, or j, or their respective surrounding pixels are
        negative or exceed the boundaries of the image, it returns None.

        i: float column
        j: float row
        from_image: image to take the pixels from
    '''
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


def transplane_image(from_image, to_image, homography, interpolate=False):
    '''
        Transplanes one image to the plane of the other given the right
        homography, it does so by the rounding method, unless the interpolation
        parameter is True, returns the transplaned image.

        from_image: source image to transplane.
        to_image: image to transplane source onto.
        homography: 3x3 numpy array.
        interpolate: Boolean to determine the use of bi-linear interpolation.
    '''
    to_height, to_width, to_channels = to_image.shape
    for i in range(0, to_width):
        for j in range(0, to_height):
            # Transplane each point of the to_image
            trans_point = transplane_point(np.array([i, j, 1]), homography)
            # If specified, use bi-linear interpolation
            if interpolate:
                new_point = interpolated_approximation(
                    trans_point[0], trans_point[1], from_image)
            else:
                new_point = rounded_approximation(
                    trans_point[0], trans_point[1], from_image)
            # if a match was found update the pixel of to_image
            if new_point is not None:
                to_image[j][i] = new_point
    return to_image
