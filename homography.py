import numpy as np


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


x = [(514, 252), (714, 252), (514, 402), (714, 402)]
x_new = [(514, 252), (739, 229), (523, 458), (747, 845)]

print(compute_homography(x, x_new))
