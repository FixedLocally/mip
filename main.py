import simplex
import numpy as np

M = 1e9


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000, suppress=True)
    print(simplex.mip([
        [2, 4, 5, 0, 0, 0, -1, 100],   # 2x1 + 4x2 + 5x3 <= 100
        [1, 1, 1, 0, 0, 0, -1, 30],    # x1 + x2 + x3 <= 30
        [10, 5, 2, 0, 0, 0, -1, 204],  # 10x1 + 5x2 + 2x3 <= 204
        [1, 0, 0, -M, 0, 0, -1, 0],    # y1=(x1>0)
        [0, 1, 0, 0, -M, 0, -1, 0],    # y2=(x2>0)
        [0, 0, 1, 0, 0, -M, -1, 0],    # y3=(x3>0)
    ], ['I', 'I', 'I', 'B', 'B', 'B'], [52, 30, 20, -500, -400, -300]))
