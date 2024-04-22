import torch
import numpy as np
import matplotlib.pyplot as plt

from tasks.utils.dct import bdct, ibdct
from tasks.utils.io import draw_image_batch, rescale_tensor


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / 3.1415


angles = []
for i in range(1, 512):
    a, b = np.arange(i) % 2, np.ones(i) / 2
    angles.append(angle_between(a, b))
print(angles)
plt.plot(angles)
plt.show()


def is_orthogonal(matrix):
    # 计算矩阵的转置
    transpose = torch.transpose(matrix, 0, 1)
    # 计算矩阵乘积
    product = torch.matmul(matrix, transpose)
    # 计算单位矩阵
    identity = torch.eye(matrix.size(0), dtype=matrix.dtype)
    # 判断乘积矩阵是否接近单位矩阵
    is_orthogonal = torch.allclose(product, identity)
    return is_orthogonal

# images = draw_image_batch(dtype='tensor')
# dct_images = bdct(images)
# a = dct_images[0, 60, :2, :5]
#
# # mix up
# img_0, img_1 = dct_images[0], dct_images[1]
# chs = [0]
# dct_images[0, chs], dct_images[1, chs] = img_1[chs], img_0[chs]
#
# recovered_images = ibdct(dct_images)
# re_dct_images = bdct(recovered_images)
# b = re_dct_images[0, 60, :2, :5]
# print(a, b)

# recovered_images = rescale_tensor(recovered_images)
# plt.imshow(recovered_images[0])
# plt.show()
