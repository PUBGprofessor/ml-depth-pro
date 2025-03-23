import torch
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt

depth_map = torch.load('output/depth.pth')

# 归一化深度图
depth_map = depth_map.numpy()
# depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
# depth_map = 1 - depth_map
# depth_map = cv2.resize(depth_map, (img.width, img.height))
print(depth_map.shape)
# np.save("depth_map.npy", depth_map)
# 显示深度图
plt.imshow(depth_map, cmap="magma")
plt.axis("off")
plt.show()
