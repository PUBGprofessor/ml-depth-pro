from PIL import Image
import depth_pro
import torch
print(1)
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()
# model.to("cuda")
# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(r"F:\3DGS_code\dataset\DIV2K_valid_LR_bicubic\X2\0801x2.png")
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
print(depth)
torch.save(depth, 'output/depth.pth')
focallength_px = prediction["focallength_px"]  # Focal length in pixels.