import torch
from segmentation.segmentation import deeplabv3plus_resnet50

model = deeplabv3plus_resnet50(pretrained=True)
model.eval()

out = model(torch.randn((1, 3, 224, 224)))

print(out['out'].shape)
