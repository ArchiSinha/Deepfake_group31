import torch, sys, json
sys.path.append('.')
from src.discriminator import Discriminator

ckpt = torch.load('checkpoints/best.pt', map_location='cpu')
model = Discriminator(pretrained=False)
model.load_state_dict(ckpt['model'])
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy,
    'checkpoints/deepfake_detector.onnx',
    input_names=['input'],
    output_names=['logit'],
    opset_version=11
)
print('ONNX exported successfully')

with open('checkpoints/threshold.json', 'w') as f:
    json.dump({'thr': 0.5}, f)
print('Threshold saved')