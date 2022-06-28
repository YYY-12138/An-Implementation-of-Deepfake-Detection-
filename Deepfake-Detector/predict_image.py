import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from PIL import Image

def main():
	args = parse.parse_args()
	model_path = args.model_path
	image_path = args.image_path
	torch.backends.cudnn.benchmark=True
	model= model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	img_cv = Image.open(image_path)
	transform2 = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 224)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)
	]
	)
	img_cv_Tensor = transform2(img_cv)
	image = img_cv_Tensor.unsqueeze(0)
	with torch.no_grad():
			image = image.cuda()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			#print(preds)
			if(preds == 0):
				print("fake")
			else:
				print("real")


if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--model_path', '-mp', type=str, default='model.pth')
	parse.add_argument('--image_path', '-ip', type=str, default='real_deepfakedetection.png')
	main()
	