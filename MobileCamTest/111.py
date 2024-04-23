import torch
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device
from utils.datasets import LoadImagesAndLabels
from utils.metrics import ConfusionMatrix
import matplotlib.pyplot as plt

weights = 'best_adam.pt'
device = select_device('')
model = attempt_load(weights, map_loc[ati](https://geek.csdn.net/educolumn/150d7073277950db5f09620704e791cf?spm=1055.2569.3001.10083)on=device)
imgsz = check_img_size(720,s=model.stride.max())
img = torch.zeros((1,3,imgsz,imgsz),device=device)
pred = model(img)
data = 'disk.yaml'
imgsz = check_img_size(imgsz, s = model.stride.max())
dataset = LoadImagesAndLabels(data, image = imgsz, batch_size = 1, rect = False)
labels = [x[1].tolist() for x in dataset.dataset]

pred_boxes = pred[0][:, :4].detach().cpu().numpy()
pred_labels = pred[0][:, :-1].detach().cpu().numpy()

confusion_matrix = ConfusionMatrix(num_classes = dataset.nc)
confusion_matrix.process_batch(pred_boxes,pred_labels,labels)

fig, ax = plt.subplots(figsize=(10, 10))
confusion_matrix.plot(ax =ax, normalize = True)
plt.show()

