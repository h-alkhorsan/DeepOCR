from model import CRNN
from dataset import CustomDataset, split_images, get_clean_images
from transforms import ToTensor, Resize, Normalize
import json 
from torch.utils.data import DataLoader


# read once
with open('.data/annotations.json') as json_file:
    annotations_data = json.load(json_file)

transformations = [
    ToTensor(),
    Resize((32,32)),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]


clean_images = get_clean_images('.data/images', annotations_data)
train_images, val_images, test_images = split_images(clean_images)

train_dataset = CustomDataset('.data/images', train_images, annotations_data, transform=transformations)
val_dataset = CustomDataset('.data/images', val_images, annotations_data, transform=transformations)
test_dataset = CustomDataset('.data/images', test_images, annotations_data, transform=transformations)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

sample = train_dataset[0]
print(sample[0].size(), len(sample[1]))