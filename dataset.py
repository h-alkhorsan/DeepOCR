from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import os 
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

def get_clean_images(image_dir, annot_data):
    images = [image.split('.')[0] for image in os.listdir(image_dir)]
    clean_images = []

    has_image = lambda annot_data, image_id :  True if image_id in annot_data['imgToAnns'] else False
    for image in images:
        if has_image(annot_data, image):
            clean_images.append(image)

    return clean_images 
            

def get_image_annotations(image_id, annotations_data):
    strings = []
    images_annotations = annotations_data['imgToAnns'][image_id]
    for annotation_id in images_annotations:
        strings.append(annotations_data['anns'][annotation_id]['utf8_string'])
    return strings 

def split_images(images, train_size=0.8, val_size=0.1):

    remaining_size = 1.0 - train_size 

    val_size_remaining = val_size / remaining_size 

    train_images, remaining_images = train_test_split(images, train_size=train_size, shuffle=True, random_state=42)
    val_images, test_images = train_test_split(remaining_images, train_size=val_size_remaining, shuffle=True, random_state=42)
    return train_images, val_images, test_images

def images_to_annotations(annotations_filepath, images_filepath):
    images = os.listdir(images_filepath)

    dataset = defaultdict(list)

    with open(annotations_filepath) as json_file:
        data = json.load(json_file)

    key_error_count = 0
    for image in images:        
        try:
            image_id, _ = os.path.splitext(image)
            image_annotations = data['imgToAnns'][image_id]
            for annotation_id in image_annotations:
                dataset[image_id].append({data['anns'][annotation_id]['id'] : data['anns'][annotation_id]['utf8_string']})

                # print(image_id, data['anns'][annotation_id])
                    # assert False
                # dataset[image_id][annotation_id] = data['anns'][image_id + '_' + annotation_id]

        except KeyError:
            key_error_count += 1
            continue      

class CustomDataset(Dataset):
    def __init__(self, root_dir, indices, annotations_data, transform=None):
        self.indices = indices 
        self.transform = transform 
        self.root_dir = root_dir
        self.annotations_data = annotations_data
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.max_length = 40

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # image = Image.open(self.indices[idx])
        image = Image.open(os.path.join(self.root_dir, self.indices[idx] + '.jpg'))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # assert(image.mode == 'RGB') # remove
        if self.transform:
            for T in self.transform:
                image = T(image)

        image_id = self.indices[idx].split('/')[-1].split('.')[0]
        annotations = get_image_annotations(image_id, self.annotations_data)
        encoding = self.tokenizer(annotations, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        label_encoded = encoding['input_ids'].squeeze(0)
        return image, label_encoded