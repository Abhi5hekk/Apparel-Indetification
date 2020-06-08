"""
Expose a method `tag` that accepts a PIL Image 
based image as input
"""

from tensorflow.keras.applications import resnet50
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

IMG_SIZE = 224
NUM_OUTPUTS = 3

EPOCHS = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.00007
NUM_FEATURES = 1000
NUM_CLASSES = NUM_OUTPUTS

class MulticlassClassification2(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification2, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

torch_model2 = MulticlassClassification2(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
torch_model2.load_state_dict(torch.load('torch-2-vvlarge.dict', map_location=torch.device('cpu')))

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

resnet50_model = resnet50.ResNet50(weights='imagenet')

# classify_model = pickle.load(open('model.pickle', 'rb'))
scaler_model = pickle.load(open('torch-2-vvlarge.scaler.pickle', 'rb'))



labels = ["Shirt", "TShirt", "Pants"]
# labels = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def tag(image):
  image = image.resize((IMG_SIZE, IMG_SIZE))
  image = np.array(image)

  img_batch = np.expand_dims(image, axis=0)
  processed_batch = preprocess_input(img_batch, mode="caffe")
  vec_batch = resnet50_model.predict(processed_batch)
  testx2 = scaler_model.transform(vec_batch)
  testy2 = np.zeros(testx2.shape[0])
  test_dataset2 = ClassifierDataset(torch.from_numpy(testx2).float(), torch.from_numpy(testy2).long())
  test_loader2 = DataLoader(dataset=test_dataset2, batch_size=1)
  y_pred_list = []
  with torch.no_grad():
      torch_model2.eval()
      for X_batch, _ in test_loader2:
          y_test_pred = torch_model2(X_batch)
          y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
          _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
          y_pred_list.append(y_pred_tags.cpu().numpy())
  y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
  print(y_pred_list)
  prediction = y_pred_list[0]
  # pred_batch = classify_model.predict(vec_batch)
  # prediction = pred_batch[0]

  print(f'{prediction} - { labels[prediction] }')

  return {"tag": str(prediction), "label": labels[prediction]}
  # img = cv2.resize(raw_image, (28, 28))
  # imgd = prepare(img)
  # return labels[np.argmax(model.predict(imgd))]
