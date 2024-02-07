import numpy as np #maths and pixels
import cv2 #opencv for ip
import matplotlib.pyplot as plt #for visualization
import os
from imgaug import augmenters as iaa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def create_image_with_simp_shape(shape, width, height, background_color,shape_color):
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:] = background_color

    if shape == "circle":
        cv2.circle(image, (width//2, height//2), min(width, height)//3, shape_color, -1)
        cv2.rectangle(image, ((width+100)//4, (height+200)//4), (3*width//4, 3*height//4), shape_color, -1)
    elif shape == "square":
        cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), shape_color, -1)
    elif shape == "triangle":
        pts = np.array([[width//4, height//4], [3*width//4, height//4], [width//2, 3*height//4]])
        cv2.fillPoly(image, [pts], shape_color)
    elif shape == "rectangle":
        cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), shape_color, -1)

    return image

def create_mask(image):
      
      mask = cv2.compare(image, np.array([255, 255, 255]), cv2.CMP_EQ)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
      mask = cv2.bitwise_not(mask)
      gray[mask == 255] = 255 
      gray[mask != 255] = 0 
      return gray

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
  
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):  
        x1 = self.encoder(x)  
        x2 = self.decoder(x1)
        return x2

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

num_epochs = 15
batch_size = 4
learning_rate = 0.001

num_images = 100  #before augmentation
output_img=r'C:\Desktop\algosupassign\img here\img'
output_mask=r'C:\Desktop\algosupassign\img here\masked_img'
os.makedirs(output_img, exist_ok=True)
os.makedirs(output_mask, exist_ok=True)

dataset_images = []
dataset_masks = []

for i in range(num_images):
    shape = np.random.choice(["circle", "square","triangle","rectangle"])  
    width = np.random.randint(200, 300)
    height = np.random.randint(200, 300)
    shape_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  
    random_index = np.random.randint(0, len(shape_colors))

    shape_color = shape_colors[random_index]
    background_color=(255, 255, 255)
   
    image = create_image_with_simp_shape(shape, width, height, background_color ,shape_color )
    gray = create_mask(image)
  

    output_pathimg = os.path.join(output_img, f"image_{i}.png")
    cv2.imwrite(output_pathimg, image)

    output_pathmask = os.path.join(output_mask, f"mask_{i}.png")
    cv2.imwrite(output_pathmask,gray )

   
    dataset_images.append(image)
    dataset_masks.append(gray)
    
print("image done")
plt.imshow(image)
plt.show()

"C:\Desktop\algosupassign\img here\img\image_{i}.png"
"C:\Desktop\algosupassign\img here\masked_img\mask_{i}.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_paths = [f"C:\Desktop\algosupassign\img here\img\image_{i}.png" for i in range(num_images)]
mask_paths = [f"C:\Desktop\algosupassign\img here\masked_img\mask_{i}.png" for i in range(num_images)]
transform = ToTensor()
dataset = CustomDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("images dataset done")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("starting training")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print("training on")
    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        print(running_loss)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


torch.save(model.state_dict(), "segmentation_model.pth")
print("training done")

model.eval()


image_tensor = transform(image).unsqueeze(0).to(device)#testing the model here
with torch.no_grad():
        output = model(image_tensor)
        segmented_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
plt.imshow(segmented_mask)
plt.show()


