import torch
from torchvision import models
from PIL import Image
from lightning_modules.finetune_dex import FinetuneDEX
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms

image = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# TODO: Set to weights=VGG16_Weights.DEFAULT (?)
vgg16 = models.vgg16(pretrained=True)

# "When training for classification, the output layer is adapted to 101 output neurons
# corresponding to natural numbers from 0 to 100..."
vgg16.classifier[6] = torch.nn.Linear(4096, 101)

# CHECKPOINT = "my_model_epoch=09_val_loss=0.00.ckpt"
CHECKPOINT = "my_model_epoch=00_val_loss=0.00.ckpt"
model = FinetuneDEX.load_from_checkpoint(
    CHECKPOINT,
    model=vgg16,
    batch_size=128,
    learning_rate=1e-3,
)
model.eval()

with torch.no_grad():
    output = model(torch.unsqueeze(transform(image), 0).to("cuda")).to("cpu")

    prob = torch.nn.functional.softmax(output, dim=1)[0]

    age_range = torch.arange(1, 102)  # [1, 2, ..., 101]

    # Get weighted sum of ages to arrive at a single age estimate
    estimated_age = (prob * age_range).sum()

    print(estimated_age)
