from PIL import Image
import torch
from torchvision import transforms
from model import create_model
from dataset import get_dataloaders


def load_model(model_path, num_classes, device):

    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, device, class_names):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image = Image.open(image_path).convert('RGB')

    image = transform(image).unsqueeze(0)


    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_label = class_names[predicted.item()]
    return class_label


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader = get_dataloaders('data/food-101/images', batch_size=32, split_ratio=0.8)
    class_names = test_loader.dataset.dataset.classes

    model = load_model('food101_model.pth', num_classes=len(class_names), device=device)

    image_path = 'C:/Users/MAYANK/PycharmProjects/pythonProject6/data/food-101/images/apple_pie/134.jpg'
    predicted_class = predict_image(model, image_path, device, class_names)
    print(f'The predicted class is: {predicted_class}')
