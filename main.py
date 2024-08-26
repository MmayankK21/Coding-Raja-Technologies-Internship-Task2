import torch
from dataset import get_dataloaders, download_food101
from model import create_model
from train import train_model
from evaluate import evaluate_model

def main():

    download_food101('data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders('data/food-101/images', batch_size=32, split_ratio=0.8)

    model = create_model(num_classes=len(train_loader.dataset.dataset.classes))
    model = model.to(device)

    train_model(model, train_loader, device, epochs=10, learning_rate=0.001)

    torch.save(model.state_dict(), 'food101_model.pth')

    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()
