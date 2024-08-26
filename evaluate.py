import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_predictions(model, test_loader, device):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    imshow(torchvision.utils.make_grid(images))
    print('Predicted:', ' '.join(f'{test_loader.dataset.dataset.classes[preds[j]]}' for j in range(4)))