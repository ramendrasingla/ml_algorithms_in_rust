import numpy as np
from torchvision import datasets, transforms

def save_cifar10():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),         # Convert to tensor
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    # Convert dataset to numpy arrays
    def dataset_to_numpy(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(np.array(img))
            labels.append(label)
        return np.array(images), np.array(labels)

    train_images, train_labels = dataset_to_numpy(train_dataset)
    test_images, test_labels = dataset_to_numpy(test_dataset)

    # Save to .npz files
    np.savez_compressed("./data/cifar10_train.npz", images=train_images, labels=train_labels)
    np.savez_compressed("./data/cifar10_test.npz", images=test_images, labels=test_labels)

    print("CIFAR-10 dataset downloaded and saved as .npz files in 'data' directory.")

if __name__ == "__main__":
    save_cifar10()
