from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip,  RandomVerticalFlip

def get_transform(train=True):
    if train:
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return train_transform
    else:
        test_transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return test_transform