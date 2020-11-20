import torchvision.transforms as transforms


def image_resize(image):
    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    image_tensor = t(image)
    return image_tensor