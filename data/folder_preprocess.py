
from PIL import Image
from torchvision import transforms, models

import glob


def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def get_images_from_folders(style_folder='./style_images/', content_folder='./content_images/'):
    # load style and content images

    style_image_list, content_image_list = [], []
    for file in glob.glob(style_folder + '*.jpg'):
        im = load_image(file)
        style_image_list.append(im)
    for file in glob.glob(content_folder + '*.jpg'):
        im = load_image(file)
        content_image_list.append(im)

    return style_image_list, content_image_list
