
from PIL import Image
from torchvision import transforms, models
import numpy as np
import glob
import cv2


def load_image_and_preprocess(img_path, max_size=400, shape=None):

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
        im = load_image_and_preprocess(file)
        style_image_list.append(im)
    for file in glob.glob(content_folder + '*.jpg'):
        im = load_image_and_preprocess(file)
        content_image_list.append(im)

    return style_image_list, content_image_list


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    #  (0.5, 0.5, 0.5)
    #image = image * np.array((0.229, 0.224, 0.225)) + np.array(
    #    (0.485, 0.456, 0.406))
    #image = image * np.array((0.5, 0.5, 0.5)) + np.array(
    #        (0.5, 0.5, 0.5))
    image = image.clip(0, 1)

    return image


def save_tensor_to_file(img, save_path):
    img = im_convert(img)
    final_styled_cv2 = np.uint8(255 * img)
    final_styled_cv2_bgr = final_styled_cv2[:, :, [2, 1, 0]]
    cv2.imwrite(save_path, final_styled_cv2_bgr)
    #  print("Image saved to: ", save_path)
