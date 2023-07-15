
import torch
import torch.optim as optim

from model.resnet import get_features_resnet
from utils.loss import gram_matrix
from data.folder_preprocess import get_images_from_folders
from torchvision.models import resnet50


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(content_image, style_image, model):
    target = content_image.clone().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([target])

    content_layer = 'conv3_5'
    style_layer_list = ['conv0_0', 'conv1_2', 'conv2_3', 'conv3_5', 'conv4_2']
    style_weight, content_weight = 1e17, 1
    # optimize directly the image "target"
    num_steps = 1000
    content_features = get_features_resnet(content_image, model)
    style_features = get_features_resnet(style_image, model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    for step in range(num_steps):
        def closure():
            optimizer.zero_grad()
            target_features = get_features_resnet(target, model)

            content_loss = torch.mean((target_features[content_layer] -
                                       content_features[content_layer]) ** 2)
            style_loss = 0
            for layer in style_layer_list:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                # _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]

                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += style_weight * layer_style_loss

            total_loss = content_weight * content_loss + style_loss
            total_loss.backward()
            print(total_loss)
            return content_weight * content_loss + style_loss

        optimizer.step(closure)

    return target


if __name__ == "__main__":
    style_img_list, content_image_list = get_images_from_folders()
    #repo = 'pytorch/vision'
    #weights = torch.hub.load(repo, 'resnet50', weights='ResNet50_Weights.DEFAULT')
    #model = resnet50(weights='ResNet50_Weights.DEFAULT').to(device).eval()
    model = resnet50().to(device).eval()
    train(content_image_list[0], style_img_list[0], model)
    print(style_img_list)
    print(content_image_list)






