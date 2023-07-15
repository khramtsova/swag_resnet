
import torch
import torch.optim as optim
import time
from model.resnet import get_features_resnet
from utils.loss import gram_matrix
from data.folder_preprocess import get_images_from_folders, save_tensor_to_file
from torchvision.models import resnet50
from data.CIFAR10_data_module import CombinedCifarDataModule, CifarCorruptedDataModule, \
    CifarCorruptedDataset, CifarValDataModule
from torchvision import transforms, datasets


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def train(content_image, style_image, model):
    target = content_image.clone().cuda().requires_grad_(True)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    optimizer = optim.LBFGS([target])

    content_layer = 'conv3_5'
    style_layer_list = ['conv0_0', 'conv1_2', 'conv2_3', 'conv3_5', 'conv4_2']
    style_weight, content_weight = 1e17, 1
    # optimize directly the image "target"
    num_steps = 1001

    content_features = get_features_resnet(content_image, model)
    style_features = get_features_resnet(style_image, model)
    style_layer_weights = {'conv0_0': 1.0,
                     'conv1_2': 1.0,
                     'conv2_3': 1.0,
                     'conv3_5': 1.0,
                     'conv4_2': 1.0}
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    t1 = time.time()
    for step in range(0, num_steps, 20):
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

                layer_style_loss = style_layer_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += style_weight * layer_style_loss

            total_loss = content_weight * content_loss + style_loss
            total_loss.backward()
            return content_weight * content_loss + style_loss

        optimizer.step(closure)
        if step % 500 == 0:
            print("Step [{}/{}]".format(step, num_steps))
            # save_tensor_to_file(target, "./results/cifar_resnet50/target_img_st{}.png".format(step))
    t2 = time.time()
    print("time needed per sample", t2 - t1)
    return target


def test_by_folder(model):

    style_img_list, content_image_list = get_images_from_folders(style_folder='./style_images/',
                                                                 content_folder='./content_images/')
    target_img = train(content_image_list[0], style_img_list[0], model)
    save_tensor_to_file( style_img_list[0], "./results/cifar_resnet50/test/style_img.png")
    save_tensor_to_file(content_image_list[0], "./results/cifar_resnet50/test/content_img.png")
    save_tensor_to_file(target_img, "./results/cifar_resnet50/test/target_img.png")



if __name__ == "__main__":
    # Folder-based loader
    # style_img_list, content_image_list = get_images_from_folders(style_folder='./style_images/',
    #                                                             content_folder='./content_images/')
    # target_img = train(content_image_list[0], style_img_list[0], model)

    # Model-related
    #repo = 'pytorch/vision'
    model_weights = torch.load("./model/model_weights/resnet50-11ad3fa6.pth")
    # weights = torch.hub.load("./model/model_weights/", 'resnet50', weights='ResNet50_Weights.DEFAULT')
    model = resnet50().to(device).eval()
    model.load_state_dict(model_weights)
    # model = resnet50().to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    #test_by_folder(model)
    # raise

    # Data-related
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    index = 1

    # Style image comes from a clean dataset
    clean_dataset = datasets.CIFAR10("/opt/data/CIFAR10/",
                                     train=False, download=False, transform=transform)

    # Mine for style images per class
    clean_images = {}
    for img, label in clean_dataset:
        clean_images[label] = img

        if len(clean_images) == 10:
            break
    # Save style images:
    #for label in range(10):
    #    save_tensor_to_file(clean_images[label], "./results/cifar_resnet50/style_img_st{}.png".format(label))
    #raise
    # style_img = clean_dataset[index][0].unsqueeze(0)

    # Content image comes from a corrupted dataset
    corrupted = CifarCorruptedDataset(base_c_path='/opt/data/CIFAR-10-C/',
                                      corruption='gaussian_noise', severity=5,
                                      transform=transform)

    for index in range(len(corrupted)):
        # Introduce a batch dimension
        content_img = corrupted[index][0].unsqueeze(0)
        content_label = corrupted[index][1]

        save_tensor_to_file(content_img, "./results/cifar_resnet50/{}/content_img_st{}_c{}.png"
                            .format(content_label, index, content_label))

        # For each corrupted image - optimize for 10 style images, one per class
        for label in range(10):
            style_img = clean_images[label].unsqueeze(0)
            print(style_img.shape)
            print(content_img.shape)
            # save the target image
            save_tensor_to_file(style_img, "/opt/logs/results/cifar_resnet50/{}/style_img_st{}_c{}.png"
                                .format(content_label, index, label))

            target_img = train(content_img, style_img, model)

            # save the target image
            save_tensor_to_file(target_img, "/opt/logs/results/cifar_resnet50/{}/target_img_st{}_c{}.png"
                                .format(content_label, index, label))




