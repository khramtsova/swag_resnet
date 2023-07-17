
import torch
import torch.optim as optim
import time
from model.resnet import get_features_resnet
from utils.loss import gram_matrix
from data.folder_preprocess import get_images_from_folders, save_tensor_to_file
from torchvision.models import resnet50

from robustbench.utils import load_model
from model.wideresnet import WideResNet

from data.CIFAR10_data_module import CombinedCifarDataModule, CifarCorruptedDataModule, \
    CifarCorruptedDataset, CifarValDataModule
from torchvision import transforms, datasets
import argparse
import os
from utils.custom_logger import Logger


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def get_prediction(model, image):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        _, output = model(image)
        prediction = output.argmax(1)
    return prediction


def train(content_image, style_image, model, log_dict, style_weight):
    target = content_image.clone().cuda().requires_grad_(True)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    optimizer = optim.LBFGS([target])

    content_layer = "conv2_3"  #  'conv2_3'
    style_layer_list = ['conv0_0',
                        'conv1_3',
                        'conv2_3',
                        'conv3_3']  #,  'conv2_3', 'conv3_5', 'conv4_2']
    content_weight = 1  #
    # optimize directly the image "target"
    num_steps = 1

    content_features = model.get_early_features_wideresnet(content_image)
    style_features = model.get_early_features_wideresnet(style_image)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    t1 = time.time()
    for step in range(1, num_steps+1):

        def closure():
            optimizer.zero_grad()
            target_features = model.get_early_features_wideresnet(target)

            content_loss = torch.mean((target_features[content_layer] -
                                       content_features[content_layer]) ** 2)
            style_loss = 0
            for layer in style_layer_list:

                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                # _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]

                #  style_layer_weights[layer] *
                layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += style_weight * layer_style_loss

            total_loss = content_weight * content_loss + style_loss
            total_loss.backward()
            return content_weight * content_loss + style_loss

        optimizer.step(closure)
        #if step % 10 == 0:
        #    print("Step [{}/{}]".format(step, num_steps))
        # save_tensor_to_file(target, "/opt/logs/results/DEL/cifar_resnet50/target_img_st{}.png".format(step))

        if step in [1, 5]:
            # print("Step: ", step)
            # print("Time: ", time.time() - t1)
            # t1 = time.time()
            pred = get_prediction(model, target)

            log_dict["prediction"] = pred.item()
            log_dict["step"] = step
            # print("Log dict: ", log_dict)
            csv_logger.add_record_to_log_file(log_dict)
            # create a folder for the current corruption type

            dir_to_save = "{}/{}/".format(csv_logger.dir_to_save, log_dict["content_label"])
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)
            # save the target image
            save_tensor_to_file(target, "{}/target_st{}_c{}_step{}.png"
                                .format(dir_to_save, log_dict["content_index"], log_dict["style_label"], step))

    t2 = time.time()

    # save_tensor_to_file(style_img, "/opt/logs/results/DEL/cifar_resnet50/style_img_st{}.png".format(step))
    # print("time needed per sample", t2 - t1)
    return target


if __name__ == "__main__":
    # Folder-based loader
    # style_img_list, content_image_list = get_images_from_folders(style_folder='./style_images/',
    #                                                             content_folder='./content_images/')
    # target_img = train(content_image_list[0], style_img_list[0], model)

    # read corruption type from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str, default="gaussian_noise", help="corruption type")
    parser.add_argument("--style_weight", type=int, default=1000, help="weight of the style loss")
    parser.add_argument("--log_dir", type=str, default="/opt/logs/", help="log directory")
    parser.add_argument("--data_dir", type=str, default="/opt/data/", help="data directory")
    args = parser.parse_args()

    dir_to_save = "{}/results/cifar_wideresnet28/{}/{}/".format(args.log_dir,
                                                                       args.style_weight,
                                                                       args.corruption)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    csv_logger = Logger(dir_to_save, "{}_summary_{}.csv".format(args.corruption, args.style_weight))

    # Model-related
    model = WideResNet()
    model_weights = load_model(model_name="Standard",
                            dataset="cifar10",
                            model_dir=args.data_dir + "/CV_models/",
                            threat_model="corruptions").state_dict()
    model.load_state_dict(model_weights)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)


    #test_by_folder(model)
    # raise

    # Data-related
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    index = 1
    # Content image comes from a corrupted dataset
    clean_dataset = datasets.CIFAR10("{}/CIFAR10/".format(args.data_dir),
                                     train=False, download=False, transform=transform)

    # Mine for style images per class
    clean_images = {}
    for img, label in clean_dataset:
        if type(label) == torch.Tensor:
            label = label.item()
        clean_images[label] = img

        if len(clean_images) == 10:
            break
    # Save style images:
    #for label in range(10):
    #    save_tensor_to_file(clean_images[label], "./results/cifar_resnet50/style_img_st{}.png".format(label))
    #raise
    # style_img = clean_dataset[index][0].unsqueeze(0)

    # Style image comes from a clean dataset
    corrupted = CifarCorruptedDataset(base_c_path='{}/CIFAR-10-C/'.format(args.data_dir),
                                      corruption=args.corruption, severity=5,
                                      transform=transform)

    for index in range(len(corrupted)):
        # Introduce a batch dimension
        content_img = corrupted[index][0].unsqueeze(0)
        content_label = corrupted[index][1]

        #save_tensor_to_file(content_img, "/opt/logs/results/DEL/cifar_resnet50/{}/{}/content_img_st{}_c{}.png"
        #                    .format(args.corruption, content_label,  index, content_label))

        pred = get_prediction(model, content_img)
        temp = {"corruption": args.corruption,
                "content_index": index,
                "content_label": content_label.item(),
                "style_label": -1,
                "prediction": pred.item(),
                "step": -1}
        csv_logger.add_record_to_log_file(temp)

        # For each corrupted image - optimize for 10 style images, one per class
        for label in range(10):

            # Logging
            temp = {"corruption": args.corruption,
                    "content_index": index,
                    "content_label": content_label.item(),
                    "style_label": label,
                    }

            style_img = clean_images[label].unsqueeze(0)
            target_img = train(content_img, style_img, model, log_dict=temp, style_weight=args.style_weight)

        if index % 100 == 0:
            print("Done with {} images".format(index))




