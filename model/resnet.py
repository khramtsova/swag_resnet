


def get_features_resnet(image, model, layers=None):

    if layers is None:
        layers1 = {'0': 'conv1_0',
                   '1': 'conv1_1',
                   '2': 'conv1_2'
                   }
        layers2 = {'0': 'conv2_0',
                   '1': 'conv2_1',
                   '2': 'conv2_2',
                   '3': 'conv2_3'
                   }
        layers3 = {'0': 'conv3_0',
                   '1': 'conv3_1',
                   '2': 'conv3_2',
                   '3': 'conv3_3',
                   '4': 'conv3_4',
                   '5': 'conv3_5'
                   }
        layers4 = {'0': 'conv4_0',
                   '1': 'conv4_1',
                   '2': 'conv4_2'
                   }
    # softmax = nn.Softmax2d()

    # T = 100
    T = 1
    alpha = 0.001

    features = {}
    x = image
    x = model.conv1(x)

    x = model.bn1(x)
    x = model.relu(x)
    features['conv0_0'] = x
    # features['conv0_0'] = softmax(x / T)
    # features['conv0_0'] = softmax3d(x / T)
    # features['conv0_0'] = alpha * x
    x = model.maxpool(x)
    features['conv0_1'] = x
    # features['conv0_1'] = softmax(x / T)
    # features['conv0_1'] = softmax3d(x / T)
    # features['conv0_1'] = alpha * x

    # Although we found adding softmax smoothing can always improve the results, the best results sometimes are obtained by only smoothing the deeper layers,
    # as the paper suggests, deeper layers are more peaky and have small entropy

    for name, layer in enumerate(model.layer1):
        x = layer(x)
        if str(name) in layers1:
            features[layers1[str(name)]] = x
            # features[layers1[str(name)]] = softmax(x / T)
            # features[layers1[str(name)]] = softmax3d(x / T)
            # features[layers1[str(name)]] = alpha * x
    for name, layer in enumerate(model.layer2):
        x = layer(x)
        if str(name) in layers2:
            features[layers2[str(name)]] = x
            # features[layers2[str(name)]] = softmax(x / T)
            # features[layers2[str(name)]] = softmax3d(x / T)
            # features[layers2[str(name)]] = alpha * x
    for name, layer in enumerate(model.layer3):
        x = layer(x)
        if str(name) in layers3:
            # features[layers3[str(name)]] = softmax(x / T)
            # features[layers3[str(name)]] = softmax3d(x / T)
            features[layers3[str(name)]] = alpha * x
    for name, layer in enumerate(model.layer4):
        x = layer(x)
        if str(name) in layers4:
            # features[layers4[str(name)]] = softmax(x / T)
            # features[layers4[str(name)]] = softmax3d(x / T)
            features[layers4[str(name)]] = alpha * x

    return features