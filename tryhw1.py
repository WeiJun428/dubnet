from dubnet import *

def conv_net():
    """
    Convolution FLOPS = (c * size * size) * (n * k * h / stride * w / stride)
    Connected FLOPS = k * n
    Image size of CIFAR = 32 * 32
    Number of filter, k = 1

    *As each convolution is followed by a max pool layer of stride = 2,
    the image size is halved after each convolution.

    FLOPS of 1st convolutional layer (c = 3, n = 8, size = 3, stride = 1) =
        (3 * 3 * 3) * ((8 * 1 * 32) / 1 * 32 / 1) = 221,184

    FLOPS of 2nd convolutional layer (c = 8, n = 16, size = 3, stride = 1) =
        (8 * 3 * 3) * ((16 * 1 * 16) / 1 * 16 / 1) = 294,912

    FLOPS of 3rd convolutional layer (c = 16, n = 32, size = 3, stride = 1) =
        (16 * 3 * 3) * ((32 * 1 * 8) / 1 * 8 / 1) = 294,912

    FLOPS of 4th convolutional layer (c = 32, n = 64, size = 3, stride = 1) =
        (32 * 3 * 3) * ((64 * 1 * 4) / 1 * 4 / 1) = 294,912

    FLOPS of fully connected layer =
        256 * 10 = 2,560

    Total FLOPS = 221,184 + 3 * 294,912 + 2,560 = 1,108,480
    """
    l = [   make_convolutional_layer(3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conn_net():
    """
    Connected FLOPS = k * n
    Image Size = 3 * 32 * 32 = 3072
    FLOPS = 3072 * 320 + 320 * 256 + 256 * 128 + 128 * 64 + 64 * 10 = 1106560
    """
    l = [   make_connected_layer(3072, 320),
            make_activation_layer(RELU),
            make_connected_layer(320, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 128),
            make_activation_layer(RELU),
            make_connected_layer(128, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .005

m = conn_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# Convnet (Convolution)
# - training accuracy: %f 0.4914200007915497
# - test accuracy:     %f 0.4878000020980835
#
# Connnet (Connected)
# - training accuracy: %f 0.4152199923992157
# - test accuracy:     %f 0.41339999437332153
#
# Overall, convolution works better than the fully connected layers as the convolution achieves about 18%
# greater testing accuracy than fully connected network under the same parameters. This might be because
# the convolution is more sparse and the locality of features are heavily considered. The maxpooling
# layer also reduces noise to avoid overfitting and reduces size of inputs/weights. The fully connected layer
# is so dense that most of the weight connecting two distant nodes are not useful, hence makes it weaker when
# the number of operations are similar with the convolution.



