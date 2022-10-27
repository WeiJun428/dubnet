from dubnet import *

mnist = 0

inputs = 784 if mnist else 3072

def softmax_model():
    l = [make_connected_layer(inputs, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(inputs, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
if mnist:
    train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
    test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
else:
    train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
    test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 3000
rate = .01
momentum = .9
decay = .0005

# m = softmax_model()
m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

"""
For MNIST, I achieve
training accuracy: %f 0.9803833365440369
test accuracy:     %f 0.9728000164031982

For CIFAR, I achieve
training accuracy: %f 0.5413399934768677
test accuracy:     %f 0.5102999806404114

When I change the model, CIFAR and MNIST behave similarly, except CIFAR seems to be slightly
more responsive to changes
"""