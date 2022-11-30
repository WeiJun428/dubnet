from dubnet import *
def conv_net():
    l = [   make_convolutional_layer(3, 8, 3, 2),
            make_batchnorm2d_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(8, 16, 3, 1),
            make_batchnorm2d_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(16, 32, 3, 1),
            make_batchnorm2d_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .1
momentum = .9
decay = .005

#m = conn_net()
m = conv_net()
print("training...")
train_image_classifier(m, train, batch, 300, 0.1, momentum, decay)
train_image_classifier(m, train, batch, 150, 0.01, momentum, decay)
train_image_classifier(m, train, batch, 150, 0.001, momentum, decay)
train_image_classifier(m, train, batch, 100, 0.0001, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:

# Without Batch normalization:
# training accuracy: %f 0.41238000988960266
# test accuracy:     %f 0.4072999954223633

# With Batch normalization
# training accuracy: %f 0.5342000126838684
# test accuracy:     %f 0.527400016784668

# The accuracy of convnet increases by 12% after adding batch normalization.
# It also converges much faster (more stable).
# The magnitude of learning rate also increases. With 0.1 learning rate and train for 500 iterations,
# the model achieves similar performance and does not suffer for non-covergence:
# training accuracy: %f 0.5289400219917297
# test accuracy:     %f 0.5202999711036682

# By training for 600 epochs with learning rate 0.1 (300 epochs), 0.01 (150 epochs), 0.001 (150 epochs), and 0.0001 (100 epochs),
# the model achieve its best accuracy:
# training accuracy: %f 0.5598400235176086
# test accuracy:     %f 0.5523999929428101