#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer y = f(x)
tensor forward_activation_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    ACTIVATION a = l->activation;
    tensor y = tensor_copy(x);

    // TODO: 2.0
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row

    assert(x.n >= 2);

    size_t i, j;
    float sum;
    for(i = 0; i < x.size[0]; ++i){
        tensor x_i = tensor_get_(x, i);
        tensor y_i = tensor_get_(y, i);
        size_t len = tensor_len(x_i);
        sum = 0;
        for (j = 0; j < len; j++) {
            float temp = x_i.data[j];
            if (a == LRELU) {
                y_i.data[j] = temp >= 0.0 ? temp : temp * 0.01;
            } else if (a == RELU) {
                y_i.data[j] = fmax(temp, 0.0);
            } else if (a == SOFTMAX) {
                y_i.data[j] = expf(temp);
                sum += y_i.data[j];
            } else if (a == LOGISTIC) {
                y_i.data[j] = 1.0 / (1.0 + expf(-temp));
            }
        }
        if (a == SOFTMAX) {
            for (j = 0; j < len; j++) {
                y_i.data[j] /= sum;
            }
        }
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
tensor backward_activation_layer(layer *l, tensor dy)
{
    tensor x = l->x;
    tensor dx = tensor_copy(dy);
    ACTIVATION a = l->activation;

    // TODO: 2.1
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    size_t i, j;
    for(i = 0; i < dx.size[0]; ++i){
        tensor x_i = tensor_get_(x, i);
        tensor dx_i = tensor_get_(dx, i);
        size_t len = tensor_len(dx_i);
        for (j = 0; j < len; j++) {
            float temp = x_i.data[j];
            if (a == LRELU) {
                dx_i.data[j] *= (temp > 0 ? 1 : 0.01f);
            } else if (a == RELU) {
                dx_i.data[j] *= (temp > 0 ? 1 : 0);
            } else if (a == LOGISTIC) {
                float log = 1 / (1 + expf(-temp));
                dx_i.data[j] *= log * (1 - log);
            }
        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer *l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
