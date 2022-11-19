#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"

#define max(x, y) ((x) > (y) ? (x) : (y))


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    tensor y = tensor_vmake(4,
        x.size[0],  // same # data points and # of channels (N and C)
        x.size[1],
        (x.size[2]-1)/l->stride + 1, // H and W scaled based on stride
        (x.size[3]-1)/l->stride + 1);

    // This might be a useful offset...
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int ptr = 0;
    for (int i = 0; i < x.size[0]; i++) {
        for (int j = 0; j < x.size[1]; j++) {
            int offset = (i * x.size[1] + j) * x.size[2] * x.size[3];
            for (int a = 0; a < x.size[2]; a += l->stride) {
                for (int b = 0; b < x.size[3]; b += l->stride) {
                    float mx = -1;
                    for (int c = 0; c < l->size; c++) {
                        for (int d = 0; d < l->size; d++) {
                            int xx = a + pad + c;
                            int yy = b + pad + d;
                            if (xx < 0 || xx >= x.size[2] || yy < 0 || yy >= x.size[3]) {
                                continue;
                            }
                            mx = max(mx, x.data[offset + xx * x.size[3] + yy]);
                        }
                    }
                    y.data[ptr++] = mx;
                }
            }
        }
    }

    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x    = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int ptr = 0, col = 0, row = 0;
    for (int i = 0; i < x.size[0]; i++) {
        for (int j = 0; j < x.size[1]; j++) {
            int offset = (i * x.size[1] + j) * x.size[2] * x.size[3];
            for (int a = 0; a < x.size[2]; a += l->stride) {
                for (int b = 0; b < x.size[3]; b += l->stride) {
                    float mx = -1;
                    for (int c = 0; c < l->size; c++) {
                        for (int d = 0; d < l->size; d++) {
                            int xx = a + pad + c;
                            int yy = b + pad + d;
                            if (xx < 0 || xx >= x.size[2] || yy < 0 || yy >= x.size[3]) {
                                continue;
                            }
                            if (x.data[offset + xx * x.size[3] + yy] > mx) {
                                mx = x.data[offset + xx * x.size[3] + yy];
                                row = xx;
                                col = yy;
                            }
                        }
                    }
                    dx.data[offset + row * x.size[3] + col] += dy.data[ptr++];
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

