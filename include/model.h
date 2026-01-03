#ifndef MODEL_H
#define MODEL_H

#include "loading.h"
#include "tensor4f.h"
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct Model
{
    float step = 0.001;
    float rho1 = 0.9, rho2 = 0.999; // First and second moment estimate decay
    size_t timestep = 0;
    std::mt19937 rng;

    IdxImages _testImages;
    IdxLabels _testLabels;
    IdxImages _trainingImages;
    IdxLabels _trainingLabels;
    IdxImages _validationImages;
    IdxLabels _validationLabels;

    std::vector<Tensor4F> batchesQueued;
    std::vector<std::vector<uint8_t>> batchLabelsQueued;

    // Parameters
    Tensor4F W1, B1;
    Tensor4F W2, B2;
    Tensor4F W3, B3;

    // Gradients
    Tensor4F dW1, dB1;
    Tensor4F dW2, dB2;
    Tensor4F dW3, dB3;

    // First and second moments for adam
    Tensor4F mW1, vW1, mB1, vB1;
    Tensor4F mW2, vW2, mB2, vB2;
    Tensor4F mW3, vW3, mB3, vB3;

    Model();

    void load_data(
        std::string imagePathTest, 
        std::string labelPathTest, 
        std::string imagePathTraining,
        std::string labelPathTraining
    );
    void reset_batches();
    void adam(size_t patience);
    void test();
    
    void zero_gradients();
};

struct ForwardCache
{
    Tensor4F X0; // Input

    // Convolutional layer 1
    Tensor4F Z1; 
    Tensor4F reluMask1;
    Tensor4F A1;
    Tensor4F poolArgMax1;
    Tensor4F P1;

    // Convolutional layer 2
    Tensor4F Z2; 
    Tensor4F reluMask2;
    Tensor4F A2;
    Tensor4F poolArgMax2;
    Tensor4F P2;

    // Final FC layer
    Tensor4F F;
    Tensor4F Y;
    Tensor4F probs;
};

static void shuffle_data(IdxImages& images, IdxLabels& labels, std::mt19937& rng);

#endif