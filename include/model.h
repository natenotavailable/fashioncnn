#ifndef MODEL_H
#define MODEL_H

#include "loading.h"
#include "tensor4f.h"
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

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
    Tensor4F logits;
    Tensor4F probs;
};

struct Model
{
    const float trainstep = 0.001;
    const float rho1 = 0.9, rho2 = 0.999; // First and second moment estimate decay
    const float epsilon = 1e-8f; // Constant for numerical stability
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
    ForwardCache cache;

    // Parameters
    Tensor4F W1, B1;
    Tensor4F W2, B2;
    Tensor4F W3, B3;

    // Best so far parameters
    Tensor4F bestW1, bestB1;
    Tensor4F bestW2, bestB2;
    Tensor4F bestW3, bestB3;

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
    float train_batch(const Tensor4F& X, const std::vector<uint8_t>& labels);
    float validation_loss();
    void test();
    
    void zero_gradients();
    void forward(const Tensor4F& X);
    float backward(const std::vector<uint8_t>& labels);

    void save_best_params();
    void load_best_params();

};

static void shuffle_data(IdxImages& images, IdxLabels& labels, std::mt19937& rng);

#endif