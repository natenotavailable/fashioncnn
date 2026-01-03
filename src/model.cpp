#include "model.h"
#include "loading.h"
#include "tensor4f.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <iostream>

using std::string;
using std::vector;
using std::mt19937;

/**
 * @brief Static helper for adam param updates
 */
static void adam_update(
    Tensor4F& P,
    Tensor4F& M,
    Tensor4F& V,
    const Tensor4F& G,
    float trainstep,
    float rho1, // Rhos are exponential decay rates
    float rho2,
    float epsilon,
    size_t timestep
)
{
    const float r1t = std::pow(rho1, float(timestep));
    const float r2t = std::pow(rho2, float(timestep));
    const float inv1 = 1.0f / (1.0f - r1t); // Bias correction factors
    const float inv2 = 1.0f / (1.0f - r2t);

    // For each param, apply adam update
    for (size_t i = 0; i < P.tensorData.size(); ++i)
    {
        const float gi = G.tensorData[i];

        M.tensorData[i] = rho1 * M.tensorData[i] + (1.0f - rho1) * gi;
        V.tensorData[i] = rho2 * V.tensorData[i] + (1.0f - rho2) * (gi * gi);

        const float mhat = M.tensorData[i] * inv1;
        const float vhat = V.tensorData[i] * inv2;

        P.tensorData[i] -= trainstep * mhat / (std::sqrt(vhat) + epsilon); // epsilon is small constant so no divide by 0
    }
}

void shuffle_data(IdxImages& images, IdxLabels& labels, mt19937& rng)
{
    const size_t n = images.n;
    const size_t imgSize = size_t(images.h) * images.w;

    // Create the permutations.
    vector<size_t> permutations(n);
    std::iota(permutations.begin(), permutations.end(), 0);
    std::shuffle(permutations.begin(), permutations.end(), rng);

    // Apply permutations to image and label buffers.
    vector<float> newImageData(images.imageData.size());
    vector<uint8_t> newLabels(labels.labels.size());

    for (size_t newIdx = 0; newIdx < n; ++newIdx)
    {
        const size_t oldIdx = permutations[newIdx];

        // Copy image block.
        const size_t src = oldIdx * imgSize;
        const size_t dst = newIdx * imgSize;
        std::copy_n(images.imageData.begin() + src, imgSize, newImageData.begin() + dst);

        // Copy label.
        newLabels[newIdx] = labels.labels[oldIdx];
    }

    // Move permutated data into buffers.
    images.imageData = std::move(newImageData);
    labels.labels = std::move(newLabels);
}

Model::Model()
{
    // Architecture is two convolution layers (with activation + maxout) and a fully-connected layer
    W1.resize(8, 1, 3, 3);
    B1.resize(1, 8, 1, 1);
    W2.resize(16, 8, 3, 3);
    B2.resize(1, 16, 1, 1);
    W3.resize(10, 784, 1, 1);
    B3.resize(1, 10, 1, 1);

    // Initialize weights & biases.
    W1.normal(0.0f, 0.05f, 0);
    W2.normal(0.0f, 0.05f, 1);
    W3.normal(0.0f, 0.05f, 2);
    B1.zero();
    B2.zero();
    B3.zero();

    bestW1.resize(W1.n, W1.c, W1.h, W1.w);
    bestB1.resize(B1.n, B1.c, B1.h, B1.w);
    bestW2.resize(W2.n, W2.c, W2.h, W2.w);
    bestB2.resize(B2.n, B2.c, B2.h, B2.w);
    bestW3.resize(W3.n, W3.c, W3.h, W3.w);
    bestB3.resize(B3.n, B3.c, B3.h, B3.w);

    // Initialize gradients & moments
    dW1.resize(W1.n, W1.c, W1.h, W1.w);
    dB1.resize(B1.n, B1.c, B1.h, B1.w);
    dW2.resize(W2.n, W2.c, W2.h, W2.w);
    dB2.resize(B2.n, B2.c, B2.h, B2.w);
    dW3.resize(W3.n, W3.c, W3.h, W3.w);
    dB3.resize(B3.n, B3.c, B3.h, B3.w);

    mW1.resize(W1.n, W1.c, W1.h, W1.w);
    vW1.resize(W1.n, W1.c, W1.h, W1.w);
    mB1.resize(B1.n, B1.c, B1.h, B1.w);
    vB1.resize(B1.n, B1.c, B1.h, B1.w);
    mW2.resize(W2.n, W2.c, W2.h, W2.w);
    vW2.resize(W2.n, W2.c, W2.h, W2.w);
    mB2.resize(B2.n, B2.c, B2.h, B2.w);
    vB2.resize(B2.n, B2.c, B2.h, B2.w);
    mW3.resize(W3.n, W3.c, W3.h, W3.w);
    vW3.resize(W3.n, W3.c, W3.h, W3.w);
    mB3.resize(B3.n, B3.c, B3.h, B3.w);
    vB3.resize(B3.n, B3.c, B3.h, B3.w);

    dW1.zero();
    dW2.zero();
    dW3.zero();
    dB1.zero();
    dB2.zero();
    dB3.zero();

    mW1.zero();
    vW1.zero();
    mB1.zero();
    vB1.zero();
    mW2.zero();
    vW2.zero();
    mB2.zero();
    vB2.zero();
    mW3.zero();
    vW3.zero();
    mB3.zero();
    vB3.zero();
}

void Model::load_data(
    std::string imagePathTest, 
    std::string labelPathTest, 
    std::string imagePathTraining,
    std::string labelPathTraining
)
{
    _testLabels = load_idx1_labels(labelPathTest);
    IdxLabels totalTrainingLabels = load_idx1_labels(labelPathTraining);
    _testImages = load_idx3_images(imagePathTest);
    IdxImages totalTrainingImages = load_idx3_images(imagePathTraining);

    shuffle_data(_testImages, _testLabels, rng);
    shuffle_data(totalTrainingImages, totalTrainingLabels, rng);

    const size_t n = totalTrainingImages.n;
    const size_t imgSize = size_t(totalTrainingImages.h) * totalTrainingImages.w;

    const size_t cutoff = size_t(std::floor(n * 0.9));

    std::vector<float> trainingImages(cutoff * imgSize);
    std::vector<uint8_t> trainingLabels(cutoff);

    std::vector<float> validationImages((n - cutoff) * imgSize);
    std::vector<uint8_t> validationLabels(n - cutoff);

    // Training data
    std::copy_n(
        totalTrainingImages.imageData.begin(),
        cutoff * imgSize,
        trainingImages.begin()
    );

    std::copy_n(
        totalTrainingLabels.labels.begin(),
        cutoff,
        trainingLabels.begin()
    );

    _trainingImages.imageData = std::move(trainingImages);
    _trainingImages.h = totalTrainingImages.h;
    _trainingImages.w = totalTrainingImages.w;
    _trainingImages.n = cutoff;

    _trainingLabels.labels = std::move(trainingLabels);
    _trainingLabels.n = cutoff;

    // Validation data
    std::copy_n(
        totalTrainingImages.imageData.begin() + cutoff * imgSize,
        (n - cutoff) * imgSize,
        validationImages.begin()
    );

    std::copy_n(
        totalTrainingLabels.labels.begin() + cutoff,
        n - cutoff,
        validationLabels.begin()
    );

    _validationImages.imageData = std::move(validationImages);
    _validationImages.h = totalTrainingImages.h;
    _validationImages.w = totalTrainingImages.w;
    _validationImages.n = n - cutoff;

    _validationLabels.labels = std::move(validationLabels);
    _validationLabels.n = n - cutoff;
}

void Model::reset_batches()
{
    // Shuffle training data.
    shuffle_data(_trainingImages, _trainingLabels, rng);

    const size_t n = size_t(_trainingImages.n);
    const size_t h = size_t(_trainingImages.h);
    const size_t w = size_t(_trainingImages.w);
    const size_t imgSize = h * w;

    // Batch into Tensors (batch size 32) & parallel labels.
    constexpr size_t batchSize = 32;
 
    batchesQueued.clear();
    batchLabelsQueued.clear();

    const size_t numBatches = (n + batchSize - 1) / batchSize;
    batchesQueued.reserve(numBatches);
    batchLabelsQueued.reserve(numBatches);

    for (size_t start = 0; start < n; start += batchSize)
    {
        const size_t bn = std::min(batchSize, n - start);

        // (bn, 1, h, w) for inputs (Fashion MNIST has only 1 input channel).
        Tensor4F batch(uint32_t(bn), 1u, uint32_t(_trainingImages.h), uint32_t(_trainingImages.w));

        const size_t srcOffset = start * imgSize;
        const size_t countFloats = bn * imgSize;

        if (batch.tensorData.size() != countFloats)
            throw std::runtime_error("reset_batches: Tensor4F storage size unexpected!");

        std::copy_n(
            _trainingImages.imageData.begin() + srcOffset,
            countFloats,
            batch.tensorData.begin()
        );

        // Add parallel labels for the batch.
        std::vector<uint8_t> y(bn);
        std::copy_n(
            _trainingLabels.labels.begin() + start,
            bn,
            y.begin()
        );

        batchesQueued.push_back(std::move(batch));
        batchLabelsQueued.push_back(std::move(y));
    }
}

void Model::adam(size_t patience)
{
    float bestVal = std::numeric_limits<float>::infinity(); // Any batch will always have better than infinite loss
    size_t epochsSinceImprovement = 0;
    int epoch = 0;

    while (epochsSinceImprovement < patience) {
        reset_batches();
        float totalTrainLossSum = 0.0f;
        size_t trainSeen = 0;

        for (size_t batch = 0; batch < batchesQueued.size(); ++batch)
        {
            if (batch % 100 == 0)
                std::cout << "Epoch progress: " << batch << "/" << batchesQueued.size() << " batches" << std::endl;

            zero_gradients();

            const Tensor4F& X = batchesQueued[batch];
            const std::vector<uint8_t>& Y = batchLabelsQueued[batch];
            float batchMeanLoss = train_batch(X, Y);
            totalTrainLossSum += batchMeanLoss * float(X.n);
            trainSeen += size_t(X.n);
        }

        const float avgValLoss = validation_loss();
        const float epsilon = 1e-8;

        if (avgValLoss + epsilon < bestVal)
        {
            bestVal = avgValLoss;
            epochsSinceImprovement = 0;
            save_best_params();
        }
        else 
        {
            ++epochsSinceImprovement;
        }

        ++epoch;
        std::cout << "Epoch: " << epoch << std::endl;;

        const float avgTrainLoss = totalTrainLossSum / float(trainSeen);
        std::cout << "Avg Training loss: " << avgTrainLoss << std::endl;
        std::cout << "Avg Validation loss: " << avgValLoss << std::endl;
    }

    load_best_params();
}

float Model::train_batch(const Tensor4F& X, const std::vector<uint8_t>& labels)
{
    // Cache batch unit/activation values
    forward(X);

    // Cache grads through backprop
    const float loss = backward(labels);

    // Update params using adam
    ++timestep;

    // First conv layer
    adam_update(W1, mW1, vW1, dW1, trainstep, rho1, rho2, epsilon, timestep);
    adam_update(B1, mB1, vB1, dB1, trainstep, rho1, rho2, epsilon, timestep);

    // Second conv layer
    adam_update(W2, mW2, vW2, dW2, trainstep, rho1, rho2, epsilon, timestep);
    adam_update(B2, mB2, vB2, dB2, trainstep, rho1, rho2, epsilon, timestep);

    // FC layer
    adam_update(W3, mW3, vW3, dW3, trainstep, rho1, rho2, epsilon, timestep);
    adam_update(B3, mB3, vB3, dB3, trainstep, rho1, rho2, epsilon, timestep);

    return loss;
}

float Model::validation_loss()
{
    const size_t n = size_t(_validationImages.n);
    const size_t h = size_t(_validationImages.h);
    const size_t w = size_t(_validationImages.w);
    const size_t imgSize = h * w;
    constexpr size_t batchSize = 32;

    float totalLossSum = 0.0f;
    size_t seen = 0;

    ForwardCache valCache;

    // Go through the batch and accumulate loss
    for (size_t start = 0; start < n; start += batchSize)
    {
        const size_t bn = std::min(batchSize, n - start);

        // Build input batch tensor X with (bn, 1, h, w) (h,w = 28)
        Tensor4F X(uint32_t(bn), 1u, uint32_t(_validationImages.h), uint32_t(_validationImages.w));

        const size_t srcOffset = start * imgSize;
        const size_t countFloats = bn * imgSize;

        std::copy_n(
            _validationImages.imageData.begin() + srcOffset,
            countFloats,
            X.tensorData.begin()
        );

        // Forward pass into local cache
        valCache.X0 = X;

        valCache.Z1 = conv2d_forward(valCache.X0, W1, B1, 1, 1);
        valCache.A1 = relu_forward(valCache.Z1, valCache.reluMask1);
        valCache.P1 = maxpool_forward(valCache.A1, valCache.poolArgMax1);

        valCache.Z2 = conv2d_forward(valCache.P1, W2, B2, 1, 1);
        valCache.A2 = relu_forward(valCache.Z2, valCache.reluMask2);
        valCache.P2 = maxpool_forward(valCache.A2, valCache.poolArgMax2);

        valCache.F = valCache.P2.flatten();
        valCache.logits = fc_forward(valCache.F, W3, B3);

        // We dont care about the gradient for forward-only validation, so dlogits isn't used
        Tensor4F dlogitsUnused;

        // Create a view of labels for this batch
        std::vector<uint8_t> yBatch(bn);
        std::copy_n(_validationLabels.labels.begin() + start, bn, yBatch.begin());

        const float batchMeanLoss =
            softmax_ce_loss(valCache.logits, yBatch, valCache.probs, dlogitsUnused);

        // Convert mean loss back to sum so final averaging is correct across last partial batch (we average after
        // accumulating through entire validation set)
        totalLossSum += batchMeanLoss * float(bn);
        seen += bn;
    }

    return totalLossSum / float(seen);
}

void Model::zero_gradients()
{
    dW1.zero();
    dB1.zero();
    dW2.zero();
    dB2.zero();
    dW3.zero();
    dB3.zero();
}

void Model::forward(const Tensor4F& X)
{
    cache.X0 = X;

    cache.Z1 = conv2d_forward(cache.X0, W1, B1, 1, 1); // Pad of 1 for 3x3 kernal (same padding)
    cache.A1 = relu_forward(cache.Z1, cache.reluMask1);
    cache.P1 = maxpool_forward(cache.A1, cache.poolArgMax1);

    cache.Z2 = conv2d_forward(cache.P1, W2, B2, 1, 1); // Pad of 1 for 3x3 kernal (same padding)
    cache.A2 = relu_forward(cache.Z2, cache.reluMask2);
    cache.P2 = maxpool_forward(cache.A2, cache.poolArgMax2);

    cache.F = cache.P2.flatten(); // Flattens to (N, 784, 1, 1)
    cache.logits = fc_forward(cache.F, W3, B3); // FC outputs (N, 10, 1, 1)
}

float Model::backward(const std::vector<uint8_t>& labels)
{
    // Reset gradients per minibatch.
    dW1.zero(); 
    dB1.zero();
    dW2.zero(); 
    dB2.zero();
    dW3.zero(); 
    dB3.zero();

    // Now we do backprop, the relevent gradients will be accumulated into the coresponding data members.
    Tensor4F dlogits;
    float loss = softmax_ce_loss(cache.logits, labels, cache.probs, dlogits);

    Tensor4F dF;
    fc_backward(cache.F, W3, dlogits, dF, dW3, dB3);

    Tensor4F dP2 = dF.unflatten(16, 7, 7); // Undo flatten for convolution layers

    Tensor4F dA2 = maxpool_backward(dP2, cache.poolArgMax2, cache.A2.h, cache.A2.w);
    Tensor4F dZ2 = relu_backward(dA2, cache.reluMask2);

    Tensor4F dP1;
    conv2d_backward(cache.P1, W2, dZ2, 1, 1, dP1, dW2, dB2);

    Tensor4F dA1 = maxpool_backward(dP1, cache.poolArgMax1, cache.A1.h, cache.A1.w);
    Tensor4F dZ1 = relu_backward(dA1, cache.reluMask1);

    Tensor4F dX;
    conv2d_backward(cache.X0, W1, dZ1, 1, 1, dX, dW1, dB1);

    return loss;
}


void Model::save_best_params()
{
    bestW1.tensorData = W1.tensorData;
    bestB1.tensorData = B1.tensorData;

    bestW2.tensorData = W2.tensorData;
    bestB2.tensorData = B2.tensorData;

    bestW3.tensorData = W3.tensorData;
    bestB3.tensorData = B3.tensorData;
}

void Model::load_best_params()
{
    W1.tensorData = bestW1.tensorData;
    B1.tensorData = bestB1.tensorData;

    W2.tensorData = bestW2.tensorData;
    B2.tensorData = bestB2.tensorData;

    W3.tensorData = bestW3.tensorData;
    B3.tensorData = bestB3.tensorData;
}

void Model::test()
{
    const size_t n = size_t(_testImages.n);

    if (n == 0)
        throw std::runtime_error("test: test set is empty");

    if (_testLabels.labels.size() != n)
        throw std::runtime_error("test: labels size mismatch");

    const size_t h = size_t(_testImages.h);
    const size_t w = size_t(_testImages.w);
    const size_t imgSize = h * w;

    if (_testImages.imageData.size() != n * imgSize)
        throw std::runtime_error("test: imageData size mismatch");

    constexpr size_t batchSize = 32;

    float totalLossSum = 0.0f;
    size_t totalCorrect = 0;
    size_t seen = 0;

    ForwardCache tc;

    // Go through all test data in batches
    for (size_t start = 0; start < n; start += batchSize)
    {
        const size_t bn = std::min(batchSize, n - start);

        // Build test input batch tensor: (bn, 1, h, w) for h, w = 28
        Tensor4F X(uint32_t(bn), 1u, uint32_t(_testImages.h), uint32_t(_testImages.w));

        const size_t srcOffset = start * imgSize;
        const size_t countFloats = bn * imgSize;

        std::copy_n(
            _testImages.imageData.begin() + srcOffset,
            countFloats,
            X.tensorData.begin()
        );

        // Forward)
        tc.X0 = X;

        tc.Z1 = conv2d_forward(tc.X0, W1, B1, 1, 1);
        tc.A1 = relu_forward(tc.Z1, tc.reluMask1);
        tc.P1 = maxpool_forward(tc.A1, tc.poolArgMax1);

        tc.Z2 = conv2d_forward(tc.P1, W2, B2, 1, 1);
        tc.A2 = relu_forward(tc.Z2, tc.reluMask2);
        tc.P2 = maxpool_forward(tc.A2, tc.poolArgMax2);

        tc.F = tc.P2.flatten();
        tc.logits = fc_forward(tc.F, W3, B3);

        // Prepare labels for this batch
        std::vector<uint8_t> yBatch(bn);
        std::copy_n(_testLabels.labels.begin() + start, bn, yBatch.begin());

        // Feed logits into softmax and loss for classification
        Tensor4F dlogits_unused;
        const float batchMeanLoss =
            softmax_ce_loss(tc.logits, yBatch, tc.probs, dlogits_unused);

        totalLossSum += batchMeanLoss * float(bn);

        // Accuracy is argmax over logits
        for (uint32_t i = 0; i < uint32_t(bn); ++i)
        {
            uint32_t bestK = 0;
            float bestV = tc.logits(i, 0, 0, 0);

            for (uint32_t k = 1; k < tc.logits.c; ++k)
            {
                float v = tc.logits(i, k, 0, 0);
                if (v > bestV) { bestV = v; bestK = k; }
            }

            if (uint8_t(bestK) == yBatch[i])
                ++totalCorrect;
        }

        seen += bn;
    }

    const float avgLoss = totalLossSum / float(seen);
    const float accuracy = float(totalCorrect) / float(seen);
    const float errorRate = 1.0f - accuracy;

    std::cout << "Test loss: " << avgLoss << "\n";
    std::cout << "Test accuracy: " << accuracy << "\n";
    std::cout << "Test error: " << errorRate << "\n";
}