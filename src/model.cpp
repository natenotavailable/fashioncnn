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

using std::string;
using std::vector;
using std::mt19937;

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

    W3.resize(10, 1, 1, 784);
    B3.resize(1, 10, 1, 1);

    // Initialize weights & biases.
    W1.normal(0.0f, 0.05f, 0);
    W2.normal(0.0f, 0.05f, 1);
    W3.normal(0.0f, 0.05f, 2);

    B1.zero();
    B2.zero();
    B3.zero();

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

    _trainingLabels.labels = std::move(validationLabels);
    _trainingLabels.n = n - cutoff;
}

void Model::reset_batches()
{
    // Shuffle training data.
    shuffle_data(_trainingImages, _trainingLabels, rng);

    const size_t n = size_t(_trainingImages.n);
    const size_t h = size_t(_trainingImages.h);
    const size_t w = size_t(_trainingImages.w);
    const size_t imgSize = h * w;

    // Batch into Tensors (batch size 32) + parallel labels.
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
            throw std::runtime_error("reset_batches: Tensor4F storage size unexpected");

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

    while (epochsSinceImprovement < patience) {
        reset_batches();

        for (size_t batch = 0; batch < batchesQueued.size(); ++batch)
        {
            zero_gradients();

            const Tensor4F& X = batchesQueued[batch];
            const std::vector<uint8_t>& Y = batchLabelsQueued[batch];

            train_batch(X, Y);
        }

        const float val = validation_loss();
        const float epsilon = 1e-8;

        if (val + epsilon < bestVal)
        {
            bestVal = val;
            epochsSinceImprovement = 0;

            // Save params;
        }
        else 
        {
            ++epochsSinceImprovement;
        }
    }

    // Load checkpoint;
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