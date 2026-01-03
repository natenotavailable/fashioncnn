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
    std::mt19937 rng;

    IdxImages _testImages;
    IdxLabels _testLabels;
    IdxImages _trainingImages;
    IdxLabels _trainingLabels;
    IdxImages _validationImages;
    IdxLabels _validationLabels;

    std::vector<Tensor4F> batchesQueued;
    std::vector<std::vector<uint8_t>> batchLabelsQueued;

    void load_data(
        std::string imagePathTest, 
        std::string labelPathTest, 
        std::string imagePathTraining,
        std::string labelPathTraining
    );
    void reset_batches();
    void adam(size_t patience);
    void test();
};

static void shuffle_data(IdxImages& images, IdxLabels& labels, std::mt19937& rng);

#endif