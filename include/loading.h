#ifndef LOADING_H
#define LOADING_H

#include <cstdint>
#include <fstream>
#include <vector>
#include <string>

struct IdxImages
{
    uint32_t n = 0, h = 0, w = 0;
    std::vector<float> imageData;
};

struct IdxLabels
{
    uint32_t n = 0;
    std::vector<uint8_t> labels;
};

static uint32_t read_u32_from_stream(std::ifstream& stream);
IdxImages load_idx3_images(const std::string& path);
IdxLabels load_idx1_labels(const std::string& path);

#endif