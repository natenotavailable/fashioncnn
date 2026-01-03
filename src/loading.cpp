#include "loading.h"

#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

using std::ifstream;
using std::vector;
using std::string;
using std::runtime_error;

static uint32_t read_u32_from_stream(ifstream& stream)
{
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0] << 24) | (uint32_t(bytes[1] << 16)) | (uint32_t(bytes[2]) << 8) | (uint32_t(bytes[3])));
}

static IdxImages load_idx3_images(const string& path)
{
    ifstream fileStream(path, std::ios::binary);
    if (!fileStream)
        throw runtime_error("Failed to open file: " + path);
    
    (void)read_u32_from_stream(fileStream); // Discard the magic number since we know the format.

    IdxImages out;
    out.n = read_u32_from_stream(fileStream);
    out.h = read_u32_from_stream(fileStream);
    out.w = read_u32_from_stream(fileStream);

    const size_t count = size_t(out.n) * out.h * out.w;
    vector<uint8_t> buf(count);
    fileStream.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(count));

    if (!fileStream) 
        throw runtime_error("Failed to read image bytes at: " + path);

    out.imageData.resize(count);
    for (size_t i = 0; i < count; ++i) 
        out.imageData[i] = float(buf[i]) / 255.0f;

    return out;
}

static IdxLabels load_idx1_labels(const string& path) {
    ifstream fileStream(path, std::ios::binary);
    if (!fileStream) 
        throw runtime_error("Failed to open file: " + path);

    (void)read_u32_from_stream(fileStream); // Discard the magic number since we know the format.

    IdxLabels out;
    out.n = read_u32_from_stream(fileStream);

    out.labels.resize(out.n);
    fileStream.read(reinterpret_cast<char*>(out.labels.data()), static_cast<std::streamsize>(out.n));
    if (!fileStream) 
        throw runtime_error("Failed to read label bytes at: " + path);

    return out;
}