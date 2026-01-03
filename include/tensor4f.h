#ifndef TENSOR4F_H
#define TENSOR4F_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct Tensor4F
{
    uint32_t n = 0, c = 0, h = 0, w = 0;
    std::vector<float> tensorData;

    Tensor4F() = default;
    Tensor4F(uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    float& operator[](uint32_t nIndex, uint32_t cIndex, uint32_t hIndex, uint32_t wIndex);
    const float& operator[](uint32_t nIndex, uint32_t cIndex, uint32_t hIndex, uint32_t wIndex) const;

    void operator+(const Tensor4F& other);
    void operator-(const Tensor4F& other);
    void operator*(float scale);
        
    float sum() const;

    size_t size() const;
    void resize(uint32_t nNew, uint32_t cNew, uint32_t hNew, uint32_t wNew);
    void fill(float value);
    void zero();

    void uniform(float low, float high, uint32_t seed = 0);
    void normal(float mu, float sigma, uint32_t seed = 0);

    Tensor4F flatten() const;
    Tensor4F unflatten(uint32_t cNew, uint32_t hNew, uint32_t wNew) const;
};

Tensor4F relu_forward(const Tensor4F& X, Tensor4F& reluMask);
Tensor4F relu_backward(const Tensor4F& dY, const Tensor4F& reluMask);

Tensor4F maxpool_forward(const Tensor4F& X, Tensor4F& poolArgMax);
Tensor4F maxpool_backward(const Tensor4F& dY, const Tensor4F& poolArgMax, uint32_t inH, uint32_t inW);

Tensor4F conv2d_forward(
    const Tensor4F& X,
    const Tensor4F& W,
    const Tensor4F& B,
    uint32_t padH,
    uint32_t padW
);

void conv2d_backward(
    const Tensor4F& X,
    const Tensor4F& W, 
    const Tensor4F& dY, 
    uint32_t padH, 
    uint32_t padW, 
    Tensor4F& dX, 
    Tensor4F& dW, 
    Tensor4F& dB
);

Tensor4F fc_forward(const Tensor4F& X, const Tensor4F& W, const Tensor4F& B);
void fc_backward(
    const Tensor4F& X,
    const Tensor4F& W,
    const Tensor4F& dY,
    Tensor4F& dX,
    Tensor4F& dW,
    Tensor4F& dB
);

float softmax_ce_loss(const Tensor4F& logits, const std::vector<uint8_t>& labels, Tensor4F& probs, Tensor4F& dlogits);


#endif