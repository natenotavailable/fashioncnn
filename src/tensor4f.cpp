#include "tensor4f.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

Tensor4F::Tensor4F(uint32_t n, uint32_t c, uint32_t h, uint32_t w)
    : n{n}
    , c{c}
    , h{h}
    , w{w}
    , tensorData(size_t(n) * c * h * w, 0.0f)
{}

float& Tensor4F::operator[](uint32_t nIndex, uint32_t cIndex, uint32_t hIndex, uint32_t wIndex)
{
    return tensorData[nIndex * (c * h * w) + cIndex * (h * w) + hIndex * w + wIndex];
}

const float& Tensor4F::operator[](uint32_t nIndex, uint32_t cIndex, uint32_t hIndex, uint32_t wIndex) const
{
    return tensorData[nIndex * (c * h * w) + cIndex * (h * w) + hIndex * w + wIndex];
}

void Tensor4F::operator+(const Tensor4F& other)
{
    for (size_t i = 0; i < tensorData.size(); ++i)
        tensorData[i] += other.tensorData[i];
}

void Tensor4F::operator-(const Tensor4F& other)
{
    for (size_t i = 0; i < tensorData.size(); ++i)
        tensorData[i] -= other.tensorData[i];
}

void Tensor4F::operator*(float scale)
{
    for (size_t i = 0; i < tensorData.size(); ++i)
        tensorData[i] *= scale;
}

float Tensor4F::sum() const
{
    float sum = 0.0f;
    for (float f : tensorData)
        sum += f;
    return sum;
}

size_t Tensor4F::size() const
{
    return tensorData.size();
}

void Tensor4F::resize(uint32_t nNew, uint32_t cNew, uint32_t hNew, uint32_t wNew)
{
    n = nNew;
    c = cNew;
    h = hNew;
    w = wNew;
    tensorData.assign(size_t(n) * c * h * w, 0.0f);
}

void Tensor4F::fill(float value)
{
    std::fill(tensorData.begin(), tensorData.end(), value);
}

void Tensor4F::zero()
{
    fill(0.0f);
}

void Tensor4F::uniform(float low, float high, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(low, high);

    for (auto& f : tensorData)
        f = distribution(rng);
}

void Tensor4F::normal(float mu, float sigma, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(mu, sigma);
    
    for (auto& f : tensorData)
        f = distribution(rng);
}

Tensor4F Tensor4F::flatten() const
{
    Tensor4F flattened(n, c * h * w, 1, 1);
    const size_t valuesPerN = size_t(c) * h * w;

    for (uint32_t num = 0; num < n; ++num)
    {
        const size_t srcOffset = size_t(num) * valuesPerN;
        const size_t dstOffset = size_t(num) * valuesPerN;
        std::copy(tensorData.begin() + srcOffset, tensorData.begin() + srcOffset + valuesPerN,
                  flattened.tensorData.begin() + dstOffset);
    }

    return flattened;
}

Tensor4F Tensor4F::unflatten(uint32_t cNew, uint32_t hNew, uint32_t wNew) const
{
    Tensor4F unflattened(n, cNew, hNew, wNew);
    const size_t valuesPerN = size_t(cNew) * hNew * wNew;

    for (uint32_t num = 0; num < n; ++num)
    {
        const size_t srcOffset = size_t(num) * valuesPerN;
        const size_t dstOffset = size_t(num) * valuesPerN;
        std::copy(tensorData.begin() + srcOffset, tensorData.begin() + srcOffset + valuesPerN,
                  unflattened.tensorData.begin() + dstOffset);
    }

    return unflattened;
}

static Tensor4F relu_forward(const Tensor4F& X, Tensor4F& reluMask)
{
    Tensor4F Y(X.n, X.c, X.h, X.w);
    reluMask.resize(X.n, X.c, X.h, X.w);

    // Apply ReLU to every element, saving activations > 0 in mask for backprop.
    for (size_t i = 0; i < X.tensorData.size(); ++i)
    {
        float value = X.tensorData[i];
        if (value > 0.0f)
        {
            Y.tensorData[i] = value;
            reluMask.tensorData[i] = 1.0f;
        }
        else {
            Y.tensorData[i] = 0.0f;
            reluMask.tensorData[i] = 0.0f;
        }
    }

    return Y;
}

static Tensor4F relu_backward(const Tensor4F& dY, const Tensor4F& reluMask)
{
    Tensor4F dX(dY.n, dY.c, dY.h, dY.w);

    for (size_t i = 0; i < dY.tensorData.size(); ++i)
        dX.tensorData[i] = dY.tensorData[i] * reluMask.tensorData[i];

    return dX;
}

static Tensor4F maxpool_forward(const Tensor4F& X, Tensor4F& poolArgMax)
{
    // Stride of 2 reduces size by half.
    const uint32_t outH = X.h / 2;
    const uint32_t outW = X.w / 2;

    Tensor4F Y(X.n, X.c, outH, outW);
    poolArgMax.resize(X.n, X.c, outH, outW);

    for (uint32_t N = 0; N < X.n; ++N)
        for (uint32_t C = 0; C < X.c; ++C)
            for (uint32_t yOut = 0; yOut < outH; ++yOut)
                for (uint32_t xOut = 0; xOut < outW; ++xOut)
    {
        const uint32_t yIn = yOut * 2;
        const uint32_t xIn = xOut * 2;

        float a = X[N, C, yIn, xIn];
        float b = X[N, C, yIn, xIn + 1];
        float c = X[N, C, yIn + 1, xIn];
        float d = X[N, C, yIn + 1, xIn + 1];

        float m = a; uint32_t arg = 0;
        if (b > m) { m = b; arg = 1; }
        if (c > m) { m = c; arg = 2; }
        if (d > m) { m = d; arg = 3; }

        Y[N, C, yOut, xOut] = m;
        poolArgMax[N, C, yOut, xOut] = float(arg);
    }

    return Y;
}

static Tensor4F maxpool_backward(const Tensor4F& dY, const Tensor4F& poolArgMax, uint32_t inH, uint32_t inW)
{
    Tensor4F dX(dY.n, dY.c, inH, inW);
    dX.zero();

    for (uint32_t N = 0; N < dY.n; ++N)
        for (uint32_t C = 0; C < dY.c; ++C)
            for (uint32_t yOut = 0; yOut < dY.h; ++yOut)
                for (uint32_t xOut = 0; xOut < dY.w; ++xOut)
    {
        uint32_t arg = static_cast<uint32_t>(poolArgMax[N, C, yOut, xOut]); // In range [0, 3]
        uint32_t yIn = yOut * 2;
        uint32_t xIn = xOut * 2;

        // Since I use 2x2 pooling, we must calculate the index offset from the pooled value.
        uint32_t dy = (arg / 2);
        uint32_t dx = (arg % 2);

        dX[N, C, yIn + dy, xIn + dx] += dY[N, C, yOut, xOut];
    }

    return dX;
}

static Tensor4F conv2d_forward(
    const Tensor4F& X,
    const Tensor4F& W,
    const Tensor4F& B, // I use a tensor4, but I only use the channel to store bias. Just think of it as a vector.
    uint32_t padH,
    uint32_t padW
)
{
    const uint32_t outC = W.n; // Although n is used here, it is actually representing the out channel for W.
    const uint32_t inC = W.c; // Likewise, c represents the in channel for W.
    const uint32_t kH = W.h;
    const uint32_t kW = W.w;

    const uint32_t outH = X.h + 2 * padH - kH + 1;
    const uint32_t outW = X.w + 2 * padW - kW + 1;

    Tensor4F Y(X.n, outC, outH, outW);

    for (uint32_t N = 0; N < X.n; ++N)
        for (uint32_t cOI = 0; cOI < outC; ++cOI)
            for (uint32_t yOut = 0; yOut < outH; ++yOut)
                for (uint32_t xOut = 0; xOut < outW; ++xOut)
    {
        float acc = B[0, cOI, 0, 0];

        for (uint32_t IC = 0; IC < inC; ++IC)
            for (uint32_t ky = 0; ky < kH; ++ky)
                for (uint32_t kx = 0; kx < kW; ++kx)
        {
            // Map output + kernal coordinates onto input
            int32_t yIn = int32_t(yOut) + int32_t(ky) - int32_t(padH);
            int32_t xIn = int32_t(xOut) + int32_t(kx) - int32_t(padW);

            if (yIn < 0 || xIn < 0 || yIn >= int32_t(X.h) || xIn >= int32_t(X.w)) continue;
            acc += X[N, IC, uint32_t(yIn), uint32_t(xIn)] * W[cOI, IC, ky, kx];
        }

        Y[N, cOI, yOut, xOut] = acc;
    }

    return Y;
}

static void conv2d_backward(
    const Tensor4F& x,
    const Tensor4F& w, 
    const Tensor4F& dY, 
    uint32_t padH, 
    uint32_t padW, 
    Tensor4F& dX, 
    Tensor4F& dW, 
    Tensor4F& dB
)
{
    const uint32_t outC = w.n;
    const uint32_t inC = w.c;
    const uint32_t kH = w.h;
    const uint32_t kW = w.w;

    // Resize and clear output tensors.
    dX.resize(x.n, x.c, x.h, x.w); 
    dX.zero();
    dW.resize(w.n, w.c, w.h, w.w); 
    dW.zero();
    dB.resize(1, outC, 1, 1);      
    dB.zero();

    // dB is sum over n, yOut, and xOut
    for (uint32_t N = 0; N < dY.n; ++N)
        for (uint32_t cOI = 0; cOI < dY.c; ++cOI)
            for (uint32_t yOut = 0; yOut < dY.h; ++yOut)
                for (uint32_t xOut = 0; xOut < dY.w; ++xOut)
    {
        dB[0, cOI, 0, 0] += dY[N, cOI, yOut, xOut];
    }

    // dW is sum over n, yOut, and xOut like bias, but also over all values affected by kernal.
    for (uint32_t OC = 0; OC < outC; ++OC)
        for (uint32_t IC = 0; IC < inC; ++IC)
            for (uint32_t ky = 0; ky < kH; ++ky)
                for (uint32_t kx = 0; kx < kW; ++kx)
    {
        float acc = 0.0f;
        for (uint32_t N = 0; N < x.n; ++N)
            for (uint32_t oy = 0; oy < dY.h; ++oy)
                for (uint32_t ox = 0; ox < dY.w; ++ox)
        {
            int32_t iy = int32_t(oy) + int32_t(ky) - int32_t(padH);
            int32_t ix = int32_t(ox) + int32_t(kx) - int32_t(padW);
            if (iy < 0 || ix < 0 || iy >= int32_t(x.h) || ix >= int32_t(x.w)) continue;

            acc += x[N, IC, uint32_t(iy), uint32_t(ix)] * dY[N, OC, oy, ox];
        }
        dW[OC, IC, ky, kx] = acc;
    }

    // dX is basically just a convolution of the gradient with backward indexing.
    for (uint32_t N = 0; N < x.n; ++N)
        for (uint32_t IC = 0; IC < inC; ++IC)
            for (uint32_t iy = 0; iy < x.h; ++iy)
                for (uint32_t ix = 0; ix < x.w; ++ix)
    {
        float acc = 0.0f;
        for (uint32_t OC = 0; OC < outC; ++OC)
            for (uint32_t ky = 0; ky < kH; ++ky)
                for (uint32_t kx = 0; kx < kW; ++kx)
        {
            int32_t oy = int32_t(iy) - int32_t(ky) + int32_t(padH);
            int32_t ox = int32_t(ix) - int32_t(kx) + int32_t(padW);
            if (oy < 0 || ox < 0 || oy >= int32_t(dY.h) || ox >= int32_t(dY.w)) continue;

            acc += w[OC, IC, ky, kx] * dY[N, OC, uint32_t(oy), uint32_t(ox)];
        }
        dX[N, IC, iy, ix] += acc;
    }
}