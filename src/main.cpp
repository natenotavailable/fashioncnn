#include "model.h"

int main()
{
    Model m;
    m.load_data(
        "t10k-images-idx3-ubyte", 
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte", 
        "train-labels-idx1-ubyte"
    );
    m.adam(5);
    m.test();
}