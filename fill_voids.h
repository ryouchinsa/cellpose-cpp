#ifndef FILL_VOIDS_H_
#define FILL_VOIDS_H_

# include <torch/extension.h>
# include <iostream>
# include <vector>
#include "libdivide.h"

#define BACKGROUND 0
#define VISITED_BACKGROUND 1
#define FOREGROUND 2

torch::Tensor fill_voids(torch::Tensor labels);

#endif






















