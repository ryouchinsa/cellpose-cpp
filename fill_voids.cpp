#include "fill_voids.h"

inline void push_stack(
  torch::Tensor labels,  const size_t loc,
  std::stack<size_t> &stack, bool &placed
) {
  if (labels[loc].item<int>() == 0) {
    if (!placed) {
      stack.push(loc);
    }
    placed = true;
  }
  else {
    placed = false;
  }  
}

inline void add_neighbors(
  torch::Tensor visited, std::stack<size_t> &stack,
  const size_t sx, const size_t sy,
  const size_t cur, const size_t y,
  bool &yplus, bool &yminus
) {

  if (y > 0) {
    if (visited[cur-sx].item<int>()) {
      yminus = yminus || (visited[cur-sx].item<int>() == FOREGROUND);
    }
    else if (yminus) {
      stack.push( cur - sx );
      yminus = false;
    }
  }

  if (y < sy - 1) {
    if (visited[cur+sx].item<int>()) {
      yplus = yplus || (visited[cur+sx].item<int>() == FOREGROUND);
    }
    else if (yplus) {
      stack.push( cur + sx );
      yplus = false;
    }
  }
}

void initialize_stack(
    torch::Tensor labels, 
    const size_t sx, const size_t sy,
    std::stack<size_t> &stack
  ) {

  bool placed_front = false;
  bool placed_back = false;

  size_t loc;
  for (size_t x = 0; x < sx; x++) {
    loc = x;
    push_stack(labels, loc, stack, placed_front);
    
    loc = x + sx * (sy - 1);
    push_stack(labels, loc, stack, placed_back);
  }

  placed_front = false;
  placed_back = false;

  for (size_t y = 0; y < sy; y++) {
    loc = sx * y;
    push_stack(labels, loc, stack, placed_front);
    
    loc = (sx - 1) + sx * y;
    push_stack(labels, loc, stack, placed_back);
  }
}

torch::Tensor fill_voids(torch::Tensor labels){
  size_t sy = labels.size(0);
  size_t sx = labels.size(1);
  const size_t voxels = sx * sy;
  labels = labels.reshape({static_cast<long long>(voxels)});
  for (size_t i = 0; i < voxels; i++) {
    labels[i] = (labels[i] != 0) * 2;
  }

  const libdivide::divider<size_t> fast_sx(sx); 

  std::stack<size_t> stack; 
  initialize_stack(labels, sx, sy, stack);

  while (!stack.empty()) {
    size_t loc = stack.top();
    stack.pop();

    if (labels[loc].item<int>()) {
      continue;
    }

    size_t y = loc / fast_sx;
    size_t startx = y * sx;

    bool yplus = true;
    bool yminus = true;

    for (size_t cur = loc; cur < startx + sx; cur++) {
      if (labels[cur].item<int>()) {
        break;
      }
      labels[cur] = VISITED_BACKGROUND;
      add_neighbors(
        labels, stack,
        sx, sy, 
        cur, y,
        yplus, yminus
      );
    }

    yplus = true;
    yminus = true;

    for (int64_t cur = static_cast<int64_t>(loc) - 1; cur >= static_cast<int64_t>(startx); cur--) {
      if (labels[cur].item<int>()) {
        break;
      }
      labels[cur] = VISITED_BACKGROUND;
      add_neighbors(
        labels, stack,
        sx, sy,
        cur, y,
        yplus, yminus
      );
    }    
  }

  for (size_t i = 0; i < voxels; i++) {
    labels[i] = (labels[i].item<int>() != VISITED_BACKGROUND);
  }
  labels = labels.reshape({static_cast<long long>(sy), static_cast<long long>(sx)});

  return labels;
}
























