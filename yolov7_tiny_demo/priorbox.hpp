#ifndef PRIORBOX_HPP
#define PRIORBOX_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "face_def.hpp"

class PriorBox {
private:
    const std::vector<std::vector<float>> anchor = {
        {{10.0f,  13.0f, 16.0f, 30.0f, 33.0f, 23.0f}},
        {{30.0f,  61.0f, 62.0f, 45.0f, 59.0f, 119.0f}},
        {{116.0f,  90.0f, 156.0f, 198.0f, 373.0f, 326.0f}}
    };
    const std::vector<int> steps = { 8, 16, 32 };
    const int num_class = 80;
    const int coords = 4;
    const int num_box = 3;

    int in_w;
    int in_h;
    int model_w;
    int model_h;

    std::vector<int> feature_map_sizes = { 80, 40, 20 };
    std::vector<Box> priors;
private:
    std::vector<Box> generate_priors();
public:
    PriorBox(const cv::Size& input_shape,
             const cv::Size& output_shape);
    ~PriorBox();
    std::vector<Face> decode(const cv::Mat& pred1, const cv::Mat& pred2, const cv::Mat& pred3, const float ignore_score=0.3);
};

#endif
