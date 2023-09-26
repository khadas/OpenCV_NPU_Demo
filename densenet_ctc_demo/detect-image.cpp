#include <vector>
#include <string>
#include <iostream>

#include <time.h>

#include "opencv2/opencv.hpp"

const char *names[] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", ",", ".", "[", "]", "{", "}", "|", "~", "@", "#", "$", "%", "^", "&", "(", ")", "<", ">", "?", ":", ";", "a", "b", "c", "d", "e", "f", "g", "h", "i", "g", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

int main(int argc, char* argv[]) {
    if (argc != 3 && argc !=4) {
        std::cout << "Usage: " << argv[0] << " <image_file_name> <net_file_name> [<visualization_flag> default false] \n";
        return -1;
    }

    // Build blob
    cv::Mat img = cv::imread(argv[1], 0);
    cv::Mat temp_img(280, 32, CV_8UC1), normalized_img;
    if (img.empty()) {
        std::cerr << "Cannot load the image file " << argv[1] << ".\n";
        return -1;
    }
    cv::resize(img, temp_img, cv::Size(280, 32));
    temp_img.convertTo(normalized_img, CV_32FC1, 1.0 / 255.0);
    cv::Mat blob = cv::dnn::blobFromImage(normalized_img);

    // Load .onnx model using OpenCV's DNN module
    cv::dnn::Net net = cv::dnn::readNet(argv[2]);

    /*NPU*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_TIMVX);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);

    /* CPU */
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Inference hyperparameters
    float conf_thresh = 0.25;
    // Result
    bool vis = false;
    if (argc==4 && std::string(argv[3]) == "true") {
        vis = true;
    }

    // Forward
    std::vector<cv::String> output_names = { "out" };
    std::vector<cv::Mat> output_blobs;
    net.setInput(blob);
    clock_t start = clock();
    net.forward(output_blobs, output_names);
    clock_t end = clock();
    std::cout << double(end - start) / CLOCKS_PER_SEC << std::endl;
    
    int i, j, max_index, stride;
    int box = 35;
    int class_num = 88;
    float max_conf, conf;
    float* predictions = (float*)(output_blobs[0].data);
    int result_len = 0;
    char result[35] = {0};
    
    int last_index = class_num - 1;
    for (i = 0; i < box; ++i)
    {
    	max_conf = 0;
    	max_index = class_num - 1;
    	for (j = 0; j < class_num; ++j)
    	{
    		conf = predictions[i * class_num + j];
    		if (conf > conf_thresh && conf > max_conf)
    		{
    			max_conf = conf;
    			max_index = j;
    		}
    	}
    	if (max_index != class_num - 1 && max_index != last_index)
    	{
    		result[result_len] = *names[max_index];
    		result_len++;
    	}
    	last_index = max_index;
    }
    
    //printf("%d\n", result_len);
    printf("%s\n", result);

    return 0;
}
