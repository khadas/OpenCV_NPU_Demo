#include <vector>
#include <string>
#include <iostream>

#include "priorbox.hpp"
#include "utils.hpp"
#include <time.h>

#include "opencv2/opencv.hpp"

const char *class_names[] = {"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int main(int argc, char* argv[]) {
    if (argc != 3 && argc !=4) {
        std::cout << "Usage: " << argv[0] << " <image_file_name> <net_file_name> [<visualization_flag> default false] \n";
        return -1;
    }

    // Build blob
    cv::Mat orig_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    int orig_img_width = orig_img.cols;
    int orig_img_height = orig_img.rows;
    cv::Mat img;
    if (orig_img.empty()) {
        std::cerr << "Cannot load the image file " << argv[1] << ".\n";
        return -1;
    }
    
    int x_padding, y_padding;
    if (orig_img_width >= orig_img_height)
    {
    	x_padding = 0;
    	y_padding = orig_img_width - orig_img_height;
    }
    else
    {
    	x_padding = orig_img_height - orig_img_width;
    	y_padding = 0;
    }
    
    cv::copyMakeBorder(orig_img, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::Size img_shape = img.size();
    cv::resize(img, img, cv::Size(640, 640));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    cv::Mat blob = cv::dnn::blobFromImage(img);

    // Load .onnx model using OpenCV's DNN module
    cv::dnn::Net net = cv::dnn::readNet(argv[2]);

    /*NPU*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_TIMVX);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);

    /* CPU */
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Inference hyperparameters
    float conf_thresh = 0.3;
    float nms_thresh = 0.4;
    int keep_top_k = 750;
    // Result
    bool vis = false;
    if (argc==4 && std::string(argv[3]) == "true") {
        vis = true;
    }
    std::string save_fpath = "./result.jpg";

    // Forward
    std::vector<cv::String> output_names = { "output", "266", "267" };
    std::vector<cv::Mat> output_blobs;
    net.setInput(blob);
    clock_t start = clock();
    net.forward(output_blobs, output_names);
    clock_t end = clock();
    std::cout << double(end - start) / CLOCKS_PER_SEC << std::endl;

    // Decode bboxes, landmarks and scores
    PriorBox pb(img_shape, cv::Size(640, 640));
    std::vector<Face> dets = pb.decode(output_blobs[0], output_blobs[1], output_blobs[2], conf_thresh);
    for (int i = 0; i < dets.size(); ++i) {
    	dets[i].bbox_tlwh.x = dets[i].bbox_tlwh.x * img_shape.width;
    	dets[i].bbox_tlwh.y = dets[i].bbox_tlwh.y * img_shape.height;
    	dets[i].bbox_tlwh.width = dets[i].bbox_tlwh.width * img_shape.width;
    	dets[i].bbox_tlwh.height = dets[i].bbox_tlwh.height * img_shape.height;
    }

    // NMS
    std::vector<Face> nms_dets;
    if (dets.size() > 1) {
        std::vector<cv::Rect> face_boxes;
        std::vector<float> face_scores;
        for (auto d: dets) {
            face_boxes.push_back(d.bbox_tlwh);
            face_scores.push_back(d.score);
        }
        std::vector<int> keep_idx;
        cv::dnn::NMSBoxes(face_boxes, face_scores, conf_thresh, nms_thresh, keep_idx, 1.f, keep_top_k);
        for (size_t i = 0; i < keep_idx.size(); i++) {
            size_t idx = keep_idx[i];
            nms_dets.push_back(dets[idx]);
        }
    }
    else if (dets.size() < 1) {
        std::cout << "No object found." << std::endl;
        return 1;
    }
    std::cout << "Detection results: " << nms_dets.size() << " objects found." << std::endl;
    for (auto i = 0; i < nms_dets.size(); ++i) {
        Box bbox = nms_dets[i].bbox_tlwh;
        float score = nms_dets[i].score;
        std::cout << "[" << bbox.x << ", " << bbox.y << "] [" << bbox.x + bbox.width << ", " << bbox.y + bbox.height << "] " << class_names[nms_dets[i].class_index] << " " << score << std::endl;
    }

    // Draw and display
    draw(orig_img, nms_dets);
    if (vis) {
        cv::String title = cv::String("Detection Results on") + cv::String(argv[1]);
        cv::imshow(title, orig_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    cv::imwrite(save_fpath, orig_img);

    return 0;
}
