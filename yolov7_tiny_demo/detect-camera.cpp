#include <vector>
#include <string>
#include <iostream>

#include "priorbox.hpp"
#include "utils.hpp"

#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]) {

    if(argc != 3)
    {
        printf("Usage: %s <camera index>\n", argv[0]);
        return -1;
    }

    // Load .onnx model using OpenCV's DNN module
    cv::dnn::Net net = cv::dnn::readNet(argv[2]);

    /*NPU*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_TIMVX);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);

    /* CPU */
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap;
    cv::Mat im, blob;
    cv::TickMeter cvtm;
    cv::Size img_shape;
    std::vector<cv::String> output_names = { "output", "266", "267" };
    std::vector<cv::Mat> output_blobs;

    cv::String title = cv::String("Detection Results on") + cv::String(argv[1]);

    // Inference hyperparameters
    float conf_thresh = 0.3;
    float nms_thresh = 0.4;
    int keep_top_k = 750;
    // Result
    bool vis = false;

    if( isdigit(argv[1][0]))
    {
        cap.open(argv[1][0]-'0');
        if(! cap.isOpened())
        {
            std::cerr << "Cannot open the camera." << std::endl;
            return 0;
        }
    }

    cap >> im;
    
    cv::Mat img;
    int orig_img_width = im.cols;
    int orig_img_height = im.rows;
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
    cv::copyMakeBorder(im, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    img_shape = img.size();
    cv::resize(img, img, cv::Size(640, 640));
    
    PriorBox pb(img_shape, cv::Size(640, 640));
    std::vector<Face> dets;

    if( cap.isOpened())
    {
        while(true)
        {
            cap >> im;
            cvtm.start();
            
            cv::copyMakeBorder(im, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
            cv::resize(img, img, cv::Size(640, 640));
            img.convertTo(img, CV_32FC3, 1.0 / 255.0);

            // Build blob
            blob = cv::dnn::blobFromImage(img, 1.0,  cv::Size(), cv::Scalar());

            // Forward
            net.setInput(blob);
            net.forward(output_blobs, output_names);

            // Decode bboxes, landmarks and scores
            dets = pb.decode(output_blobs[0], output_blobs[1], output_blobs[2], conf_thresh);
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
                //std::cout << "No object found." << std::endl;
            }

            cvtm.stop();

            std::string timeLabel = cv::format("Inference time: %.2f ms", cvtm.getTimeMilli());
            cv::putText(im, timeLabel, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            draw(im, nms_dets);

            cvtm.reset();
            cv::imshow(title, im);
            if((cv::waitKey(1)& 0xFF) == 27)
                break;
        }
    }

    return 0;
}
