#include "utils.hpp"

const char *coco_names[] = {"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

void draw(cv::Mat& img,
          const std::vector<Face>& faces) {

    const int thickness = 2;
    const cv::Scalar bbox_color = {  0, 255,   0};
    const cv::Scalar text_color = {255, 255, 255};
    const std::vector<cv::Scalar> landmarks_color = {
        {255,   0,   0}, // right eye
        {  0,   0, 255}, // left eye
        {  0, 255,   0}, // nose
        {255,   0, 255}, // mouth right
        {  0, 255, 255}  // mouth left
    };

    for (auto i = 0; i < faces.size(); ++i) {
        // draw bbox
        cv::rectangle(img,
                      cv::Rect(faces[i].bbox_tlwh),
                      bbox_color,
                      thickness);
        // put score by the corner of bbox
        std::string str_score = std::to_string(faces[i].score);
        if (str_score.size() > 6) {
            str_score.erase(6);
        }
        std::string name = coco_names[faces[i].class_index];
        std::string text = name + " " + str_score;
        
        cv::putText(img,
                    text,
                    cv::Point(faces[i].bbox_tlwh.x, faces[i].bbox_tlwh.y + 12),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5, // Font scale
                    text_color);
    }
}
