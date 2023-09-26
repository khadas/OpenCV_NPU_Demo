#include "priorbox.hpp"

PriorBox::PriorBox(const cv::Size& input_shape,
                   const cv::Size& model_shape) {
    // initialize
    in_w = input_shape.width;
    in_h = input_shape.height;
    model_w = model_shape.width;
    model_h = model_shape.height;
}

PriorBox::~PriorBox() {}

static float logistic_activate(float x){return 1./(1. + exp(-x));}

std::vector<Face> PriorBox::decode(const cv::Mat& pred1,
                                   const cv::Mat& pred2,
                                   const cv::Mat& pred3,
                                   const float ignore_score) {
    std::vector<Face> dets; // num * [x1, y1, x2, y2]

    float* output[3] = {NULL};
    output[0] = (float*)(pred1.data);
    output[1] = (float*)(pred2.data);
    output[2] = (float*)(pred3.data);
    int bb_size = num_class + coords + 1;
    int channel_num = bb_size * num_box;
    int total_num = 0;
    float object_conf;
    
    for (int n = 0; n < 3; ++n)
    {
    	int feature_map_length = feature_map_sizes[n] * feature_map_sizes[n];
    	for (int i =  0; i < feature_map_sizes[n]; ++i)
    	{
    		for (int j = 0; j < feature_map_sizes[n]; ++j)
    		{
    			for (int k = 0; k < num_box; ++k)
    			{
    				int loc = i * feature_map_sizes[n] + j;
    				object_conf = logistic_activate(output[n][loc + (k * bb_size + 4) * feature_map_length]);
    				if (object_conf >= ignore_score)
    				{
    					float max_conf = 0;
    					float max_index = -1;
    					for (int m = 0; m < num_class; ++m)
    					{
    						float conf = logistic_activate(output[n][loc + (k * bb_size + 5 + m) * feature_map_length]);
    						if (conf > max_conf && conf > ignore_score)
    						{
    							max_conf = conf;
    							max_index = m;
    						}
    					}
    					if (max_conf * object_conf >= ignore_score)
    					{
    						float cx = (j + logistic_activate(output[n][loc + (k * bb_size + 0) * feature_map_length]) * 2 - 0.5) / feature_map_sizes[n];
    						float cy = (i + logistic_activate(output[n][loc + (k * bb_size + 1) * feature_map_length]) * 2 - 0.5) / feature_map_sizes[n];
    						float w = logistic_activate(output[n][loc + (k * bb_size + 2) * feature_map_length]) * logistic_activate(output[n][loc + (k * bb_size + 2) * feature_map_length]) * 4 * anchor[n][2*k] / model_w;
    						float h = logistic_activate(output[n][loc + (k * bb_size + 3) * feature_map_length]) * logistic_activate(output[n][loc + (k * bb_size + 3) * feature_map_length]) * 4 * anchor[n][2*k+1] / model_h;
    						float x = cx - w / 2;
    						float y = cy - h / 2;
    						
    						Face face;
    						face.bbox_tlwh = { x, y, w, h };
    						face.score = max_conf * object_conf;
    						face.class_index = max_index;
    						dets.push_back(face);
    						total_num++;
    					}
    				}
    			}
 
    		}
    	}
    }
    return dets;
}
