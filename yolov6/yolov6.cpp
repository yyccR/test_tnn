
#include "iostream"
#include <string>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef std::vector<int> YoloXAnchor;

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

void generate_anchors(const int target_height, const int target_width, std::vector<int> &strides, std::vector<YoloXAnchor> &anchors)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_width / stride;
        int num_grid_h = target_height / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1)
        {
            for (int g0 = 0; g0 < num_grid_w; ++g0)
            {
                anchors.push_back((YoloXAnchor) {g0, g1, stride});
            }
        }
    }
}

void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}

std::string fdLoadFile(std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        int size      = file.tellg();
        char* content = new char[size];
        file.seekg(0, file.beg);
        file.read(content, size);
        std::string fileContent;
        fileContent.assign(content, size);
        delete[] content;
        file.close();
        return fileContent;
    } else {
        return "";
    }
}

int main() {

    std::string image_file("/Users/yang/CLionProjects/test_tnn/data/images/img.jpg");
    cv::Mat image = cv::imread(image_file);
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(640,640));
    float w_scale = (float) image.cols / 640.f;
    float h_scale = (float) image.rows / 640.f;


    std::string proto("/Users/yang/CLionProjects/test_tnn/yolov6/yolov6n.opt.tnnproto");
    std::string model("/Users/yang/CLionProjects/test_tnn/yolov6/yolov6n.opt.tnnmodel");

    TNN_NS::TNN tnn;
    TNN_NS::ModelConfig model_config;
    model_config.params.push_back(fdLoadFile(proto));
    model_config.params.push_back(fdLoadFile(model));
    tnn.Init(model_config);

    TNN_NS::NetworkConfig config;
    config.device_type = TNN_NS::DEVICE_X86;
    TNN_NS::Status status;
    auto net_instance = tnn.CreateInst(config, status);

    std::cout << (status == TNN_NS::TNN_OK) << std::endl;

    TNN_NS::BlobMap blobs;
    net_instance->GetAllInputBlobs(blobs);
    TNN_NS::DimsVector input_shape;
    for(auto &a : blobs){
        std::cout << a.first << " " << a.second << " " << a.second->GetBlobDesc().name << " " << a.second->GetBlobDesc().data_format  << std::endl;
        input_shape = a.second->GetBlobDesc().dims;
        for(int i = 0; i < a.second->GetBlobDesc().dims.size(); i++){
            std::cout << a.second->GetBlobDesc().dims[i] << std::endl;
        }
    }

    TNN_NS::BlobMap out_blobs;
    std::string output_name;
    net_instance->GetAllOutputBlobs(out_blobs);
    for(auto &a : out_blobs){
        output_name = a.first;
        std::cout << a.first << " " << a.second << " " << a.second->GetBlobDesc().name  << std::endl;
        for(int i = 0; i < a.second->GetBlobDesc().dims.size(); i++){
            std::cout << a.second->GetBlobDesc().dims[i] << std::endl;
        }
    }

    auto input_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_X86, TNN_NS::N8UC3, input_shape, input_image.data);
    TNN_NS::MatConvertParam input_param;
    input_param.scale = {1.0/255.0f, 1.0/255.0f, 1.0/255.0f, 1.0/255.0f};
    input_param.bias = {0.0f, 0.0f, 0.0f, 0.0f};
//    input_param.reverse_channel = true;
    net_instance->SetInputMat(input_mat, input_param);


    TNN_NS::Status forward_status;
    forward_status = net_instance->Forward();

    std::shared_ptr<tnn::Mat> output_mat;
    tnn::MatConvertParam output_param; // default
    tnn::Status output_status;
    net_instance->GetOutputMat(output_mat, output_param, output_name, TNN_NS::DEVICE_X86, TNN_NS::NCHW_FLOAT);

    auto pred_dims = output_mat->GetDims();
    const unsigned int num_anchors = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;
    std::cout << "num_anchors: " << num_anchors << " num_classes: " << num_classes << std::endl;

    float score_threshold = 0.4;
    std::vector<BoxInfo> boxes;
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        const float *offset_obj_cls_ptr =
                (float *) output_mat->GetData() + (i * (num_classes + 5)); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < score_threshold) continue; // filter first.

        float cls_conf = offset_obj_cls_ptr[5];
        int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax

        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold) continue; // filter

        float cx = offset_obj_cls_ptr[0];
        float cy = offset_obj_cls_ptr[1];
        float w = offset_obj_cls_ptr[2];
        float h = offset_obj_cls_ptr[3];
        float x1 = (cx - w / 2.f) * w_scale;
        float y1 = (cy - h / 2.f) * h_scale;
        float x2 = (cx + w / 2.f) * w_scale;
        float y2 = (cy + h / 2.f) * h_scale;

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) image.cols - 1.f);
        box.y2 = std::min(y2, (float) image.rows - 1.f);
        box.score = conf;
        box.label = label;
        boxes.push_back(box);
    }
    nms(boxes, 0.3);
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);
    return 0;
}