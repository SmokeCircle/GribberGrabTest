#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <dirent.h>
#include "cmdline.h"
#include "yaml-cpp/yaml.h"
using namespace std;
using namespace cv;

static std::string prefix = "/data/GribberGrabTest";
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

class Test_bbox
{    
public:
    float resize;  // resize ratio 
    cv::Rect original_box, resized_box;  // original bbox xyhw (before resize)
    cv::Mat src, box; 
    int tmpl_id;

    Test_bbox(cv::Mat src, cv::Rect r, int id, float resize=1.0, int extra=30){
        this->src = src;
        this->resize = resize;
        this->original_box = r;
        this->tmpl_id = id;

        int x = max(0, int(resize*r.x-extra));
        int y = max(0, int(resize*r.y-extra));
        int w = min(src.cols-x, int(resize*r.width+2*extra));
        int h = min(src.rows-y, int(resize*r.height+2*extra));

        this->resized_box = Rect(x,y,w,h);
        this->box = src(resized_box);
        //cv::imshow("box", this->box);
        //cv::waitKey(0);
    }

    auto get_matches(line2Dup::Detector detector, std::vector<std::string> ids, float thresh, bool debug){
        int padding = 0;
        cv::Mat test_img = this->box;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                    test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));
        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        test_img = padded_img(roi).clone();
        assert(test_img.isContinuous());

        std::vector<line2Dup::Match> matches;
        if (this->tmpl_id==0){
            matches = detector.match(test_img, thresh, ids);
        }
        else{
            std::vector<std::string> tmp;
            tmp.push_back(ids[this->tmpl_id-1]);
            matches = detector.match(test_img, thresh, tmp);
        }
        
        if(matches.size()>0){
            auto match = matches[0];
            // add offset from box to all image
            match.x = match.x;
            match.y = match.y;
            auto templ = detector.getTemplates(match.class_id,
                                            match.template_id);

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }      
        }
        if (debug){
            //cv::imshow("/data/GribberGrabTest/test.png", test_img);
            //cv::waitKey(0);
        }

            
        return matches;
    }
};

void train_or_test(cmdline::parser args){
    line2Dup::Detector detector(256, {4, 8}, 30.0, 15.0);
    std::string mode = args.get<string>("mode");
    size_t top5 = args.get<int>("max_match");
    int blur = args.get<int>("blur");
    
    if(mode == "train"){
        std::vector<string> traing_imgs;
        std::vector<string> class_ids;
        DIR *pDir = NULL;
        struct dirent * pEnt = NULL;

        YAML::Node config;

        int img_nums = 0;
        pDir = opendir(args.get<string>("filename").data());
        bool vis = args.get<bool>("vis");
        while (1){
            pEnt = readdir(pDir);
            if(pEnt != NULL ){
                if (pEnt->d_type == DT_REG){
                    std::string d_name(pEnt->d_name);
                    // strcmp: only train png image, ignore yaml file
                    if (strcmp(d_name.c_str()+d_name.length()-4, ".png")==0){
                        traing_imgs.push_back(args.get<string>("filename") + '/' + d_name);
                        img_nums++;
                        class_ids.push_back(cv::format("test_%d", img_nums));
                        std::cout << "filename: " << d_name << " class_id: " << cv::format("test_%d", img_nums) << std::endl;

                        config[img_nums] = d_name;
                    }

                }
		    }
		    else{
			    break;
            }
        }
        std::cout << "Writing piece_id.yaml to " << args.get<string>("filename")+'/'+std::string("piece_id.yaml") << std::endl;
        std::ofstream fout(args.get<string>("filename")+'/'+std::string("piece_id.yaml"));
        fout << config;
        fout.close();
        
        int i = 0;
        for (auto& iter: traing_imgs)
        {
            Mat img = imread(iter);
            assert(!img.empty() && "check your img path");

            // resize and blur training image
            if (args.get<float>("resize")!=1.0){
            Size dsize = Size(int(img.cols*args.get<float>("resize")), int(img.rows*args.get<float>("resize")));
            cv::resize(img, img, dsize);
            }
            

            Mat mask = Mat(img.size(), CV_8UC1, {255});

            // padding to avoid rotating out
            int padding = 100;
            int m = max(img.rows, img.cols);
            cv::Mat padded_img = cv::Mat(m + 2*padding, m + 2*padding, img.type(), cv::Scalar::all(0));
            img.copyTo(padded_img(Rect(padding + (m-img.cols)/2, padding + (m-img.rows)/2, img.cols, img.rows)));
            cv::Mat padded_mask = cv::Mat(m + 2*padding, m + 2*padding, mask.type(), cv::Scalar::all(255));
            mask.copyTo(padded_mask(Rect(padding + (m-img.cols)/2, padding + (m-img.rows)/2, img.cols, img.rows)));

            cv::GaussianBlur(padded_img, padded_img ,Size(blur,blur), 0, 0);

            shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);

            shapes.angle_range = {0, 360};
            shapes.angle_step = 90;
            shapes.scale_range = {0.85f, 1.05f};
            shapes.scale_step = 0.01f;
            shapes.produce_infos();
            std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
            
            for(auto& info: shapes.infos){
                //cv::imshow("train", shapes.src_of(info));
                //cv::waitKey(1);

                int templ_id = detector.addTemplate(shapes.src_of(info), class_ids[i], shapes.mask_of(info));
                if(templ_id != -1){
                    infos_have_templ.push_back(info);

                    // Debug : visualize train features
                    if (vis)
                    {cv::Mat Q = shapes.src_of(info).clone();
                    auto templ = detector.getTemplates(class_ids[i],
                                               templ_id);

                    for(int i=0; i<templ[0].features.size(); i++){
                        auto feat = templ[0].features[i];
                        cv::circle(Q, {feat.x, feat.y}, 2, (0,0,255), -1);
                    }
                    std::cout << "feature num:" << templ[0].features.size() << std::endl;
                    cv::imshow("train", Q);
                    cv::waitKey(1);}
                    
                }
            }
            detector.writeClasses(prefix+"/data/yaml/"+args.get<string>("nest")+"/%s_templ.yaml");
            shapes.save_infos(infos_have_templ, prefix+"/data/yaml/"+args.get<string>("nest")+"/common_info.yaml");
            std::cout << "successfuly trained template " << i << " class id: " << class_ids[i] << std::endl;
            i++;
        }
        
        
        std::cout << "train end" << std::endl;
    }
    else if (mode=="test" && args.get<string>("yaml")!=""){
        const string DB_CONF=args.get<string>("yaml");
        YAML::Node conf = YAML::LoadFile(DB_CONF);

        Mat test_img = imread(conf["path"].as<string>());
        assert(!test_img.empty() && "check your img path");

        std::vector<std::string> ids;
        for (int i=0; i < args.get<int>("object_nums"); i++){
            ids.push_back(cv::format("test_%d", i+1));
        }
        detector.readClasses(ids, prefix+"/data/yaml/"+args.get<string>("nest")+"/%s_templ.yaml");

        if (args.get<float>("resize")!=1.0){
        Size dsize = Size(int(test_img.cols*args.get<float>("resize")), int(test_img.rows*args.get<float>("resize")));
        cv::resize(test_img, test_img, dsize);
        }
        cv::GaussianBlur(test_img, test_img ,Size(blur, blur), 0, 0);

        std::vector<Test_bbox> bboxes;
        for (int i=0; i<conf["nums"].as<int>(); i++){
            bboxes.push_back(Test_bbox(test_img, 
                            cv::Rect(conf["infos"][i]["x"].as<int>(), conf["infos"][i]["y"].as<int>(),
                                     conf["infos"][i]["w"].as<int>(), conf["infos"][i]["h"].as<int>()),
                            conf["infos"][i]["piece_id"].as<int>(), args.get<float>("resize")));
        }

        // begin matching in each bbox
        cv::Mat to_show = test_img.clone();
        for (auto test_bbox:bboxes){
            if (args.get<bool>("debug"))
            {std::cout << "\n!!!!!!!!!!!!Test begin" << std::endl;}
            
            auto matches = test_bbox.get_matches(detector, ids, args.get<float>("threshold"), args.get<bool>("debug"));

            if (args.get<bool>("debug"))
            {std::cout << "matches.size(): " << matches.size() << std::endl;}
            
            int N=top5;
            if(top5>matches.size()) N=matches.size();
            //load infos (scale and angle) fetched by template id
            auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix+"/data/yaml/"+args.get<string>("nest")+"/common_info.yaml");
            for(int idx=0; idx<N; idx++){
                auto match = matches[idx];
                // add offset from box to all image
                match.x = match.x + test_bbox.resized_box.x;
                match.y = match.y + test_bbox.resized_box.y;
                auto templ = detector.getTemplates(match.class_id,
                                                match.template_id);

                int x =  templ[0].width + match.x;
                int y = templ[0].height + match.y;  // end point {x y}
                int r = templ[0].width/2;
                cv::Vec3b randColor;
                randColor[0] = rand()%155 + 100;
                randColor[1] = rand()%155 + 100;
                randColor[2] = rand()%155 + 100;

                
                for(int i=0; i<templ[0].features.size(); i++){
                    auto feat = templ[0].features[i];
                    if (args.get<bool>("debug"))
                    {cv::circle(to_show, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);}
                }
                if (args.get<bool>("debug")){
                    cv::putText(to_show, to_string(int(round(match.similarity))),
                                Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
                    cv::rectangle(to_show, {match.x, match.y}, {x, y}, randColor, 2);               
                    std::cout << "match.class_id: " << match.class_id << std::endl;
                    std::cout << "match.template_id: " << match.template_id << std::endl;
                    std::cout << "match.similarity: " << match.similarity << std::endl;
                    std::cout << cv::format("match.x, match.y: {%d, %d}", match.x, match.y) << std::endl;
                    std::cout << "scale: " << infos[match.template_id].scale << std::endl;
                    std::cout << "angle: " << infos[match.template_id].angle << std::endl;
                }
                else{
                    std::cout << "match id " << match.class_id << cv::format(" x %d y %d angle %d scale %.2f sim %.2f", 
                                            int(match.x/args.get<float>("resize")), int(match.y/args.get<float>("resize")), int(infos[match.template_id].angle), infos[match.template_id].scale, match.similarity) << std::endl;
                }
            }



        }

        if (args.get<bool>("debug")) {
            cv::resize(to_show, to_show, Size(), 0.25, 0.25);
            cv::imwrite("/data/GribberGrabTest/test.png", to_show);
            //cv::waitKey(0);
            //cv::destroyAllWindows();
        }

        std::cout << "test end" << std::endl;
    }
}


int main(int argc,char **argv){
    cmdline::parser args;
    args.add<string>("filename", '\0', "file name", false, "/home/chuan/linemod-2D/data/Test_image1.png");
    args.add<string>("mode", 'm', "mode:train or test", false, "test");
    args.add<string>("nest", '\0', "nest ID", false, "");
    args.add<float>("resize", 'r', "resize ratio, default=1", false, 1.0);
    args.add<int>("object_nums", 'n', "max object numbers", false, 1);
    args.add<float>("threshold", 't', "match threshold", false, 90);
    args.add<int>("max_match", 'x', "max match number", false, 5);
    args.add<string>("yaml", '\0', "if set, use bbx yaml file path(test mode only)", false, "");
    args.add<int>("blur", '\0', "Gaussian blur parameter (odd)", false, 15);
    args.add<bool>("vis", 'v', "if set, visualize traing process)", false, false);
    args.add<bool>("debug", '\0', "if set, output debug info", false, true);
    args.parse_check(argc, argv);
    train_or_test(args);
    return 0;
}
