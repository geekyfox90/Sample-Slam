/**
 * @copyright Copyright (c) 2017 B-com http://www.b-com.com/
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <boost/log/core.hpp>
#include "core/Log.h"
#include "xpcf/xpcf.h"


// ADD COMPONENTS HEADERS HERE, e.g #include "SolarComponent.h"

#include "PipelineManager.h"

using namespace SolAR;
using namespace SolAR::PIPELINE;

namespace xpcf  = org::bcom::xpcf;

#define MARKER_CONFIGFILE "fiducialMarker.yml"

#include "SolARModuleOpencv_traits.h"
#include "SolARImageViewerOpencv.h"
#include "SolAR3DOverlayBoxOpencv.h"

using namespace SolAR;
using namespace SolAR::api;

int main(){
#if NDEBUG
    boost::log::core::get()->set_logging_enabled(false);
#endif

    LOG_ADD_LOG_TO_CONSOLE();
    clock_t start, end;
    int count = 0;
    start = clock();

    PipelineManager pipeline;
    if (pipeline.init("PipelineSlam.xml", "577ccd2c-de1b-402a-8829-496747598588"))
    {
        auto imageViewerResult = xpcf::getComponentManagerInstance()->create<MODULES::OPENCV::SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();
        auto overlay3DComponent = xpcf::getComponentManagerInstance()->create<MODULES::OPENCV::SolAR3DOverlayBoxOpencv>()->bindTo<display::I3DOverlay>();

        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("position")->setFloatingValue(0,0);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("position")->setFloatingValue(0,1);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("position")->setFloatingValue(0,2);

        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("orientation")->setFloatingValue(0,0);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("orientation")->setFloatingValue(0,1);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("orientation")->setFloatingValue(0,2);

        float marker_width=0.157;
        float marker_height=0.157;


        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("size")->setFloatingValue(marker_width,0);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("size")->setFloatingValue(marker_height,1);
        overlay3DComponent->bindTo<xpcf::IConfigurable>()->getProperty("size")->setFloatingValue(0.1,2);

        overlay3DComponent->bindTo<xpcf::IConfigurable>()->onConfigured();


        // Set camera parameters
        CamCalibration intrinsic_param = CamCalibration::Identity();
        CamDistortion  distorsion_param = CamDistortion::Zero();
        PipelineManager::CamParams calib = pipeline.getCameraParameters();
        intrinsic_param(0,0) = calib.focalX;
        intrinsic_param(1,1) = calib.focalY;
        intrinsic_param(0,2) = (float)calib.centerX;
        intrinsic_param(1,2) = (float)calib.centerY;

        overlay3DComponent->setCameraParameters(intrinsic_param, distorsion_param);

        unsigned char* r_imageData=new unsigned char[calib.width*calib.height*3];
        SRef<Image> camImage=xpcf::utils::make_shared<Image>(r_imageData,calib.width,calib.height,SolAR::Image::LAYOUT_BGR,SolAR::Image::INTERLEAVED,SolAR::Image::TYPE_8U);

        Transform3Df s_pose;

        if (pipeline.start(camImage->data()))
        {
            while (true)
            {
                PipelineManager::Pose pose;

                PIPELINEMANAGER_RETURNCODE returnCode = pipeline.udpate(pose);
                if(returnCode==PIPELINEMANAGER_RETURNCODE::_ERROR){
                    pipeline.stop();
                    break;
                }
                if ((returnCode & PIPELINEMANAGER_RETURNCODE::_NEW_POSE))
                {
//                    LOG_INFO("Camera Pose translation ({}, {}, {})", pose.translation(0), pose.translation(1), pose.translation(2));
                    for(int i=0;i<3;i++)
                         for(int j=0;j<3;j++)
                             s_pose(i,j)=pose.rotation(i,j);
                    for(int i=0;i<3;i++)
                             s_pose(i,3)=pose.translation(i);
                    for(int j=0;j<3;j++)
                        s_pose(3,j)=0;
                    s_pose(3,3)=1;
//                    LOG_INFO("pose.matrix():\n {} \n",s_pose.matrix())
                    overlay3DComponent->draw(s_pose, camImage);
                    count++;
                }

                if (imageViewerResult->display(camImage) == SolAR::FrameworkReturnCode::_STOP){
                    pipeline.stop();
                    break;
                }
             }
        }
    }
    // display stats on frame rate
    end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    printf("\n\nElasped time is %.2lf seconds.\n", duration);
    printf("Number of computed poses per second : %8.2f\n", count / duration);

}





