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

//#define USE_FREE
#define USE_IMAGES_SET

#include <iostream>
#include <string>
#include <vector>

#include <boost/log/core.hpp>

// ADD MODULES TRAITS HEADERS HERE
#include "SolARModuleOpencv_traits.h"
#include "SolARModuleOpengl_traits.h"
#include "SolARModuleTools_traits.h"

#ifndef USE_FREE
    #include "SolARModuleNonFreeOpencv_traits.h"
#endif

// ADD XPCF HEADERS HERE
#include "xpcf/xpcf.h"
#include "xpcf/threading/SharedBuffer.h"
#include "xpcf/threading/BaseTask.h"

// ADD COMPONENTS HEADERS HERE
#include "api/input/devices/ICamera.h"
#include "api/features/IKeypointDetector.h"
#include "api/features/IDescriptorsExtractor.h"
#include "api/features/IDescriptorMatcher.h"
#include "api/solver/pose/I3DTransformFinderFrom2D2D.h"
#include "api/solver/map/ITriangulator.h"
#include "api/solver/map/IMapper.h"
#include "api/solver/map/IKeyframeSelector.h"
#include "api/solver/map/IMapFilter.h"
#include "api/solver/pose/I2D3DCorrespondencesFinder.h"
#include "api/solver/pose/I3DTransformFinderFrom2D3D.h"
#include "api/features/IMatchesFilter.h"
#include "api/display/I2DOverlay.h"
#include "api/display/IMatchesOverlay.h"
#include "api/display/I3DOverlay.h"
#include "api/display/IImageViewer.h"
#include "api/display/I3DPointsViewer.h"

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
#ifndef USE_FREE
using namespace SolAR::MODULES::NONFREEOPENCV;
#endif
using namespace SolAR::MODULES::OPENGL;
using namespace SolAR::MODULES::TOOLS;

namespace xpcf = org::bcom::xpcf;

int main(int argc, char **argv){

#if NDEBUG
    boost::log::core::get()->set_logging_enabled(false);
#endif

    LOG_ADD_LOG_TO_CONSOLE();

    /* instantiate component manager*/
    /* this is needed in dynamic mode */
    SRef<xpcf::IComponentManager> xpcfComponentManager = xpcf::getComponentManagerInstance();

    if(xpcfComponentManager->load("conf_SLAM.xml")!=org::bcom::xpcf::_SUCCESS)
    {
        LOG_ERROR("Failed to load the configuration file conf_SLAM.xml")
        return -1;
    }

    // declare and create components
    LOG_INFO("Start creating components");

 // component creation
#ifdef USE_IMAGES_SET
    auto camera = xpcfComponentManager->create<SolARImagesAsCameraOpencv>()->bindTo<input::devices::ICamera>();
#else
    auto camera =xpcfComponentManager->create<SolARCameraOpencv>()->bindTo<input::devices::ICamera>();
#endif
#ifdef USE_FREE
    auto keypointsDetector =xpcfComponentManager->create<SolARKeypointDetectorOpencv>()->bindTo<features::IKeypointDetector>();
    auto descriptorExtractor =xpcfComponentManager->create<SolARDescriptorsExtractorAKAZE2Opencv>()->bindTo<features::IDescriptorsExtractor>();
#else
   auto  keypointsDetector = xpcfComponentManager->create<SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
   auto descriptorExtractor = xpcfComponentManager->create<SolARDescriptorsExtractorSURF64Opencv>()->bindTo<features::IDescriptorsExtractor>();
#endif

 //   auto descriptorExtractorORB =xpcfComponentManager->create<SolARDescriptorsExtractorORBOpencv>()->bindTo<features::IDescriptorsExtractor>();
    SRef<features::IDescriptorMatcher> matcher =xpcfComponentManager->create<SolARDescriptorMatcherKNNOpencv>()->bindTo<features::IDescriptorMatcher>();
    SRef<solver::pose::I3DTransformFinderFrom2D2D> poseFinderFrom2D2D =xpcfComponentManager->create<SolARPoseFinderFrom2D2DOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D2D>();
    SRef<solver::map::ITriangulator> triangulator =xpcfComponentManager->create<SolARSVDTriangulationOpencv>()->bindTo<solver::map::ITriangulator>();
    SRef<features::IMatchesFilter> matchesFilter =xpcfComponentManager->create<SolARGeometricMatchesFilterOpencv>()->bindTo<features::IMatchesFilter>();
    SRef<solver::pose::I3DTransformFinderFrom2D3D> PnP =xpcfComponentManager->create<SolARPoseEstimationPnpOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D3D>();
    SRef<solver::pose::I2D3DCorrespondencesFinder> corr2D3DFinder =xpcfComponentManager->create<SolAR2D3DCorrespondencesFinderOpencv>()->bindTo<solver::pose::I2D3DCorrespondencesFinder>();
    SRef<solver::map::IMapFilter> mapFilter =xpcfComponentManager->create<SolARMapFilter>()->bindTo<solver::map::IMapFilter>();
    SRef<solver::map::IMapper> mapper =xpcfComponentManager->create<SolARMapper>()->bindTo<solver::map::IMapper>();
    SRef<solver::map::IKeyframeSelector> keyframeSelector =xpcfComponentManager->create<SolARKeyframeSelector>()->bindTo<solver::map::IKeyframeSelector>();

    SRef<display::IMatchesOverlay> matchesOverlay =xpcfComponentManager->create<SolARMatchesOverlayOpencv>()->bindTo<display::IMatchesOverlay>();
    SRef<display::IMatchesOverlay> matchesOverlayBlue =xpcfComponentManager->create<SolARMatchesOverlayOpencv>("matchesBlue")->bindTo<display::IMatchesOverlay>();
    SRef<display::IMatchesOverlay> matchesOverlayRed =xpcfComponentManager->create<SolARMatchesOverlayOpencv>("matchesRed")->bindTo<display::IMatchesOverlay>();

    SRef<display::IImageViewer> imageViewer =xpcfComponentManager->create<SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();
    SRef<display::I3DPointsViewer> viewer3DPoints =xpcfComponentManager->create<SolAR3DPointsViewerOpengl>()->bindTo<display::I3DPointsViewer>();

    /* in dynamic mode, we need to check that components are well created*/
    /* this is needed in dynamic mode */

    if ( !camera || !keypointsDetector || !descriptorExtractor || !descriptorExtractor || !matcher ||
         !poseFinderFrom2D2D || !triangulator || !mapFilter || !mapper || !keyframeSelector || !PnP ||
         !corr2D3DFinder || !matchesFilter || !matchesOverlay || !imageViewer  || !viewer3DPoints)
    {
        LOG_ERROR("One or more component creations have failed");
        return -1;
    }


    // data structure declarations
    SRef<Image>                                         view1, view2;
    SRef<Keyframe>                                      keyframe1;
    SRef<Keyframe>                                      keyframe2;
    std::vector<SRef<Keypoint>>                         keypointsView1, keypointsView2, keypoints;
    SRef<DescriptorBuffer>                              descriptorsView1, descriptorsView2, descriptors;
    std::vector<DescriptorMatch>                        matches;

    Transform3Df                                        poseFrame1 = Transform3Df::Identity();
    Transform3Df                                        poseFrame2;
    Transform3Df                                        newFramePose;
    Transform3Df                                        lastPose;

    std::vector<SRef<CloudPoint>>                       cloud, filteredCloud;

    std::vector<Transform3Df>                           keyframePoses;
    std::vector<Transform3Df>                           framePoses;

    SRef<Frame>                                         newFrame;

    SRef<Keyframe>                                      referenceKeyframe;
    SRef<Keyframe>                                      newKeyframe;

    SRef<Image>                                         imageMatches, imageMatches2;

    SRef<Map> map;

    // initialize pose estimation with the camera intrinsic parameters (please refeer to the use of intrinsec parameters file)
    PnP->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());
    poseFinderFrom2D2D->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());
    triangulator->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());

    LOG_DEBUG("Intrincic parameters : \n {}", camera->getIntrinsicsParameters());

    if (camera->start() != FrameworkReturnCode::_SUCCESS)
    {
        LOG_ERROR("Camera cannot start");
        return -1;
    }

    // Here, Capture the two first keyframe view1, view2
    bool imageCaptured = false;
    while (!imageCaptured)
    {
        if (camera->getNextImage(view1) == SolAR::FrameworkReturnCode::_ERROR_)
            break;
#ifdef USE_IMAGES_SET
        imageViewer->display(view1);
#else
        if (imageViewer->display(view1) == SolAR::FrameworkReturnCode::_STOP)
#endif
        {
            keypointsDetector->detect(view1, keypointsView1);
            descriptorExtractor->extract(view1, keypointsView1, descriptorsView1);

            keyframe1 = xpcf::utils::make_shared<Keyframe>(keypointsView1, descriptorsView1, view1, poseFrame1);
            mapper->update(map, keyframe1);
            keyframePoses.push_back(poseFrame1); // used for display
            imageCaptured = true;
        }
    }

    bool bootstrapOk= false;
    while (!bootstrapOk)
    {
        if (camera->getNextImage(view2) == SolAR::FrameworkReturnCode::_ERROR_)
            break;

        keypointsDetector->detect(view2, keypointsView2);

        descriptorExtractor->extract(view2, keypointsView2, descriptorsView2);
        SRef<Frame> frame2 = xpcf::utils::make_shared<Frame>(keypointsView2, descriptorsView2, view2, keyframe1);
        matcher->match(descriptorsView1, descriptorsView2, matches);
        int nbOriginalMatches = matches.size();
        matchesFilter->filter(matches, matches, keypointsView1, keypointsView2);

        matchesOverlay->draw(view2, imageMatches, keypointsView1, keypointsView2, matches);
        if(imageViewer->display(imageMatches) == SolAR::FrameworkReturnCode::_STOP)
           return 1;

       if (keyframeSelector->select(frame2, matches))
        {
            // Estimate the pose of of the second frame (the first frame being the reference of our coordinate system)
            poseFinderFrom2D2D->estimate(keypointsView1, keypointsView2, poseFrame1, poseFrame2, matches);
            LOG_INFO("Nb matches for triangulation: {}\\{}", matches.size(), nbOriginalMatches);
            LOG_INFO("Estimate pose of the camera for the frame 2: \n {}", poseFrame2.matrix());
            frame2->setPose(poseFrame2);

            // Triangulate
            double reproj_error = triangulator->triangulate(keypointsView1,keypointsView2, matches, std::make_pair(0, 1),poseFrame1, poseFrame2,cloud);
            mapFilter->filter(poseFrame1, poseFrame2, cloud, filteredCloud);
            keyframePoses.push_back(poseFrame2); // used for display
            keyframe2 = xpcf::utils::make_shared<Keyframe>(frame2);
            mapper->update(map, keyframe2, filteredCloud, matches);

            bootstrapOk = true;
        }
    }

    referenceKeyframe = keyframe2;
    lastPose = poseFrame2;

    // Start tracking
    /*
     *      Threads Definition
     */
    bool keyFrameDetectionOn;
    // Camera capture Thread
    xpcf::SharedBuffer< SRef<Image> > displayBufferCamImages(1);
    xpcf::SharedBuffer< SRef<Image> > workingBufferCamImages(1);

    std::function<void(void)> getCameraImages = [camera,&displayBufferCamImages,&workingBufferCamImages](){
        SRef<Image> view;
        camera->getNextImage(view);
        workingBufferCamImages.push(view);
    };

    // Keypoints Detection Thread
    xpcf::SharedBuffer< std::pair< SRef<Image>,std::vector<SRef<Keypoint>> > > outBufferKeypoints(1);

    std::function<void(void)> getKeyPoints = [&workingBufferCamImages,keypointsDetector, &outBufferKeypoints](){
        SRef<Image> camImage=workingBufferCamImages.pop();
        std::vector< SRef<Keypoint>> kp;
        keypointsDetector->detect(camImage, kp);
        outBufferKeypoints.push(std::make_pair(camImage,kp));
    };

    // Keypoints Descriptor extraction Thread
    xpcf::SharedBuffer< SRef<Frame > > outBufferDescriptors(1);

    std::function<void(void)> getDescriptors = [&outBufferKeypoints,descriptorExtractor, &referenceKeyframe, &outBufferDescriptors](){
        std::pair<SRef<Image>,std::vector<SRef<Keypoint> > > kp=outBufferKeypoints.pop();
        SRef<DescriptorBuffer> camDescriptors;
        descriptorExtractor->extract(kp.first, kp.second, camDescriptors);
        Transform3Df pose;
        SRef<Frame> frame=xpcf::utils::make_shared<Frame>(kp.second,camDescriptors,kp.first,referenceKeyframe);
        outBufferDescriptors.push(frame) ;
    };

    // Map updating Thread
    xpcf::SharedBuffer< std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  > >  outBufferTriangulation(1);

    std::function<void(void)> mapUpdate = [&referenceKeyframe, &map, &mapper, &mapFilter, &keyframePoses, &outBufferTriangulation](){
        std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  >   element;
        SRef<Frame>                                         newFrame;
        SRef<Keyframe>                                      newKeyframe,refKeyframe;
        std::vector<DescriptorMatch>                        foundMatches, remainingMatches;
        std::vector<SRef<CloudPoint>>                       newCloud, filteredCloud;


        element=outBufferTriangulation.pop();

        newFrame=std::get<0>(element);
        refKeyframe=std::get<1>(element);
        foundMatches=std::get<2>(element);
        remainingMatches=std::get<3>(element);
        newCloud=std::get<4>(element);

        LOG_DEBUG(" frame pose estimation :\n {}", newFrame->getPose().matrix());
        LOG_DEBUG("Number of matches: {}, number of 3D points:{}", remainingMatches.size(), newCloud.size());
        newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
        mapFilter->filter(refKeyframe->getPose(), newFrame->getPose(), newCloud, filteredCloud);
        mapper->update(map, newKeyframe, filteredCloud, foundMatches, remainingMatches);
        referenceKeyframe = newKeyframe;
        keyframePoses.push_back(newKeyframe->getPose());
        LOG_DEBUG(" cloud current size: {} \n", map->getPointCloud()->size());
        LOG_DEBUG (" Valid pose is found");
    };

    // Triangulation Thread
    xpcf::SharedBuffer< std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch> > >  keyFrameBuffer(1);

    std::function<void(void)> doTriangulation = [mapUpdate,&triangulator,&keyFrameBuffer,&outBufferTriangulation](){
        std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch> > element;
        SRef<Frame>                                         newFrame;
        SRef<Keyframe>                                      newKeyFrame;
        SRef<Keyframe>                                      refKeyFrame;
        std::vector<DescriptorMatch>                        foundMatches;
        std::vector<DescriptorMatch>                        remainingMatches;
        std::vector<SRef<CloudPoint>>                       newCloud;


        LOG_DEBUG ("**************************   doTriangulation In");

        element=keyFrameBuffer.pop();

        newFrame=std::get<0>(element);
        refKeyFrame=std::get<1>(element);
        foundMatches=std::get<2>(element);
        remainingMatches=std::get<3>(element);

        triangulator->triangulate(refKeyFrame->getKeypoints(), newFrame->getKeypoints(), remainingMatches,std::make_pair<int,int>((int)refKeyFrame->m_idx+0,(int)(refKeyFrame->m_idx+1)),
                            refKeyFrame->getPose(), newFrame->getPose(), newCloud);
        outBufferTriangulation.push(std::make_tuple(newFrame,refKeyFrame,foundMatches,remainingMatches,newCloud));
    };

    // Processing of input frames :
    // - perform match+PnP
    // - test if current frame can promoted to a keyFrame.
    // - in that case, push it in the output buffer to be processed by the triangulation thread
    //
    std::function<void(void)> processFrames = [&keyFrameDetectionOn,&outBufferTriangulation,mapUpdate,&mapper,&keyframeSelector, &matchesOverlayBlue,&matchesOverlayRed,&imageViewer,&framePoses,&outBufferDescriptors,matcher,matchesFilter,corr2D3DFinder,PnP,&referenceKeyframe, &lastPose,&keyFrameBuffer](){

         SRef<Frame> newFrame;
         SRef<Keyframe> newKeyframe;
         SRef<Keyframe> refKeyFrame;
         SRef<Image> view;
         std::vector< SRef<Keypoint> > keypoints;
         SRef<DescriptorBuffer> descriptors;
         SRef<DescriptorBuffer> refDescriptors;
         std::vector<DescriptorMatch> matches;
         Transform3Df newFramePose;

         std::vector<SRef<Point2Df>> pt2d;
         std::vector<SRef<Point3Df>> pt3d;
         std::vector<SRef<CloudPoint>> foundPoints;
         std::vector<DescriptorMatch> foundMatches;
         std::vector<DescriptorMatch> remainingMatches;
         std::vector<SRef<Point2Df>> imagePoints_inliers;
         std::vector<SRef<Point3Df>> worldPoints_inliers;


         // test if a triangulation has been performed on a previously keyframe candidate
         if(!outBufferTriangulation.empty()){
             //if so update the map et re-allow keyframe detection
             mapUpdate();
             keyFrameDetectionOn=true;
         }

         /*compute matches between reference image and camera image*/
         newFrame=outBufferDescriptors.pop();

         // referenceKeyframe can be changed outside : let's make a copy.
         refKeyFrame=referenceKeyframe;

         view=newFrame->getView();
         keypoints=newFrame->getKeypoints();
         descriptors=newFrame->getDescriptors();


         refDescriptors=refKeyFrame->getDescriptors();
         matcher->match(refDescriptors, descriptors, matches);

         /* filter matches to remove redundancy and check geometric validity */
         matchesFilter->filter(matches, matches, refKeyFrame->getKeypoints(), keypoints);

         corr2D3DFinder->find(refKeyFrame, newFrame, matches, foundPoints, pt3d, pt2d, foundMatches, remainingMatches);

         SRef<Image> imageMatches, imageMatches2;
         matchesOverlayBlue->draw(view, imageMatches, refKeyFrame->getKeypoints(), keypoints, foundMatches);
         matchesOverlayRed->draw(imageMatches, imageMatches2, refKeyFrame->getKeypoints(), keypoints, remainingMatches);
         if (imageViewer->display(imageMatches2) == FrameworkReturnCode::_STOP)
             return 0;

         if (PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, newFramePose , lastPose) == FrameworkReturnCode::_SUCCESS){
             LOG_INFO(" frame pose  :\n {}", newFramePose.matrix());
             LOG_INFO(" pnp inliers size: {} / {}",worldPoints_inliers.size(), pt3d.size());

           lastPose = newFramePose;

             // Create a new frame
           newFrame = xpcf::utils::make_shared<Frame>(keypoints, descriptors, view, refKeyFrame, newFramePose);

             // If the camera has moved enough, create a keyframe and map the scene
           if (keyFrameDetectionOn && keyframeSelector->select(newFrame, matches))
           {

               keyFrameDetectionOn=false;
               keyFrameBuffer.push(std::make_tuple(newFrame,refKeyFrame,foundMatches,remainingMatches));
            }
             else{
                 LOG_DEBUG (" No valid pose was found");
                 framePoses.push_back(newFramePose); // used for display
                 LOG_INFO(" framePoses current size: {} \n", framePoses.size());

             }
         }
    };

    xpcf::DelegateTask taskGetCameraImages(getCameraImages);
    xpcf::DelegateTask taskGetKeyPoints(getKeyPoints);
    xpcf::DelegateTask taskGetDescriptors(getDescriptors);
    xpcf::DelegateTask taskProcessFrames(processFrames);
    xpcf::DelegateTask taskDoTriangulation(doTriangulation);
    xpcf::DelegateTask taskMapUpdate(mapUpdate);

    taskGetCameraImages.start();
    taskGetKeyPoints.start();
    taskGetDescriptors.start();
    taskProcessFrames.start();
    taskDoTriangulation.start();

    // running loop process

    keyFrameDetectionOn=true;

    bool stop = false;
    while(!stop){
        if (viewer3DPoints->display(*(map->getPointCloud()), lastPose, keyframePoses, framePoses) == FrameworkReturnCode::_STOP){
               stop=true;
        }
    }

   taskDoTriangulation.stop();
   taskProcessFrames.stop();
   taskGetDescriptors.stop();
   taskGetKeyPoints.stop();
   taskGetCameraImages.stop();

}




 /*
        // match current keypoints with the keypoints of the Keyframe
        SRef<DescriptorBuffer> referenceKeyframeDescriptors = referenceKeyframe->getDescriptors();
  //      matcherKNN->match(referenceKeyframeDescriptors, descriptors, matches);

        matcher->match(referenceKeyframeDescriptors, descriptors, matches);

        //std::cout<<"original matches: "<<matches.size()<<std::endl;
        matchesFilter->filter(matches, matches, referenceKeyframe->getKeypoints(), keypoints);
        //std::cout<<"filtred matches: "<<matches.size()<<std::endl;

        std::vector<SRef<Point2Df>> pt2d;
        std::vector<SRef<Point3Df>> pt3d;
        std::vector<SRef<CloudPoint>> foundPoints;
        std::vector<DescriptorMatch> foundMatches;
        std::vector<DescriptorMatch> remainingMatches;

        corr2D3DFinder->find(referenceKeyframe->getVisibleMapPoints(), referenceKeyframe->m_idx, matches, keypoints, foundPoints, pt3d, pt2d, foundMatches, remainingMatches);



        LOG_DEBUG("found matches {}, Remaining Matches {}", foundMatches.size(), remainingMatches.size());
        // display matches
        matchesOverlayBlue->draw(view, imageMatches, referenceKeyframe->getKeypoints(), keypoints, foundMatches);
        matchesOverlayRed->draw(imageMatches, imageMatches2, referenceKeyframe->getKeypoints(), keypoints, remainingMatches);
        if (imageViewer->display(imageMatches2) == FrameworkReturnCode::_STOP)
            return 0;

        std::vector<SRef<Point2Df>> imagePoints_inliers;
        std::vector<SRef<Point3Df>> worldPoints_inliers;

        if (PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, newFramePose , lastPose) == FrameworkReturnCode::_SUCCESS){
            //LOG_DEBUG(" pnp inliers size: {} / {}",worldPoints_inliers.size(), pt3d.size());
            lastPose = newFramePose;

            // Set the pose of the new frame
            newFrame->setPose(newFramePose);

            // If the camera has moved enough, create a keyframe and map the scene
            if (keyframeSelector->select(newFrame, matches))
            {
                // triangulate with the reference keyframe
                std::vector<SRef<CloudPoint>>newCloud, filteredCloud;
                triangulator->triangulate(referenceKeyframe->getKeypoints(), keypoints, remainingMatches,std::make_pair<int,int>((int)referenceKeyframe->m_idx+0,(int)(referenceKeyframe->m_idx+1)),
                                          referenceKeyframe->getPose(), newFramePose, newCloud);

                // remove abnormal 3D points from the new cloud
                mapFilter->filter(referenceKeyframe->getPose(), newFramePose, newCloud, filteredCloud);
                LOG_DEBUG("Number of matches: {}, number of 3D points:{}", remainingMatches.size(), filteredCloud.size());

                // create a new keyframe from the current frame and add it with the cloud to the mapper
                newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
                mapper->update(map, newKeyframe, filteredCloud, foundMatches, remainingMatches);
                keyframePoses.push_back(newKeyframe->getPose());
                referenceKeyframe = newKeyframe;
                LOG_DEBUG(" cloud current size: {} \n", map->getPointCloud()->size());
            }
            else
            {
                framePoses.push_back(newFramePose); // used for display
            }

        }else{
            LOG_INFO("Pose estimation has failed");
        }
        if (viewer3DPoints->display(*(map->getPointCloud()), lastPose, keyframePoses, framePoses) == FrameworkReturnCode::_STOP)
            return 0;


    return 0;
}
*/


