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

#define USE_FREE
//#define USE_IMAGES_SET

#include <iostream>
#include <string>
#include <vector>

#include <boost/log/core.hpp>

// ADD MODULES TRAITS HEADERS HERE
#include "SolARModuleOpencv_traits.h"
#include "SolARModuleOpengl_traits.h"
#include "SolARModuleTools_traits.h"
#include "SolARModuleFBOW_traits.h"

#ifndef USE_FREE
    #include "SolARModuleNonFreeOpencv_traits.h"
#endif

// ADD XPCF HEADERS HERE
#include "xpcf/xpcf.h"
#include "xpcf/threading/SharedBuffer.h"
#include "xpcf/threading/BaseTask.h"
#include <xpcf/threading/DropBuffer.h>

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
#include "api/reloc/IKeyframeRetriever.h"

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::FBOW;
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

    std::string configxml = std::string("conf_SLAM.xml");
    if (argc == 2)
        configxml = std::string(argv[1]);
    if (xpcfComponentManager->load(configxml.c_str()) != org::bcom::xpcf::_SUCCESS)
    {
        LOG_ERROR("Failed to load the configuration file {}", configxml.c_str())
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

    // KeyframeRetriever component to relocalize
    SRef<reloc::IKeyframeRetriever> kfRetriever = xpcfComponentManager->create<SolARKeyframeRetrieverFBOW>()->bindTo<reloc::IKeyframeRetriever>();

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
    SRef<Frame>											frameToTrack;

    SRef<Keyframe>                                      referenceKeyframe;
    SRef<Keyframe>                                      newKeyframe;

    SRef<Image>                                         imageMatches, imageMatches2;

    SRef<Map> map;

    bool                                                keyFrameDetectionOn;   // if true, keyFrames can be detected

    bool												stop = false;


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
            kfRetriever->addKeyframe(keyframe1); // add keyframe for reloc
            imageCaptured = true;
        }
    }

    bool bootstrapOk= false;
    while (!bootstrapOk)
    {
        if (camera->getNextImage(view2) == SolAR::FrameworkReturnCode::_ERROR_LOAD_IMAGE)
            break;

        keypointsDetector->detect(view2, keypointsView2);

        descriptorExtractor->extract(view2, keypointsView2, descriptorsView2);
        SRef<Frame> frame2 = xpcf::utils::make_shared<Frame>(keypointsView2, descriptorsView2, view2, keyframe1);
        matcher->match(descriptorsView1, descriptorsView2, matches);
        int nbOriginalMatches = matches.size();
        matchesFilter->filter(matches, matches, keypointsView1, keypointsView2);

        matchesOverlay->draw(view2, imageMatches, keypointsView1, keypointsView2, matches);
        if(imageViewer->display(imageMatches) == SolAR::FrameworkReturnCode::_ERROR_LOAD_IMAGE)
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
            kfRetriever->addKeyframe(keyframe2); // add keyframe for reloc

            bootstrapOk = true;
        }
    }

    referenceKeyframe = keyframe2;
    lastPose = poseFrame2;

    // copy referenceKeyframe to frameToTrack
    frameToTrack = xpcf::utils::make_shared<Frame>(referenceKeyframe);
    frameToTrack->setReferenceKeyframe(referenceKeyframe);

     /*
     *      Threads Definition
     */


    // get images from camera
    xpcf::DropBuffer< SRef<Image> >  workingBufferCamImages;
    std::function<void(void)> getCameraImages = [&stop,camera,&workingBufferCamImages](){

        SRef<Image> view;
        if (camera->getNextImage(view) == SolAR::FrameworkReturnCode::_ERROR_LOAD_IMAGE) {
            stop = true;
            return;
        }
        if(workingBufferCamImages.empty())
             workingBufferCamImages.push(view);
    };

    // extract key points
    xpcf::DropBuffer< std::pair< SRef<Image>,std::vector<SRef<Keypoint>> > > outBufferKeypoints;
    std::function<void(void)> getKeyPoints = [&workingBufferCamImages,keypointsDetector,&outBufferKeypoints](){

        SRef<Image> camImage ;
        if (!workingBufferCamImages.tryPop(camImage))
            return;
        std::vector< SRef<Keypoint>> kp;
        keypointsDetector->detect(camImage, kp);
        if(outBufferKeypoints.empty())
            outBufferKeypoints.push(std::make_pair(camImage,kp));
    };

    // compute descriptors
    xpcf::DropBuffer< SRef<Frame > > outBufferDescriptors;
    std::function<void(void)> getDescriptors = [&referenceKeyframe,&outBufferKeypoints,descriptorExtractor,&outBufferDescriptors](){

        std::pair<SRef<Image>, std::vector<SRef<Keypoint> > > kp ;
        SRef<DescriptorBuffer> camDescriptors;
        SRef<Frame> frame;

        if (!outBufferKeypoints.tryPop(kp)) {
            return;
        }
        descriptorExtractor->extract(kp.first, kp.second, camDescriptors);
        frame=xpcf::utils::make_shared<Frame>(kp.second,camDescriptors,kp.first);

        if(outBufferDescriptors.empty())
            outBufferDescriptors.push(frame) ;
    };

    //xpcf::SharedBuffer< std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  > >  outBufferTriangulation(1);
    xpcf::DropBuffer< std::tuple<SRef<Keyframe>, SRef<Keyframe>, std::vector<DescriptorMatch>, std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  > >  outBufferTriangulation;

    // A new keyFrame has been detected :
    // - the triangulation has been performed
    // - the Map is updated accordingly
    //
    std::function<void(void)> mapUpdate = [&keyFrameDetectionOn,&referenceKeyframe, &map, &mapper, &mapFilter, &keyframePoses, &outBufferTriangulation, &kfRetriever, &frameToTrack](){
        //std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  >   element;
        std::tuple<SRef<Keyframe>, SRef<Keyframe>, std::vector<DescriptorMatch>, std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  >   element;

        if (!outBufferTriangulation.tryPop(element)) {
            return;
        }
        //SRef<Frame>                                         newFrame;
        SRef<Keyframe>                                      newKeyframe,refKeyframe;
        std::vector<DescriptorMatch>                        foundMatches, remainingMatches;
        std::vector<SRef<CloudPoint>>                       newCloud;

        newKeyframe=std::get<0>(element);
        refKeyframe=std::get<1>(element);
        foundMatches=std::get<2>(element);
        remainingMatches=std::get<3>(element);
        newCloud=std::get<4>(element);

        std::vector<SRef<CloudPoint>>                       filteredCloud;

        LOG_DEBUG(" frame pose estimation :\n {}", newKeyframe->getPose().matrix());
        LOG_DEBUG("Number of matches: {}, number of 3D points:{}", remainingMatches.size(), newCloud.size());
        //newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
        mapFilter->filter(refKeyframe->getPose(), newKeyframe->getPose(), newCloud, filteredCloud);
        mapper->update(map, newKeyframe, filteredCloud, foundMatches, remainingMatches);
        referenceKeyframe = newKeyframe;
        frameToTrack = xpcf::utils::make_shared<Frame>(referenceKeyframe);
        frameToTrack->setReferenceKeyframe(referenceKeyframe);
        kfRetriever->addKeyframe(referenceKeyframe); // add keyframe for reloc
        keyframePoses.push_back(newKeyframe->getPose());
        LOG_DEBUG(" cloud current size: {} \n", map->getPointCloud()->size());

        keyFrameDetectionOn = true;					// re - allow keyframe detection

    };

    xpcf::DropBuffer< std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch> > >  keyFrameBuffer;

    // A new keyFrame has been detected :
    // - perform triangulation
    // - the resulting cloud will be used to update the Map
    //
    std::function<void(void)> doTriangulation = [&triangulator,&keyFrameBuffer,&outBufferTriangulation](){
        std::tuple<SRef<Frame>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch> > element;
        SRef<Frame>                                         newFrame;
        SRef<Keyframe>                                      newKeyframe;
        SRef<Keyframe>                                      refKeyFrame;
        std::vector<DescriptorMatch>                        foundMatches;
        std::vector<DescriptorMatch>                        remainingMatches;
        std::vector<SRef<CloudPoint>>                       newCloud;


        LOG_DEBUG ("**************************   doTriangulation In");

        if (!keyFrameBuffer.tryPop(element) ){
            return;
        }

        newFrame=std::get<0>(element);
        refKeyFrame=std::get<1>(element);
        foundMatches=std::get<2>(element);
        remainingMatches=std::get<3>(element);

        newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
        triangulator->triangulate(newKeyframe, remainingMatches, newCloud);
        //triangulator->triangulate(refKeyFrame->getKeypoints(), newFrame->getKeypoints(), remainingMatches,std::make_pair<int,int>((int)refKeyFrame->m_idx+0,(int)(refKeyFrame->m_idx+1)),
        //                    refKeyFrame->getPose(), newFrame->getPose(), newCloud);
        if(outBufferTriangulation.empty())
            outBufferTriangulation.push(std::make_tuple(newKeyframe,refKeyFrame,foundMatches,remainingMatches,newCloud));
    };

    // Processing of input frames :
    // - perform match+PnP
    // - test if current frame can promoted to a keyFrame.
    // - in that case, push it in the output buffer to be processed by the triangulation thread
    //
    xpcf::DropBuffer< SRef<Image> > displayMatches;   // matches images should be displayed in the main thread
    xpcf::DropBuffer< SRef<Keyframe>> keyframeRelocBuffer;
    bool isLostTrack = false;
    std::function<void(void)> processFrames = [&displayMatches,&keyFrameDetectionOn,&outBufferTriangulation,mapUpdate,&keyframeSelector, &matchesOverlayBlue,&matchesOverlayRed,&imageViewer,&framePoses,&outBufferDescriptors,matcher,matchesFilter,corr2D3DFinder,PnP,&referenceKeyframe, &lastPose,&keyFrameBuffer, &kfRetriever, &keyframeRelocBuffer, &isLostTrack, &frameToTrack](){

         SRef<Frame> newFrame;
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

         if (isLostTrack && !outBufferTriangulation.empty() && !keyFrameBuffer.empty())
             return;

         // test if a triangulation has been performed on a previously keyframe candidate
         if(!outBufferTriangulation.empty()){
             //if so update the map
             mapUpdate();
         }

         /*compute matches between reference image and camera image*/
         newFrame=outBufferDescriptors.pop();

         // referenceKeyframe can be changed outside : let's make a copy.
         if (!keyframeRelocBuffer.empty()) {
             referenceKeyframe = keyframeRelocBuffer.pop();
             frameToTrack = xpcf::utils::make_shared<Frame>(referenceKeyframe);
             frameToTrack->setReferenceKeyframe(referenceKeyframe);
             lastPose = referenceKeyframe->getPose();
         }

         newFrame->setReferenceKeyframe(referenceKeyframe);
         refKeyFrame = newFrame->getReferenceKeyframe();


         view=newFrame->getView();
         keypoints=newFrame->getKeypoints();
         descriptors=newFrame->getDescriptors();


         refDescriptors= frameToTrack->getDescriptors();
         matcher->match(refDescriptors, descriptors, matches);

         /* filter matches to remove redundancy and check geometric validity */
         matchesFilter->filter(matches, matches, frameToTrack->getKeypoints(), keypoints);

         corr2D3DFinder->find(frameToTrack, newFrame, matches, foundPoints, pt3d, pt2d, foundMatches, remainingMatches);

         if (isLostTrack)
             imageViewer->display(view);
         else {
             SRef<Image> imageMatches, imageMatches2;
             matchesOverlayBlue->draw(view, imageMatches, refKeyFrame->getKeypoints(), keypoints, foundMatches);
             matchesOverlayRed->draw(imageMatches, imageMatches2, refKeyFrame->getKeypoints(), keypoints, remainingMatches);
             if (displayMatches.empty())
                 displayMatches.push(imageMatches2);
         }

         if (PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, newFramePose , lastPose) == FrameworkReturnCode::_SUCCESS){
             LOG_DEBUG(" frame pose  :\n {}", newFramePose.matrix());
             LOG_DEBUG(" pnp inliers size: {} / {}",worldPoints_inliers.size(), pt3d.size());

           lastPose = newFramePose;

           // update new frame
           newFrame->setPose(newFramePose);
           // update last frame
           frameToTrack = newFrame;

             // If the camera has moved enough, create a keyframe and map the scene
           if (keyFrameDetectionOn && keyframeSelector->select(newFrame, foundMatches))
           {
               keyFrameDetectionOn=false;
               keyFrameBuffer.push(std::make_tuple(newFrame,refKeyFrame,foundMatches,remainingMatches));
            }
             else{
                 LOG_DEBUG (" No valid pose was found");
                 framePoses.push_back(newFramePose); // used for display
                 LOG_DEBUG(" framePoses current size: {} \n", framePoses.size());

             }

           isLostTrack = false;
         }
         else {
             // reloc
             isLostTrack = true;
             LOG_INFO("Pose estimation has failed");
             // reloc
             std::vector < SRef <Keyframe>> ret_keyframes;
             if (kfRetriever->retrieve(newFrame, ret_keyframes) == FrameworkReturnCode::_SUCCESS) {
                 keyframeRelocBuffer.push(ret_keyframes[0]);
                 LOG_INFO("Retrieval Success");
             }
             else
                 LOG_INFO("Retrieval Failed");
         }
    };

    xpcf::DelegateTask taskGetCameraImages(getCameraImages);
    xpcf::DelegateTask taskGetKeyPoints(getKeyPoints);
    xpcf::DelegateTask taskGetDescriptors(getDescriptors);
    xpcf::DelegateTask taskProcessFrames(processFrames);
    xpcf::DelegateTask taskDoTriangulation(doTriangulation);
    xpcf::DelegateTask taskMapUpdate(mapUpdate);

    stop = false;

    taskGetCameraImages.start();
    taskGetKeyPoints.start();
    taskGetDescriptors.start();
    taskDoTriangulation.start();
    taskProcessFrames.start();


    // running loop process

    clock_t start, end;
    int count = 0;
    start = clock();

    keyFrameDetectionOn=true;

    while(!stop){
        if (!displayMatches.empty()) {
            if (imageViewer->display(displayMatches.pop()) == SolAR::FrameworkReturnCode::_STOP)
                break;
        }
        count++;
        if (viewer3DPoints->display(*(map->getPointCloud()), lastPose, keyframePoses, framePoses) == FrameworkReturnCode::_STOP){
               stop=true;
        }
    }

    // display stats on frame rate
    end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    printf("\n\nElasped time is %.2lf seconds.\n", duration);
    printf("Number of processed frame per second : %8.2f\n", count / duration);


    std::cout << "end of processes \n";

    taskGetCameraImages.stop();
    taskGetKeyPoints.stop();
    taskGetDescriptors.stop();
    taskDoTriangulation.stop();
    taskProcessFrames.stop();

    return 0;

}


