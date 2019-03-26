#include "xpcf/module/ModuleFactory.h"
#include "PipeLineSlam.h"

#include "SolARModuleOpencv_traits.h"
#include "SolARModuleTools_traits.h"
#include "SolARModuleFBOW_traits.h"
#include "SolARModuleCeres_traits.h"
#include "core/Log.h"
#include "opencv2/flann/miniflann.hpp"

#define USE_FREE
#define ONE_THREAD 1

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::CERES;
using namespace SolAR::MODULES::FBOW;
#ifndef USE_FREE
using namespace SolAR::MODULES::NONFREEOPENCV;
#endif
using namespace SolAR::MODULES::TOOLS;

namespace xpcf = org::bcom::xpcf;


#define VIDEO_INPUT


// Declaration of the module embedding the Slam pipeline
XPCF_DECLARE_MODULE("da89a6eb-3233-4dea-afdc-9d918be0bd74", "SlamModule", "The module embedding a pipeline to estimate the pose based on a multithreaded Slam")

extern "C" XPCF_MODULEHOOKS_API xpcf::XPCFErrorCode XPCF_getComponent(const boost::uuids::uuid& componentUUID,SRef<xpcf::IComponentIntrospect>& interfaceRef)
{
    xpcf::XPCFErrorCode errCode = xpcf::XPCFErrorCode::_FAIL;
    errCode = xpcf::tryCreateComponent<SolAR::PIPELINES::PipelineSlam>(componentUUID,interfaceRef);

    return errCode;
}

XPCF_BEGIN_COMPONENTS_DECLARATION
XPCF_ADD_COMPONENT(SolAR::PIPELINES::PipelineSlam)
XPCF_END_COMPONENTS_DECLARATION

// The pipeline component for the fiducial marker

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::PIPELINES::PipelineSlam)

namespace SolAR {
using namespace datastructure;
using namespace api::pipeline;
namespace PIPELINES {


PipelineSlam::PipelineSlam():ConfigurableBase(xpcf::toUUID<PipelineSlam>())
{
   addInterface<api::pipeline::IPipeline>(this);
    LOG_DEBUG(" Pipeline constructor");
    m_bootstrapOk=false;
    m_firstImageCaptured = false;
    m_keyFrameDetectionOn = true;
    m_isLostTrack = false;
}


PipelineSlam::~PipelineSlam()
{
     LOG_DEBUG(" Pipeline destructor")
}

FrameworkReturnCode PipelineSlam::init(SRef<xpcf::IComponentManager> xpcfComponentManager)
{
    // component creation
//   #ifdef USE_IMAGES_SET
//       m_camera = xpcfComponentManager->create<MODULES::OPENCV::SolARImagesAsCameraOpencv>()->bindTo<input::devices::ICamera>();
    #ifdef VIDEO_INPUT
       m_camera = xpcfComponentManager->create<MODULES::OPENCV::SolARVideoAsCameraOpencv>()->bindTo<input::devices::ICamera>();
   #else
       m_camera =xpcfComponentManager->create<MODULES::OPENCV::SolARCameraOpencv>()->bindTo<input::devices::ICamera>();
   #endif

   #ifdef USE_FREE
       m_keypointsDetector =xpcfComponentManager->create<MODULES::OPENCV::SolARKeypointDetectorOpencv>()->bindTo<features::IKeypointDetector>();
       m_descriptorExtractor =xpcfComponentManager->create<MODULES::OPENCV::SolARDescriptorsExtractorAKAZE2Opencv>()->bindTo<features::IDescriptorsExtractor>();
   #else
      m_keypointsDetector = xpcfComponentManager->create<MODULES::OPENCV::SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
      m_descriptorExtractor = xpcfComponentManager->create<MODULES::OPENCV::SolARDescriptorsExtractorSURF64Opencv>()->bindTo<features::IDescriptorsExtractor>();
   #endif

    m_matcher =xpcfComponentManager->create<MODULES::OPENCV::SolARDescriptorMatcherKNNOpencv>()->bindTo<features::IDescriptorMatcher>();
    m_poseFinderFrom2D2D =xpcfComponentManager->create<MODULES::OPENCV::SolARPoseFinderFrom2D2DOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D2D>();
    m_triangulator =xpcfComponentManager->create<MODULES::OPENCV::SolARSVDTriangulationOpencv>()->bindTo<solver::map::ITriangulator>();
    m_basicMatchesFilter = xpcfComponentManager->create<SolARBasicMatchesFilter>()->bindTo<features::IMatchesFilter>();
    m_geomMatchesFilter =xpcfComponentManager->create<MODULES::OPENCV::SolARGeometricMatchesFilterOpencv>()->bindTo<features::IMatchesFilter>();
    m_PnP_FIM =xpcfComponentManager->create<MODULES::OPENCV::SolARPoseEstimationPnpOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D3D>();
    m_PnP =xpcfComponentManager->create<MODULES::OPENCV::SolARPoseEstimationSACPnpOpencv>()->bindTo<solver::pose::I3DTransformSACFinderFrom2D3D>();
    m_corr2D3DFinder =xpcfComponentManager->create<MODULES::OPENCV::SolAR2D3DCorrespondencesFinderOpencv>()->bindTo<solver::pose::I2D3DCorrespondencesFinder>();
    m_mapFilter =xpcfComponentManager->create<MODULES::TOOLS::SolARMapFilter>()->bindTo<solver::map::IMapFilter>();
    m_mapper =xpcfComponentManager->create<MODULES::TOOLS::SolARMapper>()->bindTo<solver::map::IMapper>();
    m_keyframeSelector =xpcfComponentManager->create<MODULES::TOOLS::SolARKeyframeSelector>()->bindTo<solver::map::IKeyframeSelector>();
    m_kfRetriever = xpcfComponentManager->create<MODULES::FBOW::SolARKeyframeRetrieverFBOW>()->bindTo<reloc::IKeyframeRetriever>();
    m_bundler = xpcfComponentManager->create<MODULES::CERES::SolARBundlerCeres>()->bindTo<api::solver::map::IBundler>();

#ifdef USE_OPENGL
    m_sink = xpcfComponentManager->create<MODULES::OPENGL::SinkPoseTextureBuffer>()->bindTo<sink::ISinkPoseTextureBuffer>();
#else
    m_sink = xpcfComponentManager->create<MODULES::TOOLS::SolARBasicSink>()->bindTo<sink::ISinkPoseImage>();
#endif
    if (m_sink)
        LOG_INFO("Pose Texture Buffer Sink component loaded");

    if ( m_camera==nullptr || m_keypointsDetector==nullptr || m_descriptorExtractor==nullptr || m_matcher==nullptr ||
        m_poseFinderFrom2D2D==nullptr || m_triangulator==nullptr || m_mapFilter==nullptr || m_mapper==nullptr || m_keyframeSelector==nullptr || m_PnP==nullptr ||
        m_corr2D3DFinder==nullptr || m_geomMatchesFilter==nullptr || m_basicMatchesFilter==nullptr || m_kfRetriever==nullptr || m_sink==nullptr)
    {
       LOG_ERROR("One or more component creations have failed");
       return FrameworkReturnCode::_ERROR_  ;
    }

    // init relative to fiducial marker detection (will define the start of the process)
    m_binaryMarker =xpcfComponentManager->create<MODULES::OPENCV::SolARMarker2DSquaredBinaryOpencv>()->bindTo<input::files::IMarker2DSquaredBinary>();
    m_imageFilterBinary =xpcfComponentManager->create<MODULES::OPENCV::SolARImageFilterBinaryOpencv>()->bindTo<image::IImageFilter>();
    m_imageConvertor =xpcfComponentManager->create<MODULES::OPENCV::SolARImageConvertorOpencv>()->bindTo<image::IImageConvertor>();
    m_contoursExtractor =xpcfComponentManager->create<MODULES::OPENCV::SolARContoursExtractorOpencv>()->bindTo<features::IContoursExtractor>();
    m_contoursFilter =xpcfComponentManager->create<MODULES::OPENCV::SolARContoursFilterBinaryMarkerOpencv>()->bindTo<features::IContoursFilter>();
    m_perspectiveController =xpcfComponentManager->create<MODULES::OPENCV::SolARPerspectiveControllerOpencv>()->bindTo<image::IPerspectiveController>();
    m_patternDescriptorExtractor =xpcfComponentManager->create<MODULES::OPENCV::SolARDescriptorsExtractorSBPatternOpencv>()->bindTo<features::IDescriptorsExtractorSBPattern>();
    m_patternMatcher =xpcfComponentManager->create<MODULES::OPENCV::SolARDescriptorMatcherRadiusOpencv>()->bindTo<features::IDescriptorMatcher>();
    m_patternReIndexer = xpcfComponentManager->create<MODULES::TOOLS::SolARSBPatternReIndexer>()->bindTo<features::ISBPatternReIndexer>();
    m_img2worldMapper = xpcfComponentManager->create<MODULES::TOOLS::SolARImage2WorldMapper4Marker2D>()->bindTo<geom::IImage2WorldMapper>();

    // load marker
    LOG_INFO("LOAD MARKER IMAGE ");
   if( m_binaryMarker->loadMarker()==FrameworkReturnCode::_ERROR_){
       return FrameworkReturnCode::_ERROR_;
    }
    LOG_INFO("MARKER IMAGE LOADED");

    m_patternDescriptorExtractor->extract(m_binaryMarker->getPattern(), m_markerPatternDescriptor);
    LOG_INFO ("Marker pattern:\n {}", m_binaryMarker->getPattern()->getPatternMatrix())

    int patternSize = m_binaryMarker->getPattern()->getSize();

    m_patternDescriptorExtractor->bindTo<xpcf::IConfigurable>()->getProperty("patternSize")->setIntegerValue(patternSize);
    m_patternReIndexer->bindTo<xpcf::IConfigurable>()->getProperty("sbPatternSize")->setIntegerValue(patternSize);

    m_img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("digitalWidth")->setIntegerValue(patternSize);
    m_img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("digitalHeight")->setIntegerValue(patternSize);
    m_img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("worldWidth")->setFloatingValue(m_binaryMarker->getSize().width);
    m_img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("worldHeight")->setFloatingValue(m_binaryMarker->getSize().height);



    // specific to the
    // initialize pose estimation with the camera intrinsic parameters (please refeer to the use of intrinsec parameters file)
    m_PnP_FIM->setCameraParameters(m_camera->getIntrinsicsParameters(), m_camera->getDistorsionParameters());
    m_PnP->setCameraParameters(m_camera->getIntrinsicsParameters(), m_camera->getDistorsionParameters());
    m_poseFinderFrom2D2D->setCameraParameters(m_camera->getIntrinsicsParameters(), m_camera->getDistorsionParameters());
    m_triangulator->setCameraParameters(m_camera->getIntrinsicsParameters(), m_camera->getDistorsionParameters());


    m_i2DOverlay = xpcf::ComponentFactory::createInstance<SolAR2DOverlayOpencv>()->bindTo<api::display::I2DOverlay>();
    m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("radius")->setUnsignedIntegerValue(1);

    m_initOK = true;

    return FrameworkReturnCode::_SUCCESS;
}


// get images from camera

void PipelineSlam::getCameraImages(){

    SRef<Image> view;
    if (m_stopFlag || !m_initOK || !m_startedOK)
        return;
    if (m_camera->getNextImage(view) == SolAR::FrameworkReturnCode::_ERROR_LOAD_IMAGE) {
        m_stopFlag = true;
        return;
    }
    if(m_CameraImagesBuffer.empty())
         m_CameraImagesBuffer.push(view);

    return;
};



bool PipelineSlam::detectFiducialMarkerCore(SRef<Image>& image)
{
    SRef<Image>                     greyImage, binaryImage;
    std::vector<SRef<Contour2Df>>   contours;
    std::vector<SRef<Contour2Df>>   filtered_contours;
    std::vector<SRef<Image>>        patches;
    std::vector<SRef<Contour2Df>>   recognizedContours;
    SRef<DescriptorBuffer>          recognizedPatternsDescriptors;
    std::vector<DescriptorMatch>    patternMatches;
    std::vector<SRef<Point2Df>>     pattern2DPoints;
    std::vector<SRef<Point2Df>>     img2DPoints;
    std::vector<SRef<Point3Df>>     pattern3DPoints;


    bool poseComputed = false;

   // Convert Image from RGB to grey
    m_imageConvertor->convert(image, greyImage, Image::ImageLayout::LAYOUT_GREY);

    for (int num_threshold = 0; !poseComputed && num_threshold < m_nbTestedThreshold; num_threshold++)
    {
         // Compute the current Threshold valu for image binarization
         int threshold = m_minThreshold + (m_maxThreshold - m_minThreshold)*((float)num_threshold/(float)(m_nbTestedThreshold-1));

         // Convert Image from grey to black and white
         m_imageFilterBinary->bindTo<xpcf::IConfigurable>()->getProperty("min")->setIntegerValue(threshold);
         m_imageFilterBinary->bindTo<xpcf::IConfigurable>()->getProperty("max")->setIntegerValue(255);

        // Convert Image from grey to black and white
        m_imageFilterBinary->filter(greyImage, binaryImage);

        // Extract contours from binary image
        m_contoursExtractor->extract(binaryImage, contours);

         // Filter 4 edges contours to find those candidate for marker contours
        m_contoursFilter->filter(contours, filtered_contours);

        // Create one warpped and cropped image by contour
        m_perspectiveController->correct(binaryImage, filtered_contours, patches);

        // test if this last image is really a squared binary marker, and if it is the case, extract its descriptor
        if (m_patternDescriptorExtractor->extract(patches, filtered_contours, recognizedPatternsDescriptors, recognizedContours) != FrameworkReturnCode::_ERROR_)
        {
            // From extracted squared binary pattern, match the one corresponding to the squared binary marker
            if (m_patternMatcher->match(m_markerPatternDescriptor, recognizedPatternsDescriptors, patternMatches) == features::DescriptorMatcher::DESCRIPTORS_MATCHER_OK)
            {
                // Reindex the pattern to create two vector of points, the first one corresponding to marker corner, the second one corresponding to the poitsn of the contour
                m_patternReIndexer->reindex(recognizedContours, patternMatches, pattern2DPoints, img2DPoints);

                // Compute the 3D position of each corner of the marker
                m_img2worldMapper->map(pattern2DPoints, pattern3DPoints);

                // Compute the pose of the camera using a Perspective n Points algorithm using only the 4 corners of the marker
                if (m_PnP_FIM->estimate(img2DPoints, pattern3DPoints, m_pose) == FrameworkReturnCode::_SUCCESS)
                {
                    poseComputed = true;
                }
            }
        }
    }

    return poseComputed;

}


void PipelineSlam::detectFiducialMarker()
{
    SRef<Image>                     camImage, greyImage, binaryImage;
    std::vector<SRef<Contour2Df>>   contours;
    std::vector<SRef<Contour2Df>>   filtered_contours;
    std::vector<SRef<Image>>        patches;
    std::vector<SRef<Contour2Df>>   recognizedContours;
    SRef<DescriptorBuffer>          recognizedPatternsDescriptors;
    std::vector<DescriptorMatch>    patternMatches;
    std::vector<SRef<Point2Df>>     pattern2DPoints;
    std::vector<SRef<Point2Df>>     img2DPoints;
    std::vector<SRef<Point3Df>>     pattern3DPoints;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;
    if(m_firstImageCaptured){
        return ;
    }

    if (!m_CameraImagesBuffer.tryPop(camImage))
        return  ;

    if ( detectFiducialMarkerCore(camImage)){
        m_sink->set(m_pose, camImage);
        poseFrame1=m_pose;
        m_keypointsDetector->detect(camImage, keypointsView1);
        m_descriptorExtractor->extract(camImage, keypointsView1, descriptorsView1);
        keyframe1 = xpcf::utils::make_shared<Keyframe>(keypointsView1, descriptorsView1, camImage, poseFrame1);
        m_keyFrames.push_back(keyframe1);
        m_mapper->update(m_map, keyframe1);
        m_kfRetriever->addKeyframe(keyframe1); // add keyframe for reloc
        m_firstImageCaptured = true;
    }
    else
        m_sink->set(camImage);

    return ;
}


void PipelineSlam::doBootStrap()
{
    SRef<Image>  camImage;
    Transform3Df poseFrame2;
    SRef<Keyframe>                                      keyframe2;
    std::vector<SRef<Keypoint>>                         keypointsView2;
    SRef<DescriptorBuffer>                              descriptorsView2;
    std::vector<SRef<CloudPoint>>                       cloud, filteredCloud;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;
    if(!m_firstImageCaptured)
        return ;
    if(m_bootstrapOk)
        return ;

    if (!m_CameraImagesBuffer.tryPop(camImage))
        return  ;

    std::cout << "*";
    if(detectFiducialMarkerCore(camImage)){
        m_sink->set(m_pose, camImage);

        m_keypointsDetector->detect(camImage, keypointsView2);
        m_descriptorExtractor->extract(camImage, keypointsView2, descriptorsView2);

        poseFrame2=m_pose;
        m_matcher->match(descriptorsView1, descriptorsView2, matches);

        int nbOriginalMatches = matches.size();

        /* filter matches to remove redundancy and check geometric validity */
         m_basicMatchesFilter->filter(matches, matches, keypointsView1, keypointsView2);
         m_geomMatchesFilter->filter(matches, matches, keypointsView1, keypointsView2);


        SRef<Frame> frame2 = xpcf::utils::make_shared<Frame>(keypointsView2, descriptorsView2, camImage, keyframe1);
        frame2->setPose(poseFrame2);

        if (m_keyframeSelector->select(frame2, matches)) {

             LOG_INFO("Nb matches for triangulation: {}\\{}", matches.size(), nbOriginalMatches);
             LOG_INFO("Estimate pose of the camera for the frame 2: \n {}", poseFrame2.matrix());

             keyframe2=xpcf::utils::make_shared<Keyframe>(frame2);
             m_keyFrames.push_back(keyframe2);
             std::vector<DescriptorMatch> emptyMatches;
             m_keyFrameBuffer.push(std::make_tuple(keyframe2,keyframe1,emptyMatches,matches));

             m_bootstrapOk = true;
             LOG_INFO("BootStrap is validated \n");
        }

    }
    else {
        m_sink->set(camImage);
    }

    return ;
}

// extract key points
void PipelineSlam::getKeyPoints(){
    SRef<Image>  camImage;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;
    if(!m_bootstrapOk)
        return ;

    if (!m_CameraImagesBuffer.tryPop(camImage))
        return  ;

    std::vector< SRef<Keypoint>> kp;
    m_keypointsDetector->detect(camImage, kp);
    if(m_outBufferKeypoints.empty())
        m_outBufferKeypoints.push(std::make_pair(camImage,kp));

    return;
};

// compute descriptors
void PipelineSlam::getDescriptors(){

    std::pair<SRef<Image>, std::vector<SRef<Keypoint> > > kp ;
    SRef<DescriptorBuffer> camDescriptors;
    SRef<Frame> frame;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;
    if(!m_bootstrapOk)
        return ;

    if (!m_outBufferKeypoints.tryPop(kp)) {
        return ;
    }
    m_descriptorExtractor->extract(kp.first, kp.second, camDescriptors);
    frame=xpcf::utils::make_shared<Frame>(kp.second,camDescriptors,kp.first);

    if(m_outBufferDescriptors.empty())
        m_outBufferDescriptors.push(frame) ;

    return;
};

// A new keyFrame has been detected :
// - the triangulation has been performed
// - the Map is updated accordingly
//
void PipelineSlam::mapUpdate(){
    std::tuple<SRef<Keyframe>, SRef<Keyframe>, std::vector<DescriptorMatch>, std::vector<DescriptorMatch>, std::vector<SRef<CloudPoint>>  >   element;
    std::vector<DescriptorMatch>                        foundMatches, remainingMatches;
    std::vector<SRef<CloudPoint>>                       newCloud;
    std::vector<SRef<CloudPoint>>                       filteredCloud;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;

    if(!m_bootstrapOk)
        return ;

    if (!m_outBufferTriangulation.tryPop(element)) {
        return ;
    }
    SRef<Keyframe>                                      newKeyframe,refKeyframe;

    newKeyframe=std::get<0>(element);
    refKeyframe=std::get<1>(element);
    foundMatches=std::get<2>(element);
    remainingMatches=std::get<3>(element);
    newCloud=std::get<4>(element);


    std::map<unsigned int, unsigned int> visibleKeypoints= newKeyframe->getReferenceKeyframe()->getVisibleKeypoints();

    LOG_DEBUG(" frame pose estimation :\n {}", newKeyframe->getPose().matrix());
    LOG_DEBUG("Number of matches: {}, number of 3D points:{}", remainingMatches.size(), newCloud.size());
    //newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
    m_mapFilter->filter(refKeyframe->getPose(), newKeyframe->getPose(), newCloud, filteredCloud);

    m_mapper->update(m_map, newKeyframe, filteredCloud, foundMatches, remainingMatches);

    addToConnectivityMap(filteredCloud,newKeyframe->m_idx);

    m_referenceKeyframe = newKeyframe;
    m_frameToTrack = xpcf::utils::make_shared<Frame>(m_referenceKeyframe);
    m_frameToTrack->setReferenceKeyframe(m_referenceKeyframe);
    m_kfRetriever->addKeyframe(m_referenceKeyframe); // add keyframe for reloc
    m_keyframePoses.push_back(newKeyframe->getPose());
    LOG_DEBUG(" cloud current size: {} \n", m_map->getPointCloud()->size());


    for (auto kf:m_keyFrames){
            double er1=getReprojectionError(kf);
            std::cout << "kf id : " << kf->m_idx << " reproj err : " << er1 << "\n";
    }

    for (auto kf:m_keyFrames){
            double er1=getReprojectionError(kf,true);
            std::cout << "kf id : " << kf->m_idx << " reproj err : " << er1 << "\n";
    }

    doLocalBundleAdjustment();

    m_keyFrameDetectionOn = true;					// re - allow keyframe detection

    return;

};





// A new keyFrame has been detected :
// - perform triangulation
// - the resulting cloud will be used to update the Map
//
void PipelineSlam::doTriangulation(){
    std::tuple<SRef<Keyframe>,SRef<Keyframe>,std::vector<DescriptorMatch>,std::vector<DescriptorMatch> > element;
    SRef<Frame>                                         newFrame;
    SRef<Keyframe>                                      newKeyframe;
    SRef<Keyframe>                                      refKeyFrame;
    std::vector<DescriptorMatch>                        foundMatches;
    std::vector<DescriptorMatch>                        remainingMatches;
    std::vector<SRef<CloudPoint>>                       newCloud;

    if (m_stopFlag || !m_initOK || !m_startedOK)
        return ;

    if(!m_bootstrapOk)
        return ;


    if (!m_keyFrameBuffer.tryPop(element) ){
        return ;
    }

    newKeyframe=std::get<0>(element);
    refKeyFrame=std::get<1>(element);
    foundMatches=std::get<2>(element);
    remainingMatches=std::get<3>(element);

    if(remainingMatches.size())
            m_triangulator->triangulate(newKeyframe, remainingMatches, newCloud);

    if(m_outBufferTriangulation.empty())
        m_outBufferTriangulation.push(std::make_tuple(newKeyframe,refKeyFrame,foundMatches,remainingMatches,newCloud));

    return;
};


// Processing of input frames :
// - perform match+PnP
// - test if current frame can promoted to a keyFrame.
// - in that case, push it in the output buffer to be processed by the triangulation thread
//



bool sortRefKeyFrames(const std::pair<SRef<Keyframe>,int> &lhs, const std::pair<SRef<Keyframe>,int> &rhs)
{
    return (lhs.second == rhs.second ? lhs.first->m_idx > rhs.first->m_idx :lhs.second > rhs.second );
}

void PipelineSlam::processFrames(){

     SRef<Frame> newFrame;
     SRef<Keyframe> refKeyFrame;
     SRef<Image> camImage;
     std::vector< SRef<Keypoint> > keypoints;
     SRef<DescriptorBuffer> descriptors;
     SRef<DescriptorBuffer> refDescriptors;
     std::vector<DescriptorMatch> matches;

     std::vector<SRef<Point2Df>> pt2d;
     std::vector<SRef<Point3Df>> pt3d;
     std::vector<SRef<CloudPoint>> foundPoints;
     std::vector<DescriptorMatch> foundMatches;
     std::vector<DescriptorMatch> remainingMatches;
     std::vector<SRef<Point2Df>> imagePoints_inliers;
     std::vector<SRef<Point3Df>> worldPoints_inliers;

     std::vector < SRef <Keyframe>> ret_keyframes;

     if (m_stopFlag || !m_initOK || !m_startedOK)
         return ;

     if(!m_bootstrapOk)
         return ;

     if (m_isLostTrack && !m_outBufferTriangulation.empty() && !m_keyFrameBuffer.empty())
         return;

     // test if a triangulation has been performed on a previously keyframe candidate
     if(!m_outBufferTriangulation.empty()){
         //if so update the map
         mapUpdate();
     }

     /*compute matches between reference image and camera image*/
     if(!m_outBufferDescriptors.tryPop(newFrame)){
             return;
     }

     // referenceKeyframe can be changed outside : let's make a copy.
     if (!m_keyframeRelocBuffer.empty()) {
         m_referenceKeyframe = m_keyframeRelocBuffer.pop();
         m_frameToTrack = xpcf::utils::make_shared<Frame>(m_referenceKeyframe);
         m_frameToTrack->setReferenceKeyframe(m_referenceKeyframe);
         m_lastPose = m_referenceKeyframe->getPose();
     }

     newFrame->setReferenceKeyframe(m_referenceKeyframe);
     refKeyFrame = newFrame->getReferenceKeyframe();
     camImage=newFrame->getView();
     keypoints=newFrame->getKeypoints();
     descriptors=newFrame->getDescriptors();

     m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(0,0);
     m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(0,1);
     m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(255,2);
     m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("radius")->setUnsignedIntegerValue(2);
//     m_i2DOverlay->drawCircles(keypoints,camImage);

     refDescriptors= m_frameToTrack->getDescriptors();
     m_matcher->match(refDescriptors, descriptors, matches);

     /* filter matches to remove redundancy and check geometric validity */
     m_basicMatchesFilter->filter(matches, matches, m_frameToTrack->getKeypoints(), keypoints);
     m_geomMatchesFilter->filter(matches, matches, m_frameToTrack->getKeypoints(), keypoints);

     m_corr2D3DFinder->find(m_frameToTrack, newFrame, matches, foundPoints, pt3d, pt2d, foundMatches, remainingMatches);
     LOG_INFO(" cloud:{} #matches:{} #3D:{} #remain matches : {}",m_map->getPointCloud()->size(),matches.size(), pt3d.size(),remainingMatches.size());

     std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>> newMatches;


     if (m_PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, m_pose , m_lastPose) == FrameworkReturnCode::_SUCCESS){

            LOG_INFO(" pnp inliers size: {} / {} ==> {}%",worldPoints_inliers.size(), pt3d.size(),100.f*worldPoints_inliers.size()/pt3d.size());
            std::map<SRef<Keyframe>,int> map = m_connectivityMap[m_referenceKeyframe];

            //try to find additional matches with connected Reference keyframes
            std::vector<std::pair<SRef<Keyframe>,int>> tab;
//            computeConnectedMatches(newFrame, foundPoints,pt2d,pt3d,newMatches, map, tab);
//           computeConnectedMatches2(newFrame, foundPoints,pt2d,pt3d,newMatches, map, tab);
             computeConnectedMatches3(newFrame, foundPoints,pt2d,pt3d,newMatches, map, tab);


        m_lastPose = m_pose;

        // update new frame
        newFrame->setPose(m_pose);
        // update last frame
//        m_frameToTrack = newFrame;

        SRef<Keyframe> bestKF=selectReferenceKeyFrame(tab);

#if 0
        for(auto m:m_connectivityMap){
            auto id0=m.first->m_idx;
            for(auto k:m.second){
                auto id1=k.first->m_idx;
                std::cout << " " << id0  << " " << id1 << " : " << k.second << "\n";
            }
            std::cout << "\n";
        }
        std::cout << " reference key frame # " << m_referenceKeyframe->m_idx << "\n";
//        std::cout << " best key frame # " << bestKF->m_idx << "\n";
        for(auto t:tab){
            if(t.second>0)
                std::cout << " key frame # " << t.first->m_idx << " nb matches : " << t.second << "\n";
        }
        getchar();
#endif
        if(bestKF!=nullptr){
            if(bestKF!=m_referenceKeyframe){
                m_referenceKeyframe = bestKF;
                m_frameToTrack = xpcf::utils::make_shared<Frame>(m_referenceKeyframe);
                m_frameToTrack->setReferenceKeyframe(m_referenceKeyframe);
                m_lastPose = m_referenceKeyframe->getPose();
            }
        }



#if 1
                SRef<std::vector<SRef<CloudPoint>>> cloud;
                std::vector<SRef<Point2Df>> point2D;
                cloud=m_map->getPointCloud();
                project3Dpoints(m_pose, *cloud,point2D);
                m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(0,0);
                m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(255,1);
                m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("color")->setUnsignedIntegerValue(0,2);
                m_i2DOverlay->bindTo<xpcf::IConfigurable>()->getProperty("radius")->setUnsignedIntegerValue(1);
                m_i2DOverlay->drawCircles(point2D,camImage);
#endif

        // If the camera has moved enough, create a keyframe and map the scene

        if ( m_keyFrameDetectionOn &&  (newFrame->getReferenceKeyframe()->getVisibleMapPoints().size()*0.8f > foundMatches.size()) ){
//          if ( m_keyFrameDetectionOn &&  m_keyframeSelector->select(newFrame, foundMatches) ){
            m_keyFrameDetectionOn=false;
            LOG_INFO("New key Frame ")
            SRef<Keyframe> newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
            m_keyFrames.push_back(newKeyframe);
            LOG_INFO("# of keyFrames : {}",m_keyFrames.size());


            // triangulate the remaining matches
            std::vector<SRef<CloudPoint>> newCloud;
            if(remainingMatches.size())
                m_triangulator->triangulate(newKeyframe, remainingMatches, newCloud);

            std::vector<SRef<CloudPoint>> filteredCloud;

            m_mapFilter->filter(newKeyframe->getReferenceKeyframe()->getPose(), newKeyframe->getPose(), newCloud, filteredCloud);

            m_mapper->update(m_map, newKeyframe, filteredCloud, foundMatches, remainingMatches);

            // update connectivity map
            addToConnectivityMap(filteredCloud,newKeyframe->m_idx);

            // the 3D points in common with the connected keyFrames
            std::map<unsigned int, SRef<CloudPoint>> visibility,visibilityKF;
            foundPoints.clear();
            visibilityKF=newKeyframe->getVisibleMapPoints();
            for(auto nm:newMatches){
                SRef<CloudPoint> cp=std::get<0>(nm);
                SRef<Keyframe> kf=std::get<1>(nm);
                int kpId=std::get<2>(nm);
                if(visibility.find(kpId)==visibility.end() && visibilityKF.find(kpId)==visibilityKF.end()){
                    foundPoints.push_back(cp);
                    visibility[kpId]=cp;
                    cp->visibilityAddKeypoint(newKeyframe->m_idx,kpId);
                    (m_connectivityMap[newKeyframe])[kf]++;
                    (m_connectivityMap[kf])[newKeyframe]++;
                }
            }

            newKeyframe->addVisibleMapPoints(visibility);

            m_referenceKeyframe = newKeyframe;
            m_frameToTrack = xpcf::utils::make_shared<Frame>(m_referenceKeyframe);
            m_frameToTrack->setReferenceKeyframe(m_referenceKeyframe);
            m_kfRetriever->addKeyframe(m_referenceKeyframe); // add keyframe for reloc
            m_keyframePoses.push_back(newKeyframe->getPose());

            LOG_DEBUG(" cloud current size: {} \n", m_map->getPointCloud()->size());
            std::cout << "nb clound points :" << m_map->getPointCloud()->size() << "\n";

            for (auto kf:m_keyFrames){
                    double er1=getReprojectionError(kf);
                    double er2=getReprojectionError(kf,true);
                    std::cout << "kf id : " <<  kf->m_idx << "\t" << " reproj err : " << er1 << "\t" << er2 << "\n";
            }

            doLocalBundleAdjustment();

            m_keyFrameDetectionOn = true;					// re - allow keyframe detection

//            getchar();

             // fill the thread liaison buffer
 //           m_keyFrameBuffer.push(std::make_tuple(newKeyframe,refKeyFrame,foundMatches,remainingMatches));
        }
        m_isLostTrack = false;
        m_sink->set(m_pose, camImage);
     }
     else {
         LOG_DEBUG (" No valid pose was found");
         m_sink->set(camImage);
         m_isLostTrack = true;
         if ( m_kfRetriever->retrieve(newFrame, ret_keyframes) == FrameworkReturnCode::_SUCCESS) {
             LOG_INFO("Retrieval Success based on FBOW");
             m_keyframeRelocBuffer.push(ret_keyframes[0]);
             m_isLostTrack = false;
         }
//         else if ( detectFiducialMarkerCore(camImage)){
//              m_lastPose = m_pose;
//              m_isLostTrack = false;
//         }

        else{
             LOG_INFO("Retrieval Failed");
         }
     }

     return;
};

#ifdef USE_OPENGL
FrameworkReturnCode PipelineSlam::start(void* textureHandle)
#else
FrameworkReturnCode PipelineSlam::start(void* imageDataBuffer)
#endif
{
    if (m_initOK==false)
    {
        LOG_WARNING("Try to start the Fiducial marker pipeline without initializing it");
        return FrameworkReturnCode::_ERROR_;
    }
    m_stopFlag=false;
#ifdef USE_OPENGL
    m_sink->setTextureBuffer(textureHandle);
#else
    m_sink->setImageBuffer((unsigned char*)imageDataBuffer);
#endif
    if (m_camera->start() != FrameworkReturnCode::_SUCCESS)
    {
        LOG_ERROR("Camera cannot start")
        return FrameworkReturnCode::_ERROR_;
    }

    // create and start threads
    auto getCameraImagesThread = [this](){;getCameraImages();};
    auto detectFiducialMarkerThread = [this](){;detectFiducialMarker();};
    auto doBootStrapThread = [this](){;doBootStrap();};
    auto getKeyPointsThread = [this](){;getKeyPoints();};
    auto getDescriptorsThread = [this](){;getDescriptors();};
    auto processFramesThread = [this](){;processFrames();};
    auto doTriangulationThread = [this](){;doTriangulation();};
    auto mapUpdateThread = [this](){;mapUpdate();};

#if ONE_THREAD
    auto allTasksThread=[this](){;allTasks();};
    m_taskAll= new xpcf::DelegateTask(allTasksThread);
    m_taskAll->start();
#else
    m_taskGetCameraImages = new xpcf::DelegateTask(getCameraImagesThread);
    m_taskDetectFiducialMarker = new xpcf::DelegateTask(detectFiducialMarkerThread);
    m_taskDoBootStrap = new xpcf::DelegateTask(doBootStrapThread);
    m_taskGetKeyPoints = new xpcf::DelegateTask(getKeyPointsThread);
    m_taskGetDescriptors = new xpcf::DelegateTask(getDescriptorsThread);
    m_taskProcessFrames = new xpcf::DelegateTask(processFramesThread);
    m_taskDoTriangulation = new xpcf::DelegateTask(doTriangulationThread);
    m_taskMapUpdate = new xpcf::DelegateTask(mapUpdateThread);

    m_taskGetCameraImages->start();
    m_taskDetectFiducialMarker->start();
    m_taskDoBootStrap ->start();
    m_taskGetKeyPoints->start();
    m_taskGetDescriptors->start();
    m_taskProcessFrames ->start();
    m_taskDoTriangulation->start();
//    m_taskMapUpdate->start();
#endif

    LOG_INFO("Threads have started");
    m_startedOK = true;

    return FrameworkReturnCode::_SUCCESS;
}

FrameworkReturnCode PipelineSlam::stop()
{
    m_stopFlag=true;
    m_camera->stop();

#if ONE_THREAD
    if (m_taskAll != nullptr)
            m_taskAll->stop();
#else
    if (m_taskGetCameraImages != nullptr)
        m_taskGetCameraImages->stop();
    if (m_taskDetectFiducialMarker != nullptr)
        m_taskDetectFiducialMarker->stop();
    if (m_taskDoBootStrap != nullptr)
        m_taskDoBootStrap->stop();
    if (m_taskGetKeyPoints != nullptr)
        m_taskGetKeyPoints->stop();
    if (m_taskGetDescriptors != nullptr)
        m_taskGetDescriptors->stop();
    if (m_taskProcessFrames != nullptr)
        m_taskProcessFrames->stop();
    if (m_taskDoTriangulation != nullptr)
        m_taskDoTriangulation->stop();
//    if (m_taskMapUpdate != nullptr)
//        m_taskMapUpdate->stop();
#endif

     if(!m_initOK)
     {
         LOG_WARNING("Try to stop a pipeline that has not been initialized");
         return FrameworkReturnCode::_ERROR_;
     }
     if (!m_startedOK)
     {
         LOG_WARNING("Try to stop a pipeline that has not been started");
         return FrameworkReturnCode::_ERROR_;
     }
     LOG_INFO("Pipeline has stopped: \n");

    return FrameworkReturnCode::_SUCCESS;
}

SourceReturnCode PipelineSlam::loadSourceImage(void* sourceTextureHandle, int width, int height)
{
    return SourceReturnCode::_NOT_IMPLEMENTED;
}

SinkReturnCode PipelineSlam::update(Transform3Df& pose)
{
    if(m_stopFlag)
        return SinkReturnCode::_ERROR;
    else
        return m_sink->tryGet(pose);
}

CameraParameters PipelineSlam::getCameraParameters()
{
    CameraParameters camParam;
    if (m_camera)
    {
        Sizei resolution = m_camera->getResolution();
        CamCalibration calib = m_camera->getIntrinsicsParameters();
        camParam.width = resolution.width;
        camParam.height = resolution.height;
        camParam.focalX = calib(0,0);
        camParam.focalY = calib(1,1);
    }
    return camParam;
}


void PipelineSlam::allTasks(){
    getCameraImages();
    detectFiducialMarker();
    doBootStrap();
    getKeyPoints();
    getDescriptors();
    doTriangulation();
    processFrames();
//    mapUpdate();
}


double PipelineSlam::getReprojectionError(SRef<Keyframe> keyFrame, bool fromCloud){
    double r;
    Transform3Df pose=keyFrame->getPose();
    std::vector<SRef<CloudPoint>> cloud;
    std::vector<int>  indices;
    std::vector<SRef<Point2Df>>  point2D;

    if(fromCloud){
        auto pointCloud=*m_map->getPointCloud();
        auto visibility=keyFrame->getVisibleMapPoints();
        for(auto cp:pointCloud){
            std::map<unsigned int, unsigned int> vis=cp->getVisibility();
            auto itr_v = vis.find( keyFrame->m_idx);
            if(itr_v!=vis.end()){
                cloud.push_back(cp);
                indices.push_back(itr_v->second);
                int count=0;
                for(std::map<unsigned int, SRef<CloudPoint>>::iterator v=visibility.begin();v!=visibility.end();++v){
                    if(v->second==cp){
                        count++;
                    }
                }
                if(count==0){
                     std::cout << "cp not found \n";
                }

            }
        }
    }
    else{
        auto visibility=keyFrame->getVisibleMapPoints();
        for(std::map<unsigned int, SRef<CloudPoint>>::iterator v=visibility.begin();v!=visibility.end();++v){
            auto ind=v->first;
            auto cp=v->second;
            indices.push_back(ind);
            cloud.push_back(cp);
        }
    }
    project3Dpoints(pose,cloud,point2D);
    auto keypoints=keyFrame->getKeypoints();
    r=0;
    int i;
    for(i=0;i<point2D.size();++i){

        SRef<Keypoint> kp = keypoints[indices[i]];
        SRef<Point2Df> p2d = point2D[i];

        double dx=p2d->getX()-kp->getX();
        double dy=p2d->getY()-kp->getY();

        r+=dx*dx+dy*dy;
    }
    return sqrt(r/i);
}

bool PipelineSlam::doLocalBundleAdjustment(){

    // get reference keyframe and connected keyframes
    std::vector<SRef<Keyframe>>keyframes;
    std::vector<int>selectedKeyframes;
    std::vector<SRef<CloudPoint>>points3d;
    CamCalibration  intrinsic = m_camera->getIntrinsicsParameters();
    CamDistortion   distorsion = m_camera->getDistorsionParameters();


    std::map<SRef<Keyframe>,int> m=m_connectivityMap[m_referenceKeyframe];
    std::vector<std::pair<SRef<Keyframe>,int>> tab;
    for(auto p:m){
        tab.push_back(std::make_pair(p.first,p.second));
    }

    std::sort(tab.begin(),tab.end(),sortByNumbers);
    int max=tab[0].second;

    auto id0=m_referenceKeyframe->m_idx;
    selectedKeyframes.push_back(id0);
    for(auto k:tab){
        if(k.second<max/4)
            break;
        auto id1=k.first->m_idx;
        std::cout << " " << id0  << " " << id1 << " : " << k.second << "\n";
        selectedKeyframes.push_back(id1);
    }

    double reproj_errorFinal  = 0.f;
    points3d =*m_map->getPointCloud() ;
    reproj_errorFinal = m_bundler->solve(m_keyFrames,
                                       points3d,
                                       intrinsic,
                                       distorsion,
                                       selectedKeyframes);

    std::cout << "reprojection error final: " << reproj_errorFinal << "\n";

    for(auto k:m_keyFrames){
        std::cout << "kf : " << k->m_idx << " reproj error :" << getReprojectionError(k) << "\n";
    }

    return true;
}

bool PipelineSlam::accepMatch(const SRef<Keyframe> kf, const SRef<CloudPoint> cloudPoint,const SRef<DescriptorBuffer>& descriptors,std::vector<int>& indices,std::vector<float>& dists){
    int size=61;
    float distanceRatio=0.75;

    if(dists[0] > 10 || (dists[0]>0.8*dists[1])){
        return false;
    }

    auto v = cloudPoint->getVisibility();
    if(v.size()==0)
        return false;
    auto itr_v = v.find( kf->m_idx);
    auto desc=m_keyFrames[itr_v->first]->getDescriptors();
    auto id0=itr_v->second;
    unsigned char* b0 = (unsigned char*)(desc->data());
    cv::Mat m0(size,1,CV_8U);
    memcpy(m0.data,&b0[size*id0],size);

    int id1=indices[0];
    unsigned char* b1 = (unsigned char*)(descriptors->data());
    cv::Mat m1(size,1,CV_8U);
    memcpy(m1.data,&b1[size*id1],size);

    double dist1=cv::norm(m0,m1,cv::NORM_L2);

    int id2=indices[1];
    unsigned char* b2 = (unsigned char*)(descriptors->data());
    cv::Mat m2(size,1,CV_8U);
    memcpy(m2.data,&b2[size*id2],size);

    double dist2=cv::norm(m0,m2,cv::NORM_L2);

    if(dist1 < distanceRatio * dist2) {
        return true;
    }

    return false;
}

void PipelineSlam::project3Dpoints(const Transform3Df pose,const std::vector<SRef<CloudPoint>>& cloud,std::vector<SRef<Point2Df>>& point2D){

    //first step :  from world coordinates to camera coordinates
    Transform3Df invPose;
    invPose=pose.inverse();
    point2D.clear();
#if (_WIN64) || (_WIN32)
        Vector3f pointInCamRef;
#else
        Vector4f pointInCamRef;
#endif
        CamCalibration calib = m_camera->getIntrinsicsParameters();
        CamDistortion dist = m_camera->getDistorsionParameters();
        float k1 = dist[0];
        float k2 = dist[1];
        float p1 = dist[2];
        float p2 = dist[3];
        float k3 = dist[4];
    for (auto cld = cloud.begin();cld!=cloud.end();++cld){
#if (_WIN64) || (_WIN32)
        Vector3f point((*cld)->getX(), (*cld)->getY(), (*cld)->getZ());
#else
        Vector4f point((*cld)->getX(), (*cld)->getY(), (*cld)->getZ(),1);
#endif
        pointInCamRef=invPose*point;
        if(pointInCamRef(2)>0){
            float x=pointInCamRef(0)/pointInCamRef(2);
            float y=pointInCamRef(1)/pointInCamRef(2);

            float r2 = x * x + y * y;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            float r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            float xx = x * r_coeff + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            float yy = y * r_coeff + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y);

            float u = calib(0,0)*xx +calib(0,2);
            float v = calib(1,1)*yy +calib(1,2);

            SRef<Point2Df> p2d=xpcf::utils::make_shared<Point2Df> (u,v);

            point2D.push_back(p2d);
        }
        else
            point2D.push_back(xpcf::utils::make_shared<Point2Df> (0,0));
    }
}

void PipelineSlam::project3Dpoints(const Transform3Df pose,const std::set<SRef<CloudPoint>>& cloud,std::vector<SRef<Point2Df>>& point2D,std::vector<SRef<Point3Df>>& point3D){

        //first step :  from world coordinates to camera coordinates
        Transform3Df invPose;
        invPose=pose.inverse();
        point2D.clear();
        point3D.clear();
    #if (_WIN64) || (_WIN32)
            Vector3f pointInCamRef;
    #else
            Vector4f pointInCamRef;
    #endif
            CamCalibration calib = m_camera->getIntrinsicsParameters();
            CamDistortion dist = m_camera->getDistorsionParameters();
            float k1 = dist[0];
            float k2 = dist[1];
            float p1 = dist[2];
            float p2 = dist[3];
            float k3 = dist[4];
        for (std::set<SRef<CloudPoint>>::iterator cld=cloud.begin();cld!=cloud.end();cld++){
#if (_WIN64) || (_WIN32)
        Vector3f point((*cld)->getX(), (*cld)->getY(), (*cld)->getZ());
#else
        Vector4f point((*cld)->getX(), (*cld)->getY(), (*cld)->getZ(),1);
#endif
            pointInCamRef=invPose*point;
            if(pointInCamRef(2)>0){
                float x=pointInCamRef(0)/pointInCamRef(2);
                float y=pointInCamRef(1)/pointInCamRef(2);

                float r2 = x * x + y * y;
                float r4 = r2 * r2;
                float r6 = r4 * r2;
                float r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                float xx = x * r_coeff + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                float yy = y * r_coeff + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y);

                float u = calib(0,0)*xx +calib(0,2);
                float v = calib(1,1)*yy +calib(1,2);

                SRef<Point2Df> p2d=xpcf::utils::make_shared<Point2Df> (u,v);

                point2D.push_back(p2d);
                SRef<Point3Df> p3d=xpcf::utils::make_shared<Point3Df>(point(0),point(1),point(2));
                point3D.push_back(p3d);
            }
        }

}



void PipelineSlam::addToConnectivityMap(std::vector<SRef<CloudPoint>>& cloudPoints,int currentKFId){
    int idx0,idx1;
    SRef<Keyframe> keyFrame,currentKeyFrame=m_keyFrames[currentKFId];

    for(auto cp:cloudPoints){
        std::map<unsigned int, unsigned int>visibility=cp->getVisibility();
        std::map<unsigned int, unsigned int>::iterator it=visibility.find(currentKFId);
        if(it==visibility.end())
            continue;
        for(auto m:visibility){
            idx0=m.first;
            keyFrame=m_keyFrames[idx0];
            if(currentKFId!=idx0){
                (m_connectivityMap[currentKeyFrame])[keyFrame]++;
                (m_connectivityMap[keyFrame])[currentKeyFrame]++;
            }
        }
    }
    return;
}


bool sortByNumbers(const std::pair<SRef<Keyframe>,int> &lhs, const std::pair<SRef<Keyframe>,int> &rhs)
{
    return (lhs.second == rhs.second ? lhs.first->m_idx > rhs.first->m_idx:lhs.second > rhs.second );
}

SRef<Keyframe> PipelineSlam::selectReferenceKeyFrame(std::vector<std::pair<SRef<Keyframe>,int>>& tab){

    SRef<Keyframe> refKF;

    if(tab.size()==0)
        return nullptr;

    std::sort(tab.begin(),tab.end(),sortByNumbers);

    refKF=tab[0].first;
    return refKF;
}

void PipelineSlam::computeConnectedMatches(SRef<Frame> newFrame ,
                                           std::vector<SRef<CloudPoint>>& foundPoints,
                                           std::vector<SRef<Point2Df>> pt2d,
                                           std::vector<SRef<Point3Df>> pt3d,
                                           std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                           std::map<SRef<Keyframe>,int>& map,
                                           std::vector<std::pair<SRef<Keyframe>,int>>& tab)
{
   std::vector< SRef<Keypoint> > keypoints=newFrame->getKeypoints();
   SRef<DescriptorBuffer> descriptors=newFrame->getDescriptors();
   tab.push_back(std::make_pair(m_referenceKeyframe,foundPoints.size()));
   for(auto kf:map){

       std::vector<DescriptorMatch> matches_bis;

       std::vector<SRef<Point2Df>> pt2d_bis;
       std::vector<SRef<Point3Df>> pt3d_bis;
       std::vector<SRef<CloudPoint>> foundPoints_bis;
       std::vector<DescriptorMatch> foundMatches_bis;
       std::vector<DescriptorMatch> remainingMatches_bis;

       m_matcher->match(kf.first->getDescriptors(), descriptors, matches_bis);

       /* filter matches to remove redundancy and check geometric validity */

       m_basicMatchesFilter->filter(matches_bis, matches_bis, kf.first->getKeypoints(), keypoints);
       m_geomMatchesFilter->filter(matches_bis, matches_bis, kf.first->getKeypoints(), keypoints);
       m_corr2D3DFinder->find(kf.first, newFrame, matches_bis, foundPoints_bis, pt3d_bis, pt2d_bis, foundMatches_bis, remainingMatches_bis);
       tab.push_back(std::make_pair(kf.first,foundMatches_bis.size()));
       auto cp=foundPoints_bis.begin();
       for(auto fm:foundMatches_bis){
           newMatches.push_back(std::make_tuple(*cp,kf.first,fm.getIndexInDescriptorB())); //record which cloud point with which key point
           cp++;
       }
       pt2d.insert(pt2d.end(),pt2d_bis.begin(),pt2d_bis.end());
       pt3d.insert(pt3d.end(),pt3d_bis.begin(),pt3d_bis.end());
    }

   Transform3Df pose0;
   std::vector<SRef<Point2Df>> imagePoints_inliers;
   std::vector<SRef<Point3Df>> worldPoints_inliers;

   imagePoints_inliers.clear();
   worldPoints_inliers.clear();
   if (m_PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, pose0, m_pose ) == FrameworkReturnCode::_SUCCESS){
       m_pose=pose0;
       LOG_INFO(" pnp inliers size second pass: {} / {} ==> {}%",worldPoints_inliers.size(), pt3d.size(),100.f*worldPoints_inliers.size()/pt3d.size());
   }

 }


void PipelineSlam::computeConnectedMatches2(SRef<Frame> newFrame ,
                                            std::vector<SRef<CloudPoint>>& foundPoints,
                                            std::vector<SRef<Point2Df>> pt2d,
                                            std::vector<SRef<Point3Df>> pt3d,
                                            std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                            std::map<SRef<Keyframe>,int>& map,
                                            std::vector<std::pair<SRef<Keyframe>,int>>& tab)
{
    std::vector< SRef<Keypoint> > keypoints=newFrame->getKeypoints();
    SRef<DescriptorBuffer> descriptors=newFrame->getDescriptors();
    unsigned int w=m_camera->getResolution().width;
    unsigned int h=m_camera->getResolution().height;

   tab.push_back(std::make_pair(m_referenceKeyframe,foundPoints.size()));
   std::vector<cv::Point2f> pointsForSearch;
   for(auto kp:keypoints){
       cv::Point2f p(kp->getX(),kp->getY());
       pointsForSearch.push_back(p);
   }
   for(auto kf:map){
       std::vector<SRef<Point2Df>> point2D;
       std::vector<SRef<Point3Df>> point3D;
       std::set<SRef<CloudPoint>> set1;
       std::map<unsigned int, SRef<CloudPoint>> CpMap;

       CpMap= kf.first->getVisibleMapPoints();
       for(auto cp:CpMap){
           set1.insert(cp.second);
       }
       project3Dpoints(m_pose,set1,point2D,point3D);


       cv::flann::KDTreeIndexParams indexParams;
       cv::flann::Index kdtree(cv::Mat(pointsForSearch).reshape(1), indexParams);
       std::vector<int> indices;
       std::vector<float> dists;
       std::vector<float> query;
       std::vector<SRef<Point3Df>>::iterator it_pt3d=point3D.begin();
       auto cp= set1.begin();
       int count=0;
       for(auto p2d:point2D){
           float x=p2d->getX();
           float y=p2d->getY();
           if(x>0 && x<w && y>0 && y<h){
               query.clear();
               query.push_back(x); //Insert the 2D point we need to find neighbours to the query
               query.push_back(y);
               kdtree.knnSearch(query, indices, dists, 3);
               if(accepMatch(kf.first,*cp,descriptors,indices,dists)){
                   pt2d.push_back(p2d);
                   pt3d.push_back(*it_pt3d);
                   newMatches.push_back(std::make_tuple(*cp,kf.first,indices[0])); //record which cloud point with which key point
                   count++;
               }
           }
           ++cp;
           ++it_pt3d;
       }
       tab.push_back(std::make_pair(kf.first,count));
   }
   Transform3Df pose0;
   std::vector<SRef<Point2Df>> imagePoints_inliers;
   std::vector<SRef<Point3Df>> worldPoints_inliers;

   imagePoints_inliers.clear();
   worldPoints_inliers.clear();
   if (m_PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, pose0, m_pose ) == FrameworkReturnCode::_SUCCESS){
       m_pose=pose0;
       LOG_INFO(" pnp inliers size second pass: {} / {} ==> {}%",worldPoints_inliers.size(), pt3d.size(),100.f*worldPoints_inliers.size()/pt3d.size());
   }

 }

void PipelineSlam::computeConnectedMatches3(SRef<Frame> newFrame ,
                                           std::vector<SRef<CloudPoint>>& foundPoints,
                                           std::vector<SRef<Point2Df>> pt2d,
                                           std::vector<SRef<Point3Df>> pt3d,
                                           std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                           std::map<SRef<Keyframe>,int>& map,
                                           std::vector<std::pair<SRef<Keyframe>,int>>& tab)
{
    std::map<unsigned int, SRef<CloudPoint>> CpMap;
    std::vector<SRef<Point2Df>> point2D;
    std::vector<SRef<Point3Df>> point3D;
    std::vector<cv::Point2f> pointsForSearch;
    cv::flann::KDTreeIndexParams indexParams;
    std::vector<int> indices;
    std::vector<float> dists;
    std::vector<float> query;
 //   std::map<SRef<Keyframe>,int> countMap;
    int count;

    unsigned int w=m_camera->getResolution().width;
    unsigned int h=m_camera->getResolution().height;

    std::vector< SRef<Keypoint> > keypoints=newFrame->getKeypoints();
    SRef<DescriptorBuffer> descriptors=newFrame->getDescriptors();

    for(auto kp:keypoints){
        cv::Point2f p(kp->getX(),kp->getY());
        pointsForSearch.push_back(p);
    }

    cv::flann::Index kdtree(cv::Mat(pointsForSearch).reshape(1), indexParams);

    std::set<SRef<CloudPoint>> set0;       // will contain all the visible cloud points in the reference KF
    std::set<SRef<CloudPoint>> set1;       // will contain all the visible cloud points in the connected KF not visible in the reference keyframe.

    tab.push_back(std::make_pair(m_referenceKeyframe,foundPoints.size()));

    for(auto kf:map){
        CpMap= kf.first->getVisibleMapPoints();
        set1.clear();
        count=0;
        for(auto cp:CpMap){
              SRef<CloudPoint> c=cp.second;
              set1.insert(c);
        }

        project3Dpoints(m_pose,set1,point2D,point3D);
        std::vector<SRef<Point3Df>>::iterator it_pt3d=point3D.begin();
        std::set<SRef<CloudPoint>>::iterator it_set1=set1.begin();
        for(auto p2d:point2D){
            float x=p2d->getX();
            float y=p2d->getY();

            if(x<0 || x>w || y<0 || y>h){
                ++it_pt3d;
                ++it_set1;
                continue;
            }
            query.clear();
            query.push_back(x);
            query.push_back(y);
            kdtree.knnSearch(query, indices, dists, 3);
            if(accepMatch(kf.first,*it_set1,descriptors,indices,dists)){
                pt2d.push_back(p2d);
                pt3d.push_back(*it_pt3d);
                count++;
//                            newMatches.push_back(std::make_pair(*it_set1,indices[0])); //record which cloud point with which key point
                newMatches.push_back(std::make_tuple(*it_set1,kf.first,indices[0])); //record which cloud point with which key point
            }
            ++it_pt3d;
            ++it_set1;
        }
        tab.push_back(std::make_pair(kf.first,count));
    }

    Transform3Df pose0;
    std::vector<SRef<Point2Df>> imagePoints_inliers;
    std::vector<SRef<Point3Df>> worldPoints_inliers;

    imagePoints_inliers.clear();
    worldPoints_inliers.clear();
    if (m_PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, pose0, m_pose ) == FrameworkReturnCode::_SUCCESS){
        m_pose=pose0;
        LOG_INFO(" pnp inliers size second pass: {} / {} ==> {}%",worldPoints_inliers.size(), pt3d.size(),100.f*worldPoints_inliers.size()/pt3d.size());
    }
 }


void  PipelineSlam::filterCloud(const Transform3Df pose1, const Transform3Df pose2, const std::vector<SRef<CloudPoint>>& input,  std::vector<int>& output)
{
    if (input.size() == 0)
    {
        LOG_INFO("mapFilter opencv has an empty vector as input");
    }

    output.clear();

    Transform3Df invPose1, invPose2;
    invPose1=pose1.inverse();
    invPose2=pose2.inverse();

    for (int i = 0; i < input.size(); i++)
    {
        // Check for cheirality (if the point is in front of the camera)

        // BUG patch To correct, Vector4f should but is not accepted with windows !
#if (_WIN64) || (_WIN32)
        Vector3f point(input[i]->getX(), input[i]->getY(), input[i]->getZ());
        Vector3f pointInCam1Ref, pointInCam2Ref;
#else
        Vector4f point(input[i]->getX(), input[i]->getY(), input[i]->getZ(), 1);
        Vector4f pointInCam1Ref, pointInCam2Ref;
#endif
        pointInCam1Ref = invPose1*point;
        pointInCam2Ref = invPose2*point;

        if (((pointInCam1Ref(2) >= 0) && pointInCam2Ref(2) >=0))
        {
            // if the reprojection error is less than the threshold
            if (input[i]->getReprojError() < 2.5)
                output.push_back(i);
        }
    }

}
}//namespace PIPELINES
}//namespace SolAR


