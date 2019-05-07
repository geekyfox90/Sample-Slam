#ifndef PIPELINESLAM_H
#define PIPELINESLAM_H

#if _WIN32
#ifdef PipelineSlam_API_DLLEXPORT
#define SOLARPIPELINESLAM_EXPORT_API __declspec(dllexport)
#else //SOLARPIPELINEFIDUCIALMARKER_API_DLLEXPORT
#define SOLARPIPELINESLAM_EXPORT_API __declspec(dllimport)
#endif //SOLARPIPELINEFIDUCIALMARKER_API_DLLEXPORT
#else //_WIN32
#define SOLARPIPELINESLAM_EXPORT_API
#endif //_WIN32

#include "xpcf/core/traits.h"
#include "xpcf/component/ConfigurableBase.h"
#include "api/pipeline/IPipeline.h"

// Add the headers to datastructures and component interfaces used by the pipeline

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
#include "api/solver/pose/I3DTransformSACFinderFrom2D3D.h"
#include "api/features/IMatchesFilter.h"
#include "api/display/I2DOverlay.h"
#include "api/display/IMatchesOverlay.h"
#include "api/display/I3DOverlay.h"
#include "api/display/IImageViewer.h"
#include "api/display/I3DPointsViewer.h"
#include "api/reloc/IKeyframeRetriever.h"

#include "api/input/files/IMarker2DNaturalImage.h"
#include "api/solver/pose/I2DTransformFinder.h"
#include "api/solver/pose/IHomographyValidation.h"
#include "api/features/IKeypointsReIndexer.h"
#include "api/geom/IImage2WorldMapper.h"
#include "api/geom/I2DTransform.h"

#include "api/input/files/IMarker2DSquaredBinary.h"
#include "api/image/IImageFilter.h"
#include "api/image/IImageConvertor.h"
#include "api/features/IContoursExtractor.h"
#include "api/features/IContoursFilter.h"
#include "api/image/IPerspectiveController.h"
#include "api/features/IDescriptorsExtractorSBPattern.h"
#include "api/features/ISBPatternReIndexer.h"

#include "api/solver/pose/I2DTransformFinder.h"
#include "api/solver/pose/IHomographyValidation.h"
#include "api/features/IKeypointsReIndexer.h"
#include "api/geom/IImage2WorldMapper.h"
#include "api/geom/I2DTransform.h"
#include "api/geom/IProject.h"
#include "api/features/IMatchesFilter.h"

#include "api/solver/map/IBundler.h"


#ifdef USE_OPENGL
    #include "api/sink/ISinkPoseTextureBuffer.h"
#else
    #include "api/sink/ISinkPoseImage.h"
#endif

#include "xpcf/threading/SharedBuffer.h"
#include "xpcf/threading/DropBuffer.h"
#include "xpcf/threading/BaseTask.h"


#include "SolAR2DOverlayOpencv.h"

#include <set>

namespace SolAR {
using namespace datastructure;
using namespace api;
using namespace api::sink;
using namespace api::pipeline;
namespace PIPELINES {

class SOLARPIPELINESLAM_EXPORT_API PipelineSlam : public org::bcom::xpcf::ConfigurableBase,
    public api::pipeline::IPipeline
{
public:
    PipelineSlam();
    ~PipelineSlam();

    //// @brief Initialization of the pipeline
    /// Initialize the pipeline by providing a reference to the component manager loaded by the PipelineManager.
    /// @param[in] componentManager a shared reference to the component manager which has loaded the components and configuration in the pipleine manager
    FrameworkReturnCode init(SRef<xpcf::IComponentManager> xpcfComponentManager) override;

    /// @brief Provide the camera parameters
    /// @return the camera parameters (its resolution and its focal)
    CameraParameters getCameraParameters() override;

    /// @brief Starts the pipeline and provides a texture buffer which will be updated when required.
    /// @param[in] textureHandle a pointer to the texture buffer which will be updated at each call of the update method.
#ifdef USE_OPENGL
    FrameworkReturnCode start(void* textureHandle) override;
#else
    FrameworkReturnCode start(void* imageDataBuffer) override;
#endif

    /// @brief Stop the pipeline.
    FrameworkReturnCode stop() override;

    /// @brief update the pipeline
    /// Get the new pose and update the texture buffer with the image that has to be displayed
    SinkReturnCode update(Transform3Df& pose) override;

    SourceReturnCode loadSourceImage(void* sourceTextureHandle, int width, int height) override;

    void unloadComponent () override final;

private:

    SRef<input::devices::ICamera> m_camera;
    SRef<features::IKeypointDetector> m_keypointsDetector;
    SRef<features::IDescriptorsExtractor> m_descriptorExtractor;
    SRef<features::IDescriptorMatcher> m_matcher;
    SRef<solver::pose::I3DTransformFinderFrom2D2D> m_poseFinderFrom2D2D;
	SRef<solver::map::ITriangulator>  m_triangulator;
    SRef<solver::pose::I3DTransformFinderFrom2D3D> m_PnP_FIM;
    SRef<solver::pose::I3DTransformSACFinderFrom2D3D> m_PnP;
    SRef<solver::pose::I2D3DCorrespondencesFinder> m_corr2D3DFinder;
    SRef<geom::IProject> m_projector;
    SRef<solver::map::IMapFilter> m_mapFilter;
    SRef<solver::map::IMapper> m_mapper;
    SRef<solver::map::IKeyframeSelector> m_keyframeSelector;
    SRef<reloc::IKeyframeRetriever> m_kfRetriever;

	
	
	
    SRef<input::files::IMarker2DNaturalImage> m_naturalImagemarker;
    SRef<features::IMatchesFilter> m_basicMatchesFilter;
    SRef<features::IMatchesFilter> m_geomMatchesFilter;
    SRef<solver::pose::I2DTransformFinder> m_homographyEstimation ;
    SRef<solver::pose::IHomographyValidation> m_homographyValidation ;
    SRef<features::IKeypointsReIndexer> m_keypointsReindexer;
    SRef<solver::pose::I3DTransformFinderFrom2D3D> m_poseEstimation;
    SRef<geom::IImage2WorldMapper> m_img_mapper;
    SRef<geom::I2DTransform> m_transform2D;

    SRef<DescriptorBuffer> m_markerPatternDescriptor;
    SRef<input::files::IMarker2DSquaredBinary> m_binaryMarker;
    SRef<image::IImageFilter> m_imageFilterBinary;
    SRef<image::IImageConvertor> m_imageConvertor;
    SRef<features::IContoursExtractor> m_contoursExtractor ;
    SRef<features::IContoursFilter> m_contoursFilter;
    SRef<image::IPerspectiveController> m_perspectiveController;
    SRef<features::IDescriptorsExtractorSBPattern> m_patternDescriptorExtractor;
    SRef<features::IDescriptorMatcher> m_patternMatcher;
    SRef<features::ISBPatternReIndexer> m_patternReIndexer;
    SRef<geom::IImage2WorldMapper> m_img2worldMapper;


    SRef<api::solver::map::IBundler> m_bundler;

    // display stuff
    SRef<api::display::I2DOverlay> m_i2DOverlay;


#ifdef USE_OPENGL
    SRef<sink::ISinkPoseTextureBuffer> m_sink;
#else
    SRef<sink::ISinkPoseImage> m_sink;
#endif

    // State flag of the pipeline
    bool m_stopFlag, m_initOK, m_startedOK;


//    bool m_FiducialMarkerDetected;
    bool m_bootstrapOk;
    bool m_firstImageCaptured;
    SRef<Image> m_firstImage;
    Transform3Df m_firstPose;

    SRef<Map>                                           m_map;
    Transform3Df                                        poseFrame1;
    SRef<Keyframe>                                      keyframe1;
    std::vector<SRef<Keypoint>>                         keypointsView1;
    SRef<DescriptorBuffer>                              descriptorsView1;


    SRef<Keyframe>                                      m_referenceKeyframe;
    SRef<Frame>											m_frameToTrack;
    Transform3Df                                        m_lastPose;
    std::vector<Transform3Df>                           m_keyframePoses;
    bool                                                m_keyFrameDetectionOn;   // if true, keyFrames can be detected
    bool                                                m_isLostTrack;

    std::vector<SRef<Keyframe>>                         m_keyFrames;
    std::map<SRef<Keyframe>,std::map<SRef<Keyframe>,int>> m_connectivityMap;
    void addToConnectivityMap(std::vector<SRef<CloudPoint>>& cloudPoints,int currentKFId);
    SRef<Keyframe> selectReferenceKeyFrame(std::vector<std::pair<SRef<Keyframe>,int>>& tab);

    bool detectFiducialMarkerCore(SRef<SolAR::datastructure::Image>& image);
    // Threads

    void getCameraImages();
    void detectFiducialMarker();
    void doBootStrap();
    void getKeyPoints();
    void getDescriptors();
    void mapUpdate();
    void doTriangulation();
    void processFrames();

    void allTasks();

    void project3Dpoints(const Transform3Df pose,const std::vector<SRef<CloudPoint>>& cloud,std::vector<SRef<Point2Df>>& point2D);
    void project3Dpoints(const Transform3Df pose,const std::set<SRef<CloudPoint>>& cloud,std::vector<SRef<Point2Df>>& point2D,std::vector<SRef<Point3Df>>& point3D);
    void filterCloud(const Transform3Df pose1, const Transform3Df pose2, const std::vector<SRef<CloudPoint>>& input,  std::vector<int>& output);
    bool accepMatch(const SRef<Keyframe> kf,const SRef<CloudPoint> cloudPoint,const SRef<DescriptorBuffer>& descriptors,std::vector<int>& indices,std::vector<float>& dists);
    bool doLocalBundleAdjustment();

    void computeConnectedMatches(SRef<Frame> newFrame,
                                 std::vector<SRef<CloudPoint>>& foundPoints,
                                 std::vector<SRef<Point2Df>> pt2d,
                                 std::vector<SRef<Point3Df>> pt3d,
                                 std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                 std::map<SRef<Keyframe>,int>& map,
                                 std::vector<std::pair<SRef<Keyframe>,int>>& tab);

    void computeConnectedMatches2(SRef<Frame> newFrame,
                                  std::vector<SRef<CloudPoint>>& foundPoints,
                                  std::vector<SRef<Point2Df>> pt2d,
                                  std::vector<SRef<Point3Df>> pt3d,
                                  std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                  std::map<SRef<Keyframe>,int>& map,
                                  std::vector<std::pair<SRef<Keyframe>,int>>& tab);

    void computeConnectedMatches3(SRef<Frame> newFrame,
                                 std::vector<SRef<CloudPoint>>& foundPoints,
                                 std::vector<SRef<Point2Df>> pt2d,
                                 std::vector<SRef<Point3Df>> pt3d,
                                 std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>>& newMatches,
                                 std::map<SRef<Keyframe>,int>& map,
                                 std::vector<std::pair<SRef<Keyframe>,int>>& tab);

    std::pair<double,int> getReprojectionError(SRef<Keyframe> keyFrame, bool fromCloud=false);
    std::pair<double,int> getReprojectionErrorFull(std::vector<int>& selectedKeyframes, bool fromCloud=false);

    xpcf::DelegateTask* m_taskAll;

    xpcf::DelegateTask* m_taskGetCameraImages;
    xpcf::DelegateTask* m_taskDetectFiducialMarker;
    xpcf::DelegateTask* m_taskDoBootStrap;
    xpcf::DelegateTask* m_taskGetKeyPoints;
    xpcf::DelegateTask* m_taskGetDescriptors;
    xpcf::DelegateTask* m_taskProcessFrames;
    xpcf::DelegateTask* m_taskDoTriangulation;
    xpcf::DelegateTask* m_taskMapUpdate;

    xpcf::DropBuffer< SRef<Image> >  m_CameraImagesBuffer;
    xpcf::DropBuffer< std::pair< SRef<Image>,std::vector<SRef<Keypoint>> > > m_outBufferKeypoints;
    xpcf::DropBuffer< SRef<Frame > > m_outBufferDescriptors;

    xpcf::DropBuffer< std::tuple<SRef<Keyframe>,
                      SRef<Keyframe>,
                      std::vector<DescriptorMatch>,
                      std::vector<DescriptorMatch>,
                      std::vector<SRef<CloudPoint>>,
                      std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>> > >  m_outBufferTriangulation;

    xpcf::DropBuffer< std::tuple<SRef<Keyframe>,
                      SRef<Keyframe>,
                      std::vector<DescriptorMatch>,
                      std::vector<DescriptorMatch>,
                      std::vector<std::tuple<SRef<CloudPoint>,SRef<Keyframe>,int>> > >  m_keyFrameBuffer;

    xpcf::DropBuffer< SRef<Image> > m_displayMatches;   // matches images should be displayed in the main thread
    xpcf::DropBuffer< SRef<Keyframe>> m_keyframeRelocBuffer;


    Transform3Df m_pose;

    int m_minThreshold = -1;
    int m_maxThreshold = 220;
    int m_nbTestedThreshold = 6;


};

}//namespace PIPELINES
}//namespace SolAR

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::PIPELINES::PipelineSlam,
                             "577ccd2c-de1b-402a-8829-496747598588",
                             "PipelineSlam",
                             "A pipeline to estimate the pose based on a Slam");
#endif // PIPELINESLAM_H
