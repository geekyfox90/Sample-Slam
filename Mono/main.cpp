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

int main(int argc, char **argv) {

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
	auto camera = xpcfComponentManager->create<SolARCameraOpencv>()->bindTo<input::devices::ICamera>();
#endif
#ifdef USE_FREE
	auto keypointsDetector = xpcfComponentManager->create<SolARKeypointDetectorOpencv>()->bindTo<features::IKeypointDetector>();
	auto descriptorExtractor = xpcfComponentManager->create<SolARDescriptorsExtractorAKAZE2Opencv>()->bindTo<features::IDescriptorsExtractor>();
#else
	auto  keypointsDetector = xpcfComponentManager->create<SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
	auto descriptorExtractor = xpcfComponentManager->create<SolARDescriptorsExtractorSURF64Opencv>()->bindTo<features::IDescriptorsExtractor>();
#endif


	//   auto descriptorExtractorORB =xpcfComponentManager->create<SolARDescriptorsExtractorORBOpencv>()->bindTo<features::IDescriptorsExtractor>();
	SRef<features::IDescriptorMatcher> matcher = xpcfComponentManager->create<SolARDescriptorMatcherKNNOpencv>()->bindTo<features::IDescriptorMatcher>();
	SRef<solver::pose::I3DTransformFinderFrom2D2D> poseFinderFrom2D2D = xpcfComponentManager->create<SolARPoseFinderFrom2D2DOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D2D>();
	SRef<solver::map::ITriangulator> triangulator = xpcfComponentManager->create<SolARSVDTriangulationOpencv>()->bindTo<solver::map::ITriangulator>();
	SRef<features::IMatchesFilter> matchesFilter = xpcfComponentManager->create<SolARGeometricMatchesFilterOpencv>()->bindTo<features::IMatchesFilter>();
	SRef<solver::pose::I3DTransformFinderFrom2D3D> PnP = xpcfComponentManager->create<SolARPoseEstimationPnpOpencv>()->bindTo<solver::pose::I3DTransformFinderFrom2D3D>();
	SRef<solver::pose::I2D3DCorrespondencesFinder> corr2D3DFinder = xpcfComponentManager->create<SolAR2D3DCorrespondencesFinderOpencv>()->bindTo<solver::pose::I2D3DCorrespondencesFinder>();
	SRef<solver::map::IMapFilter> mapFilter = xpcfComponentManager->create<SolARMapFilter>()->bindTo<solver::map::IMapFilter>();
	SRef<solver::map::IMapper> mapper = xpcfComponentManager->create<SolARMapper>()->bindTo<solver::map::IMapper>();
	SRef<solver::map::IKeyframeSelector> keyframeSelector = xpcfComponentManager->create<SolARKeyframeSelector>()->bindTo<solver::map::IKeyframeSelector>();

	SRef<display::IMatchesOverlay> matchesOverlay = xpcfComponentManager->create<SolARMatchesOverlayOpencv>()->bindTo<display::IMatchesOverlay>();
	SRef<display::IMatchesOverlay> matchesOverlayBlue = xpcfComponentManager->create<SolARMatchesOverlayOpencv>("matchesBlue")->bindTo<display::IMatchesOverlay>();
	SRef<display::IMatchesOverlay> matchesOverlayRed = xpcfComponentManager->create<SolARMatchesOverlayOpencv>("matchesRed")->bindTo<display::IMatchesOverlay>();

	SRef<display::IImageViewer> imageViewer = xpcfComponentManager->create<SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();
	SRef<display::I3DPointsViewer> viewer3DPoints = xpcfComponentManager->create<SolAR3DPointsViewerOpengl>()->bindTo<display::I3DPointsViewer>();

	// KeyframeRetriever component to relocalize
	SRef<reloc::IKeyframeRetriever> kfRetriever = xpcfComponentManager->create<SolARKeyframeRetrieverFBOW>()->bindTo<reloc::IKeyframeRetriever>();

	/* in dynamic mode, we need to check that components are well created*/
	/* this is needed in dynamic mode */

	if (!camera || !keypointsDetector || !descriptorExtractor || !descriptorExtractor || !matcher ||
		!poseFinderFrom2D2D || !triangulator || !mapFilter || !mapper || !keyframeSelector || !PnP ||
		!corr2D3DFinder || !matchesFilter || !matchesOverlay || !imageViewer || !viewer3DPoints)
	{
		LOG_ERROR("One or more component creations have failed");
		return -1;
	}

	// declarations
	SRef<Image>                                         view1, view2, view;
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
	SRef<Map>                                           map;

	bool												isLostTrack = false;

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

	bool bootstrapOk = false;
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
		if (imageViewer->display(imageMatches) == SolAR::FrameworkReturnCode::_STOP)
			return 1;

		if (keyframeSelector->select(frame2, matches))
		{
			// Estimate the pose of of the second frame (the first frame being the reference of our coordinate system)
			poseFinderFrom2D2D->estimate(keypointsView1, keypointsView2, poseFrame1, poseFrame2, matches);
			LOG_INFO("Nb matches for triangulation: {}\\{}", matches.size(), nbOriginalMatches);
			LOG_INFO("Estimate pose of the camera for the frame 2: \n {}", poseFrame2.matrix());
			frame2->setPose(poseFrame2);

			// Triangulate
			keyframe2 = xpcf::utils::make_shared<Keyframe>(frame2);
			triangulator->triangulate(keyframe2, matches, cloud);
			//double reproj_error = triangulator->triangulate(keypointsView1, keypointsView2, matches, std::make_pair(0, 1), poseFrame1, poseFrame2, cloud);
			mapFilter->filter(poseFrame1, poseFrame2, cloud, filteredCloud);
			keyframePoses.push_back(poseFrame2); // used for display			
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

	// Start tracking
    clock_t start, end;
    int count = 0;
    start = clock();

	while (true)
	{
		// Get current image
        camera->getNextImage(view);
        count++;
		keypointsDetector->detect(view, keypoints);
		descriptorExtractor->extract(view, keypoints, descriptors);
		newFrame = xpcf::utils::make_shared<Frame>(keypoints, descriptors, view, referenceKeyframe);
		// match current keypoints with the keypoints of the Keyframe
		SRef<DescriptorBuffer> frameToTrackDescriptors = frameToTrack->getDescriptors();
		matcher->match(frameToTrackDescriptors, descriptors, matches);		
		matchesFilter->filter(matches, matches, frameToTrack->getKeypoints(), keypoints);

		std::vector<SRef<Point2Df>> pt2d;
		std::vector<SRef<Point3Df>> pt3d;
		std::vector<SRef<CloudPoint>> foundPoints;
		std::vector<DescriptorMatch> foundMatches;
		std::vector<DescriptorMatch> remainingMatches;
		corr2D3DFinder->find(frameToTrack, newFrame, matches, foundPoints, pt3d, pt2d, foundMatches, remainingMatches);
		//LOG_INFO("found matches {}, Remaining Matches {}", foundMatches.size(), remainingMatches.size());
		// display matches
		if (isLostTrack) {
			if (imageViewer->display(view) == FrameworkReturnCode::_STOP)
                break;
		}
		else {
			matchesOverlayBlue->draw(view, imageMatches, referenceKeyframe->getKeypoints(), keypoints, foundMatches);
			matchesOverlayRed->draw(imageMatches, imageMatches2, referenceKeyframe->getKeypoints(), keypoints, remainingMatches);
			if (imageViewer->display(imageMatches2) == FrameworkReturnCode::_STOP)
                break;
		}		

		std::vector<SRef<Point2Df>> imagePoints_inliers;
		std::vector<SRef<Point3Df>> worldPoints_inliers;
		if (PnP->estimate(pt2d, pt3d, imagePoints_inliers, worldPoints_inliers, newFramePose, lastPose) == FrameworkReturnCode::_SUCCESS) {
			LOG_INFO(" pnp inliers size: {} / {}", worldPoints_inliers.size(), pt3d.size());
			lastPose = newFramePose;

			// Set the pose of the new frame
			newFrame->setPose(newFramePose);

			// update frame to track
			frameToTrack = newFrame;
			
			// If the camera has moved enough, create a keyframe and map the scene
			if (keyframeSelector->select(newFrame, foundMatches))
			//if ((nbFrameTracking == 0) && (foundMatches.size() < 0.7 * referenceKeyframe->getVisibleMapPoints().size()))
			{			
				// create a new keyframe from the current frame
				newKeyframe = xpcf::utils::make_shared<Keyframe>(newFrame);
				// triangulate with the reference keyframe
				std::vector<SRef<CloudPoint>>newCloud, filteredCloud;
				triangulator->triangulate(newKeyframe, remainingMatches, newCloud);
				//triangulator->triangulate(referenceKeyframe->getKeypoints(), keypoints, remainingMatches, std::make_pair<int, int>((int)referenceKeyframe->m_idx + 0, (int)(mapper->getNbKeyframes())),
				//	referenceKeyframe->getPose(), newFramePose, newCloud);

				// remove abnormal 3D points from the new cloud
				mapFilter->filter(referenceKeyframe->getPose(), newFramePose, newCloud, filteredCloud);
				LOG_DEBUG("Number of matches: {}, number of 3D points:{}", remainingMatches.size(), filteredCloud.size());

				//  Add new keyframe with the cloud to the mapper
				mapper->update(map, newKeyframe, filteredCloud, remainingMatches, foundMatches);
				keyframePoses.push_back(newKeyframe->getPose());
				referenceKeyframe = newKeyframe;
				frameToTrack = xpcf::utils::make_shared<Frame>(referenceKeyframe);
				frameToTrack->setReferenceKeyframe(referenceKeyframe);
				kfRetriever->addKeyframe(referenceKeyframe); // add keyframe for reloc
				//LOG_INFO("************************ NEW KEYFRAME *************************");
				LOG_DEBUG(" cloud current size: {} \n", map->getPointCloud()->size());
			}
			else
			{
				framePoses.push_back(newFramePose); // used for display
			}

			isLostTrack = false;	// tracking is good

		}
		else {
			LOG_INFO("Pose estimation has failed");
			isLostTrack = true;		// lost tracking

			// reloc
			std::vector < SRef <Keyframe>> ret_keyframes;
			if (kfRetriever->retrieve(newFrame, ret_keyframes) == FrameworkReturnCode::_SUCCESS) {
				referenceKeyframe = ret_keyframes[0];
				frameToTrack = xpcf::utils::make_shared<Frame>(referenceKeyframe);
				frameToTrack->setReferenceKeyframe(referenceKeyframe);
				lastPose = referenceKeyframe->getPose();
				LOG_INFO("Retrieval Success");
			}
			else
				LOG_INFO("Retrieval Failed");
		}

		// display point cloud
		if (viewer3DPoints->display(*(map->getPointCloud()), lastPose, keyframePoses, framePoses) == FrameworkReturnCode::_STOP)
            break;

	}

    // display stats on frame rate
    end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    printf("\n\nElasped time is %.2lf seconds.\n", duration);
    printf("Number of processed frame per second : %8.2f\n", count / duration);

	return 0;
}



