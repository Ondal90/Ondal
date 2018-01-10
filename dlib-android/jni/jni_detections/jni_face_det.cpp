/*
 * jni_pedestrian_det.cpp using google-style
 *
 *  Created on: Oct 20, 2015
 *      Author: Tzutalin
 *
 *  Copyright (c) 2015 Tzutalin. All rights reserved.
 */
#include <android/bitmap.h>
#include <jni_common/jni_bitmap2mat.h>
#include <jni_common/jni_primitives.h>
#include <jni_common/jni_fileutils.h>
#include <jni_common/jni_utils.h>
#include <detector.h>
#include <dlib/image_processing/correlation_tracker.h>
#include <jni.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/image_processing.h>
#include <sys/time.h>
#define MAX_TRACKER_NUM 10
//////////////////////////
dlib::correlation_tracker tracker;
dlib::array2d<dlib::rgb_pixel> dlibImage;
std::string mLandMarkModel;
std::vector<dlib::correlation_tracker> tracker_vector;
dlib::correlation_tracker *tracker_array[MAX_TRACKER_NUM];
//cv::Mat bgrMat;
cv::Mat rgbaMat;
cv::Mat grayMat;
//cv::Mat bgrMat2;
timeval t1, t2,t3,t4,t5,t6,t7;
dlib::full_object_detection shape_track;
dlib::rectangle expected_rect;
dlib::shape_predictor msp;
// Output rotation and translation
cv::Mat rotation_vector; // Rotation in axis-angle form
cv::Mat translation_vector;


double elapsedTime;

using namespace cv;

extern JNI_VisionDetRet* g_pJNI_VisionDetRet;

namespace {

#define JAVA_NULL 0
using DetectorPtr = DLibHOGFaceDetector*;

class JNI_FaceDet {
 public:
  JNI_FaceDet(JNIEnv* env) {
    jclass clazz = env->FindClass(CLASSNAME_FACE_DET);
    mNativeContext = env->GetFieldID(clazz, "mNativeFaceDetContext", "J");
    env->DeleteLocalRef(clazz);
  }

  DetectorPtr getDetectorPtrFromJava(JNIEnv* env, jobject thiz) {
    DetectorPtr const p = (DetectorPtr)env->GetLongField(thiz, mNativeContext);
    return p;
  }

  void setDetectorPtrToJava(JNIEnv* env, jobject thiz, jlong ptr) {
    env->SetLongField(thiz, mNativeContext, ptr);
  }

  jfieldID mNativeContext;
};

// Protect getting/setting and creating/deleting pointer between java/native
std::mutex gLock;

std::shared_ptr<JNI_FaceDet> getJNI_FaceDet(JNIEnv* env) {
  static std::once_flag sOnceInitflag;
  static std::shared_ptr<JNI_FaceDet> sJNI_FaceDet;
  std::call_once(sOnceInitflag, [env]() {
    sJNI_FaceDet = std::make_shared<JNI_FaceDet>(env);
  });
  return sJNI_FaceDet;
}

DetectorPtr const getDetectorPtr(JNIEnv* env, jobject thiz) {
  std::lock_guard<std::mutex> lock(gLock);
  return getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
}

// The function to set a pointer to java and delete it if newPtr is empty
void setDetectorPtr(JNIEnv* env, jobject thiz, DetectorPtr newPtr) {
  std::lock_guard<std::mutex> lock(gLock);
  DetectorPtr oldPtr = getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
  if (oldPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager delete old ptr : " << oldPtr;
    delete oldPtr;
  }

  if (newPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager set new ptr : " << newPtr;
  }

  getJNI_FaceDet(env)->setDetectorPtrToJava(env, thiz, (jlong)newPtr);
}

}  // end unnamespace

#ifdef __cplusplus
extern "C" {
#endif


#define DLIB_FACE_JNI_METHOD(METHOD_NAME) \
  Java_com_tzutalin_dlib_FaceDet_##METHOD_NAME

void JNIEXPORT
    DLIB_FACE_JNI_METHOD(jniNativeClassInit)(JNIEnv* env, jclass _this) {}

jobjectArray getDetectResult(JNIEnv* env, DetectorPtr faceDetector,
                                 const int& size) {
    LOG(INFO) << "getFaceRet";
    jobjectArray jDetRetArray = JNI_VisionDetRet::createJObjectArray(env, size);
    for (int i = 0; i < size; i++) {
        jobject jDetRet = JNI_VisionDetRet::createJObject(env);
        env->SetObjectArrayElement(jDetRetArray, i, jDetRet);
        dlib::rectangle rect = faceDetector->getResult()[i];
        g_pJNI_VisionDetRet->setRect(env, jDetRet, rect.left(), rect.top(),
                                         rect.right(), rect.bottom());
        g_pJNI_VisionDetRet->setLabel(env, jDetRet, "face");
        std::unordered_map<int, dlib::full_object_detection>& faceShapeMap =
        faceDetector->getFaceShapeMap();
        if (faceShapeMap.find(i) != faceShapeMap.end()) {
            dlib::full_object_detection shape = faceShapeMap[i];
            for (unsigned long j = 0; j < shape.num_parts(); j++) {
                int x = shape.part(j).x();
                int y = shape.part(j).y();
                // Call addLandmark
                g_pJNI_VisionDetRet->addLandmark(env, jDetRet, x, y);
            }
        }
    }
    return jDetRetArray;
}

//JNIEXPORT jobjectArray JNICALL
//    DLIB_FACE_JNI_METHOD(jniDetect)(JNIEnv* env, jobject thiz,
//                                    jstring imgPath) {
//  LOG(INFO) << "jniFaceDet";
//  const char* img_path = env->GetStringUTFChars(imgPath, 0);
//  DetectorPtr detPtr = getDetectorPtr(env, thiz);
//  int size = detPtr->det(std::string(img_path));
//  env->ReleaseStringUTFChars(imgPath, img_path);
//  LOG(INFO) << "det face size: " << size;
//  return getDetectResult(env, detPtr, size);
//}

JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniBitmapDetect)(JNIEnv* env, jobject thiz,
                                          jobject bitmap) {
  LOG(INFO) << "jniBitmapFaceDet";
  jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
  cv::cvtColor(rgbaMat, grayMat, cv::COLOR_RGBA2GRAY);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  jint size = detPtr->det(grayMat);
  LOG(INFO) << "det face size: " << size;
  return getDetectResult(env, detPtr, size);;
}

jint JNIEXPORT JNICALL DLIB_FACE_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz,
                                                     jstring jLandmarkPath) {
  LOG(INFO) << "jniInit";
  std::string landmarkPath = jniutils::convertJStrToString(env, jLandmarkPath);
  DetectorPtr detPtr = new DLibHOGFaceDetector(landmarkPath);
  setDetectorPtr(env, thiz, detPtr);
    
    for(int i=0;i<MAX_TRACKER_NUM;i++)
        tracker_array[i] = nullptr;
  return JNI_OK;
}

jint JNIEXPORT JNICALL
    DLIB_FACE_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
  LOG(INFO) << "jniDeInit";
  setDetectorPtr(env, thiz, JAVA_NULL);
  return JNI_OK;
}
    
jint JNIEXPORT JNICALL
    Java_com_tzutalin_dlib_FaceDet_trackerInit(JNIEnv* env, jobject thiz) {
        LOG(INFO) << "jniFaceTracker";
        for (int i=0;i<MAX_TRACKER_NUM;i++){
            if(tracker_array[i]==nullptr){
                LOG(INFO) << "jniFaceTrackerInit" << i;
                tracker_array[i] = new dlib::correlation_tracker();
                return i;
        }
    }
    return -1;
}

void JNIEXPORT JNICALL
Java_com_tzutalin_dlib_FaceDet_trackerStart(JNIEnv* env, jobject thiz, jint i ,jint trackerID) {
    LOG(INFO) << "jniFaceTracker Start";
    gettimeofday(&t1, NULL);
    DetectorPtr detPtr = getDetectorPtr(env, thiz);
    dlib::rectangle rect = detPtr->getResult()[i];
    tracker_array[trackerID]->start_track(dlibImage, dlib::centered_rect(dlib::point((rect.left()+rect.right())/2,
                                                                        (rect.top()+rect.bottom())/2), (rect.right()-rect.left()) , (rect.bottom()-rect.top()) ));
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    LOG(INFO) << elapsedTime << " ms tracker Start.\n";
    LOG(INFO) << "start complete";
}

JNIEXPORT jfloat JNICALL
    Java_com_tzutalin_dlib_FaceDet_trackerUpdate(JNIEnv* env, jobject thiz, jobject bitmap, jint trackerID) {
        DetectorPtr detPtr = getDetectorPtr(env, thiz);
        LOG(INFO) << "jniFaceTrackerUpdate";

        jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
        dlib::assign_image(dlibImage, dlib::cv_image<dlib::rgb_alpha_pixel>(rgbaMat));

        // start timer
        gettimeofday(&t1, NULL);
        float score = tracker_array[trackerID]->update(dlibImage, tracker_array[trackerID]->get_position());
        gettimeofday(&t2, NULL);
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
        LOG(INFO) << elapsedTime << " ms tracking.\n";
        
        return score;
}


jint JNIEXPORT JNICALL
Java_com_tzutalin_dlib_FaceDet_exceptROIdetect(JNIEnv* env, jobject thiz, jobject bitmap) {
    jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
    cv::cvtColor(rgbaMat, grayMat, cv::COLOR_RGBA2GRAY);
    DetectorPtr detPtr = getDetectorPtr(env, thiz);
    LOG(INFO) << "Eliminate ROI";
    
    // 트래킹 영역 제거
    for (int i=0;i<MAX_TRACKER_NUM;i++)
        if(tracker_array[i]!=nullptr) {
            expected_rect = tracker_array[i]->get_position();
            dlib::rectangle expected_rect = tracker_array[i]->get_position();
            for (int x = std::max(long(0), expected_rect.left()); x < std::min(expected_rect.right(),long(grayMat.cols)); x++) {
                for (int y = std::max(long(0),expected_rect.top()); y < std::min(expected_rect.bottom(),long(grayMat.rows)); y++) {
                    grayMat.at<uchar>(y, x) = 255;
                }
            }
        }
    // 트래킹 영역이 제거된 부분에 대해서만 얼굴 검출
    jint size = detPtr->det(grayMat);
    return size;
}
    

//트래킹 위치 추출
JNIEXPORT jobject JNICALL
    Java_com_tzutalin_dlib_FaceDet_trackerGetPosition(JNIEnv* env, jobject thiz, jint trackerID){
    LOG(INFO) << "jniFaceTrackerGetPosition";
    jobject jDetRet = JNI_VisionDetRet::createJObject(env);
    expected_rect = tracker_array[trackerID]->get_position();
    g_pJNI_VisionDetRet->setRect(env, jDetRet, expected_rect.left(), expected_rect.top(),
                                     expected_rect.right(), expected_rect.bottom());
    g_pJNI_VisionDetRet->setLabel(env, jDetRet, "face");
    shape_track = msp(dlibImage, expected_rect);
//    image_points.clear();
//    image_points.front();
        std::vector<cv::Point2d> image_points;
        // 3D model points.
        std::vector<cv::Point3d> model_points;
    LOG(INFO) << "face index:number of parts: " << shape_track.num_parts();
        for (unsigned long k = 0; k < shape_track.num_parts(); k++) {
            if (shape_track.num_parts() !=0){
                int x = shape_track.part(k).x();
                int y = shape_track.part(k).y();
                 //Call addLandmark
                g_pJNI_VisionDetRet->addLandmark(env, jDetRet, x, y);
                if( k== 33){
                    image_points.push_back( cv::Point2d(x, y) );    // Nose tip
                }
                else if(k == 8){
                    image_points.push_back( cv::Point2d(x, y) );    // Chin
                }
                else if(k==36){
                    image_points.push_back( cv::Point2d(x, y) );    // Left eye left corner
                }
                else if(k==45){
                    image_points.push_back( cv::Point2d(x, y) );    // Right eye right corner
                }
                else if(k==48){
                    image_points.push_back( cv::Point2d(x, y) );    // Left Mouth corner
                }
                else if(k==54){
                    image_points.push_back( cv::Point2d(x, y) );    // Right mouth corner
                }
            }
        }
        LOG(INFO) << image_points[0] <<"Nose tip" << image_points[1] << "Chin" << image_points[2] <<"Leye" << image_points[3] << "Reye" << image_points[4] << "Lmouth" << image_points[5] <<"RMouth";
        model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
        model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
        model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
        model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
        model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
        model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
        // Camera internals
        double focal_length = 480; // Approximate focal length.
        Point2d center = cv::Point2d(240, 320);
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
        
        cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector,false);
        LOG(INFO) << "model complete";
        
    return jDetRet;
}

//void getEulerAngles(cv::Mat &rotCamerMatrix,cv::Vec3d &eulerAngles){
//        cv::Mat cameraMatrix,rotMatrix,transVect,rotMatrixX,rotMatrixY,rotMatrixZ;
//        double* _r = rotCamerMatrix.ptr<double>();
//        double projMatrix[12] = {_r[0],_r[1],_r[2],0,_r[3],_r[4],_r[5],0,_r[6],_r[7],_r[8],0};
//
//        cv::decomposeProjectionMatrix( Mat(3,4,CV_64FC1,projMatrix),
//                                      cameraMatrix,
//                                      rotMatrix,
//                                      transVect,
//                                      rotMatrixX,
//                                      rotMatrixY,
//                                      rotMatrixZ,
//                                      eulerAngles);
//}
    
//JNIEXPORT jfloatArray JNICALL
//    Java_com_tzutalin_dlib_FaceDet_estimateHeadPose(JNIEnv* env, jobject thiz){
//
//        // Solve for pose
//
//        cv::Mat rotCamerMatrix1;
//        cv::Rodrigues(rotation_vector,rotCamerMatrix1);
//
//        cv::Vec3d eulerAngles;
//        getEulerAngles(rotCamerMatrix1,eulerAngles);
//
//        LOG(INFO) << "yaw " << eulerAngles[1] << "pitch" << eulerAngles[0] << "roll" << eulerAngles[2];
//
//        jfloatArray result;
//        result = env->NewFloatArray(3);
//        if (result == NULL) {
//            return NULL; /* out of memory error thrown */
//        }
//
//        jfloat array1[3];
//        array1[0] = eulerAngles[0];
//        array1[1] = eulerAngles[1];
//        array1[2] = eulerAngles[2];
//        env->SetFloatArrayRegion(result, 0, 3, array1);
//        return result;
//}
    
void JNIEXPORT JNICALL
    Java_com_tzutalin_dlib_FaceDet_SetLandmarkModel(JNIEnv* env, jobject thiz, jstring jLandmarkPath){
        LOG(INFO) << "set_landmark";
        std::string landmarkPath = jniutils::convertJStrToString(env, jLandmarkPath);
        if (!landmarkPath.empty() && jniutils::fileExists(landmarkPath)) {
            dlib::deserialize(landmarkPath) >> msp;
            LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
}

//트래커 제거
void JNIEXPORT JNICALL
    Java_com_tzutalin_dlib_FaceDet_EliminateTracker(JNIEnv* env, jobject thiz, jint trackerID){
        LOG(INFO) << "Eliminate tracker";
        tracker_array[trackerID] = nullptr;
}

//////////////////////////
    
#ifdef __cplusplus
}
#endif
