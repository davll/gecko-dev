/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "MediaEngineWebRTC.h"
#include "Layers.h"
#include "ImageContainer.h"
#include "nsRect.h"
#include "nsTArray.h"

#include <opencv2/core.hpp>
#if CV_VERSION_MAJOR < 3
#error "OpenCV Ver. 3.0+ is required"
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ocl.hpp>
#include <CL/cl.h>

namespace mozilla {

void
MediaEngineWebRTCVideoSource::InitFaceDetection()
{
  // init opencl context
  //cv::ocl::initializeContext();
  
  //static const char* path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
  static const char* path = "/home/david/haarcascade_frontalface_alt.xml";
  NS_ENSURE_TRUE_VOID(mFaceDetector.load(path));
  
  for (int i = 0; i < 6; ++i)
    mFaceDetectionTime[i] = 0.0;
  mFaceDetectionCount = 0;
}

void
MediaEngineWebRTCVideoSource::DetectFaces(nsTArray<nsRect>& aFaces)
{
  layers::PlanarYCbCrImage* yimage = static_cast<layers::PlanarYCbCrImage*>(mImage.get());
  const layers::PlanarYCbCrData* ydata = yimage->GetData();

  int64_t t0 = cv::getTickCount(); // start

  // Step 1: Wrap object to OpenCV
  cv::Mat srcImg(cv::Size(mWidth, mHeight), CV_8UC1, ydata->mYChannel);
  int64_t t1 = cv::getTickCount();

  // Step 2: Load data to GPU
  //cv::ocl::oclMat srcImg_gpu(srcImg);
  cv::ocl::oclMat srcImg_gpu(cv::Size(mWidth, mHeight), CV_8UC1, ydata->mYChannel);
  int64_t t2 = cv::getTickCount();
  // Step 3: Resize image
  const bool shouldDownsample = true;
  const double ds = 0.5;
  if (shouldDownsample) {
    cv::ocl::oclMat resized;
    cv::ocl::resize(srcImg_gpu, resized, cv::Size(), ds, ds);
    srcImg_gpu = resized;
  }
  int64_t t3 = cv::getTickCount();
  // Step 4: Equalize image
  cv::ocl::equalizeHist(srcImg_gpu, srcImg_gpu);
  int64_t t4 = cv::getTickCount();
  // Step 5: Read back processed image
#if MOZ_OPENCV_USE_OPENCL
  cv::ocl::oclMat& frame_g = srcImg_gpu;
#else
  cv::Mat frame_g(srcImg_gpu);
#endif
  int64_t t5 = cv::getTickCount();
  // Step 6: Detect faces
  std::vector<cv::Rect> faces;
  //mFaceDetector.detectMultiScale(frame_g, faces, 1.165, 2,
  //                               0|cv::CASCADE_SCALE_IMAGE,
  //                               cv::Size(60, 60));
  mFaceDetector.detectMultiScale(frame_g, faces, 1.3, 2,
                                 0|cv::CASCADE_SCALE_IMAGE,
                                 cv::Size(45, 45));
  int64_t t6 = cv::getTickCount();

  std::vector<cv::Rect>::iterator it = faces.begin(), iend = faces.end();
  for (; it != iend; ++it) {
    if (shouldDownsample) {
      const double rds = 1.0 / ds;
      it->x *= rds, it->y *= rds, it->width *= rds, it->height *= rds;
    }
    cv::rectangle(srcImg, *it, cv::Scalar(0,0,0), 3);
  }

  cv::rectangle(srcImg, cv::Rect(0,0,150,120), cv::Scalar(255,255,255),-1);
  int64_t atime[] = { t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5 };
  double denom = 1.0 / (++mFaceDetectionCount);
  for (int i = 0; i < 6; ++i) {
    double tmp = mFaceDetectionTime[i];
    double a = atime[i] * 0.000001;
    mFaceDetectionTime[i] = tmp + (a - tmp) * denom;
  }
  char buf[2048];
  sprintf(buf, "[1] Wrap image to cv::Mat: %lf ms", mFaceDetectionTime[0]);
  cv::putText(srcImg, buf, cv::Point(0,10), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  sprintf(buf, "[2] Load image to GPU: %lf ms", mFaceDetectionTime[1]);
  cv::putText(srcImg, buf, cv::Point(0,25), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  sprintf(buf, "[3] Resize image: %lf ms", mFaceDetectionTime[2]);
  cv::putText(srcImg, buf, cv::Point(0,40), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  sprintf(buf, "[4] Equalize image: %lf ms", mFaceDetectionTime[3]);
  cv::putText(srcImg, buf, cv::Point(0,55), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  sprintf(buf, "[5] Read processed image: %lf ms", mFaceDetectionTime[4]);
  cv::putText(srcImg, buf, cv::Point(0,70), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  sprintf(buf, "[6] Detect faces: %lf ms", mFaceDetectionTime[5]);
  cv::putText(srcImg, buf, cv::Point(0,85), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  double sum = 0.0;
  for (int i = 0; i < 6; ++i)
    sum += mFaceDetectionTime[i];
  sprintf(buf, "TOTAL = %lf ms", sum);
  cv::putText(srcImg, buf, cv::Point(0,100), 
              cv::FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0,0,0));
  
  cv::ocl::finish();
}

void
MediaEngineWebRTCVideoSource::ShutdownFaceDetection()
{
  // TODO: release CL context explicitly
}

}
