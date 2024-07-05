#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

const int HISTORY_SIZE = 10; // Number of frames to keep in history

// Function to detect potential fire regions using color and intensity
cv::Mat detectPotentialFire(const cv::Mat& frame) {
    cv::Mat hsv, mask1, mask2, fireMask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // Detect bright regions
    cv::inRange(hsv, cv::Scalar(0, 50, 200), cv::Scalar(25, 255, 255), mask1);
    
    // Detect high-intensity regions in grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask2, 200, 255, cv::THRESH_BINARY);
    
    // Combine masks
    cv::bitwise_and(mask1, mask2, fireMask);
    
    // Morphological operations to reduce noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(fireMask, fireMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(fireMask, fireMask, cv::MORPH_CLOSE, kernel);
    
    return fireMask;
}

// Function to detect smoke using motion and color analysis
cv::Mat detectSmoke(const cv::Mat& frame, const cv::Mat& prevFrame) {
    if (frame.empty() || prevFrame.empty()) {
        return cv::Mat();  // Return an empty mat if either frame is empty
    }

    cv::Mat gray, prevGray, diff, smokeMask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
    
    // Compute difference between current and previous frame
    cv::absdiff(gray, prevGray, diff);
    cv::threshold(diff, diff, 15, 255, cv::THRESH_BINARY);
    
    // Detect grayish colors
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 0, 100), cv::Scalar(179, 30, 200), smokeMask);
    
    // Combine motion and color detection
    cv::bitwise_and(diff, smokeMask, smokeMask);
    
    // Morphological operations to reduce noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
    cv::morphologyEx(smokeMask, smokeMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(smokeMask, smokeMask, cv::MORPH_CLOSE, kernel);
    
    return smokeMask;
}

int main() {
    // For Sample Videos uncomment the below code.
    // cv::VideoCapture cap("SampleVideos/VR.mp4");
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Could not open video file" << std::endl;
    //     return -1;
    // }

    // For live camera feed (0) for default webcam of the machine
    cv::VideoCapture cap(0);
    // If the default camera doesn't work, try other indices
    if (!cap.isOpened()) {
        for (int i = 1; i < 10; i++) {
            cap.open(i);
            if (cap.isOpened()) break;
        }
    }

    // For taking in a camera feed from a specific ip hosting the feed uncomment the below code.
    // if (!cap.isOpened()) {
    //     cap.open("http://192.168.1.X:YYYY/video");  // Replace X and YYYY with your IP and port
    // }


    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam" << std::endl;
        return -1;
    }

    cv::Mat frame, prevFrame;
    std::deque<cv::Mat> frameHistory;
    int fireDetectionThreshold = 100;  // Minimum area of fire to consider
    int smokeDetectionThreshold = 1000;  // Minimum area of smoke to consider
    int fireGrowthThreshold = 50;  // Minimum growth in fire area to trigger alert
    int consecutiveFireFrames = 0;
    int consecutiveSmokeFrames = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video stream" << std::endl;
            break;
        }

        cv::Mat potentialFire = detectPotentialFire(frame);
        cv::Mat smokeMask = detectSmoke(frame, prevFrame);

        // Store frame history
        frameHistory.push_back(potentialFire.clone());
        if (frameHistory.size() > HISTORY_SIZE) {
            frameHistory.pop_front();
        }

        // Analyze fire growth
        int currentFireArea = cv::countNonZero(potentialFire);
        int previousFireArea = (frameHistory.size() > 1) ? cv::countNonZero(frameHistory.front()) : 0;
        bool significantFireGrowth = (currentFireArea - previousFireArea) > fireGrowthThreshold;

        // Detect smoke
        std::vector<std::vector<cv::Point>> smokeContours;
        cv::findContours(smokeMask, smokeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        bool significantSmoke = false;
        for (const auto& contour : smokeContours) {
            if (cv::contourArea(contour) > smokeDetectionThreshold) {
                significantSmoke = true;
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 0), 2);
            }
        }

        // Update consecutive frame counters
        if (currentFireArea > fireDetectionThreshold && significantFireGrowth) {
            consecutiveFireFrames++;
        } else {
            consecutiveFireFrames = 0;
        }

        if (significantSmoke) {
            consecutiveSmokeFrames++;
        } else {
            consecutiveSmokeFrames = 0;
        }

        // Alert system
        if (consecutiveFireFrames >= 3 && consecutiveSmokeFrames >= 3) {
            std::cout << "Alert: Fire and smoke detected!" << std::endl;
            cv::putText(frame, "FIRE ALERT!", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        // Visualize potential fire regions
        cv::Mat fireVisualization;
        cv::cvtColor(potentialFire, fireVisualization, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 0.7, fireVisualization, 0.3, 0, frame);

        // Display results
        cv::imshow("Fire and Smoke Detection", frame);
        
        prevFrame = frame.clone();

        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}