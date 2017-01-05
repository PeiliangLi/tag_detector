#include <iostream>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/SVD>
//EIgen SVD libnary, may help you solve SVD
//JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
#include <Eigen/Geometry> //rotation matrix -> euler angles

//for rand()
#include <stdlib.h>
#include <time.h>
#include <numeric>


using namespace cv;
using namespace aruco;
using namespace Eigen;

//global varialbles for aruco detector
aruco::CameraParameters CamParam;
MarkerDetector MDetector;
vector<Marker> Markers;
float MarkerSize = 0.20 / 1.5 * 1.524;
float MarkerWithMargin = MarkerSize * 1.2;
BoardConfiguration TheBoardConfig;
BoardDetector TheBoardDetector;
Board TheBoardDetected;
ros::Publisher pub_odom_yourwork;
ros::Publisher pub_odom_ref;
cv::Mat K, D;
int frame_cnt = 0;
double mean_rmse = 0;
double min_rmse = 1000000.0;
double max_rmse = 0;
// test function, can be used to verify your estimation
void calculateReprojectionError(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const cv::Mat R, const cv::Mat t)
{
    puts("calculateReprojectionError begins");
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);
    for (unsigned int i = 0; i < pts_3.size(); i++)
    {
        cv::Mat p_mat(3, 1, CV_64FC1);
        p_mat.at<double>(0, 0) = pts_3[i].x;
        p_mat.at<double>(1, 0) = pts_3[i].y;
        p_mat.at<double>(2, 0) = pts_3[i].z;
        cv::Mat p = (R * p_mat + t);
        printf("(%f, %f, %f) -> (%f, %f) and (%f, %f)\n",
               pts_3[i].x, pts_3[i].y, pts_3[i].z,
               un_pts_2[i].x, un_pts_2[i].y,
               p.at<double>(0) / p.at<double>(2), p.at<double>(1) / p.at<double>(2));
    }
    puts("calculateReprojectionError ends");
}
// test function, can be used to verify your estimation
double calculateReprojectionError(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const Matrix3d R, const Vector3d t)
{
    double sum_error = 0;

    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);
    for (unsigned int i = 0; i < pts_3.size(); i++)
    {
       Vector3d p_mat;
        p_mat(0) = pts_3[i].x;
        p_mat(1) = pts_3[i].y;
        p_mat(2) = pts_3[i].z;
        Vector3d p = (R * p_mat + t);
        sum_error += pow((un_pts_2[i].x-p(0) / p(2)),2) + pow(un_pts_2[i].y-p(1) / p(2),2);
    }
    return sum_error;
}

double calculateReprojectionError_without_undistortion(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &un_pts_2, const Matrix3d R, const Vector3d t)
{
    double sum_error = 0;

    // vector<cv::Point2f> un_pts_2;
    // cv::undistortPoints(pts_2, un_pts_2, K, D);
    for (unsigned int i = 0; i < pts_3.size(); i++)
    {
       Vector3d p_mat;
        p_mat(0) = pts_3[i].x;
        p_mat(1) = pts_3[i].y;
        p_mat(2) = pts_3[i].z;
        Vector3d p = (R * p_mat + t);
        sum_error += pow((un_pts_2[i].x-p(0) / p(2)),2) + pow(un_pts_2[i].y-p(1) / p(2),2);
    }
    return sum_error;
}

void pose_refinement(const Matrix3d intrinsic_K, Matrix3d& R, Vector3d& T, const vector<cv::Point3f> &p3, const vector<cv::Point2f> &pts_2, const vector<bool>& is_inliner)
{
    const int N = p3.size();
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);

    MatrixXd J(2,6);

    VectorXd r = MatrixXd::Zero(2,1);

    double s1,s2,s3,c1,c2,c3;
    double X, Y, Z, t1, t2, t3;

    Vector3d r_i;
    Vector3d p3i;
    const int Max_Iter_N = 5;
    int iter_n = 0;
    double  error;

    while(iter_n < Max_Iter_N)
    {
        error =  calculateReprojectionError_without_undistortion(p3, un_pts_2, R, T);

        printf("iteration %d, error: %f\n", iter_n, sqrt(error/p3.size()));
        Vector3d e = R.eulerAngles(2,1,0); //ZYX

        MatrixXd A = MatrixXd::Zero(6,6);
        VectorXd b = MatrixXd::Zero(6,1);
        t1 = T(0);t2 = T(1);t3 = T(2);
        s1 = sin(e(0));s2= sin(e(1));s3 = sin(e(2));
        c1 = cos(e(0));c2 = cos(e(1));c3 = cos(e(2));

        for(int i = 0; i < N; ++i)
        {
            if(!is_inliner[i]) continue;

            X = p3[i].x; Y = p3[i].y; Z = p3[i].z;

            J(0,0) = (Y*(c1*c3 + s1*s2*s3) - Z*(c1*s3 - c3*s1*s2) + X*c2*s1)/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3);
            J(0,1) = -(Z*c1*c2*c3 - X*c1*s2 + Y*c1*c2*s3)/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3) - ((X*c2 + Z*c3*s2 + Y*s2*s3)*(t1 - Y*(c3*s1 - c1*s2*s3) + Z*(s1*s3 + c1*c3*s2) + X*c1*c2))/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2);
            J(0,2) = ((Y*c2*c3 - Z*c2*s3)*(t1 - Y*(c3*s1 - c1*s2*s3) + Z*(s1*s3 + c1*c3*s2) + X*c1*c2))/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2) - (Y*(s1*s3 + c1*c3*s2) + Z*(c3*s1 - c1*s2*s3))/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3);
            J(0,3) = -1/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3);
            J(0,4) = 0;
            J(0,5) = (t1 - Y*(c3*s1 - c1*s2*s3) + Z*(s1*s3 + c1*c3*s2) + X*c1*c2)/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2);
            J(1,0) = -(Z*(s1*s3 + c1*c3*s2) - Y*(c3*s1 - c1*s2*s3) + X*c1*c2)/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3);
            J(1,1) = -(Z*c2*c3*s1 - X*s1*s2 + Y*c2*s1*s3)/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3) - ((X*c2 + Z*c3*s2 + Y*s2*s3)*(t2 + Y*(c1*c3 + s1*s2*s3) - Z*(c1*s3 - c3*s1*s2) + X*c2*s1))/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2);
            J(1,2) = (Y*(c1*s3 - c3*s1*s2) + Z*(c1*c3 + s1*s2*s3))/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3) + ((Y*c2*c3 - Z*c2*s3)*(t2 + Y*(c1*c3 + s1*s2*s3) - Z*(c1*s3 - c3*s1*s2) + X*c2*s1))/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2);
            J(1,3) = 0;
            J(1,4) = -1/(t3 - X*s2 + Z*c2*c3 + Y*c2*s3);
            J(1,5) = (t2 + Y*(c1*c3 + s1*s2*s3) - Z*(c1*s3 - c3*s1*s2) + X*c2*s1)/pow((t3 - X*s2 + Z*c2*c3 + Y*c2*s3),2);

            A = A + J.transpose()*J;
            p3i << p3[i].x,p3[i].y,p3[i].z;

            r_i = R*p3i+T;
            r(0) = un_pts_2[i].x-r_i(0)/r_i(2);
            r(1) = un_pts_2[i].y-r_i(1)/r_i(2);
            b = b - J.transpose()*r;
        }
        VectorXd delta = A.inverse()*b;
        R = AngleAxisd(e(0)+delta(0), Vector3d::UnitZ())*AngleAxisd(e(1)+delta(1), Vector3d::UnitY())*AngleAxisd(e(2)+delta(2), Vector3d::UnitX());
        T(0) = T(0) + delta(3);
        T(1) = T(1) + delta(4);
        T(2) = T(2) + delta(5);
        iter_n++;
    }

}


vector<bool> ransac(const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, int& max_inliner_num)
{
    int N = pts_3.size();
    const int Itera_N = 5;
    const double Error_Threshold = 0.001;
    cv::Mat r, rvec, t;
    vector<cv::Point2f> un_pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);

    vector<cv::Point3f> p3_sample;
    vector<cv::Point2f> p2_sample;

    cv::Mat p_mat(3, 1, CV_64FC1);
    cv::Mat p_tmp;
    double error = 0;
    vector<bool> is_inliner_tmp(N,0);
    vector<bool> is_inliner(N,0);
    max_inliner_num = 0;
    vector<int>sample_id(4,0);
    for(int it = 0; it < Itera_N; ++it)
    {

        vector<int>sample_id_tmp(N,0);
        srand(clock());
        int id = 1;
        for(int k = 0; k < 4; ++k)
        {
            while(sample_id_tmp[id=rand()%N]); //still have chance to be many zeros!! //TODO
            sample_id_tmp[id]=id;
            sample_id[k] = id;
            p3_sample.push_back(pts_3[id]);
            p2_sample.push_back(pts_2[id]);
        }
        cv::solvePnP(p3_sample, p2_sample, K, D, rvec, t);
        cv::Rodrigues(rvec, r);
        for(int i = 0 ; i < N; ++i)
        {
            p_mat.at<double>(0, 0) = pts_3[i].x;
            p_mat.at<double>(1, 0) = pts_3[i].y;
            p_mat.at<double>(2, 0) = pts_3[i].z;
            p_tmp = (r * p_mat + t);
            error = pow((un_pts_2[i].x-p_tmp.at<double>(0) / p_tmp.at<double>(2)),2) + pow(un_pts_2[i].y-p_tmp.at<double>(1) / p_tmp.at<double>(2),2);
            if(error < Error_Threshold)
            {
                is_inliner_tmp[i] = 1;
            }
        }
        int inliner_num = accumulate(is_inliner_tmp.begin(),is_inliner_tmp.end(), 0);
        if(inliner_num > max_inliner_num)
        {
            max_inliner_num = inliner_num;
            is_inliner = is_inliner_tmp;
        }
    }

    return is_inliner;
}

// the main function you need to work with
// pts_id: id of each point
// pts_3: 3D position (x, y, z) in world frame
// pts_2: 2D position (u, v) in image frame

void process(const vector<int> &pts_id, const vector<cv::Point3f> &pts_3, const vector<cv::Point2f> &pts_2, const ros::Time& frame_time)
{
    //version 1, as reference
    cv::Mat r, rvec, t;
    cv::solvePnP(pts_3, pts_2, K, D, rvec, t);
    cv::Rodrigues(rvec, r);

    Matrix3d R_ref;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {
            R_ref(i,j) = r.at<double>(i, j);
        }
    Quaterniond Q_ref;
    Q_ref = R_ref;
    nav_msgs::Odometry odom_ref;
    odom_ref.header.stamp = frame_time;
    odom_ref.header.frame_id = "world";
    odom_ref.pose.pose.position.x = t.at<double>(0, 0);
    odom_ref.pose.pose.position.y = t.at<double>(1, 0);
    odom_ref.pose.pose.position.z = t.at<double>(2, 0);
    odom_ref.pose.pose.orientation.w = Q_ref.w();
    odom_ref.pose.pose.orientation.x = Q_ref.x();
    odom_ref.pose.pose.orientation.y = Q_ref.y();
    odom_ref.pose.pose.orientation.z = Q_ref.z();
    pub_odom_ref.publish(odom_ref);

    // version 2, your work
    Matrix3d R;
    Vector3d T;
    R.setIdentity();
    T.setZero();
    // ROS_INFO("write your code here!");
    int N = pts_3.size();
    #define USE_RANSAC
#ifdef USE_RANSAC
    int in_N = 0;
    vector<bool> is_inliner = ransac(pts_3, pts_2, in_N);
#else
    vector<bool> is_inliner(N, 1);
    in_N = N;
#endif

    printf("N= %d , inliners: %d\n", N, in_N);
    if(N!=in_N)
        printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ found outliers! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");


    vector<cv::Point2f> un_pts_2 = pts_2;
    cv::undistortPoints(pts_2, un_pts_2, K, D);

    if(in_N < 5)
        {
            printf("$$$$$$$$$$$$ not enough inliners!  $$$$$$$$$$$$$\n");
            return;
        }

    MatrixXd A(2*in_N, 9);
    int cnt = 0;
    for(int i = 0; i < N; ++i)
    {
        if(!is_inliner[i]) continue;
        int j = cnt*2;
        int k = j+1;
        A(j,0) = pts_3[i].x;
        A(j,1) = pts_3[i].y;
        A(j,2) = 1;
        A(j,3) = 0;
        A(j,4) = 0;
        A(j,5) = 0;
        A(j,6) = -pts_3[i].x*un_pts_2[i].x;
        A(j,7) = -pts_3[i].y*un_pts_2[i].x;
        A(j,8) = -un_pts_2[i].x;

        A(k,0) = 0;
        A(k,1) = 0;
        A(k,2) = 0;
        A(k,3) = pts_3[i].x;
        A(k,4) = pts_3[i].y;
        A(k,5) = 1;
        A(k,6) = -pts_3[i].x*un_pts_2[i].y;
        A(k,7) = -pts_3[i].y*un_pts_2[i].y;
        A(k,8) = -un_pts_2[i].y;
        cnt++;
    }
    printf("%f, %f, %f, %f\n", pts_3[0].x, pts_3[0].y, un_pts_2[0].x, un_pts_2[0].y);

    JacobiSVD<MatrixXd> svdA(A, ComputeThinU | ComputeThinV);
    Matrix3d H;

    H << svdA.matrixV().col(8)(0), svdA.matrixV().col(8)(1), svdA.matrixV().col(8)(2),
        svdA.matrixV().col(8)(3), svdA.matrixV().col(8)(4), svdA.matrixV().col(8)(5),
        svdA.matrixV().col(8)(6), svdA.matrixV().col(8)(7), svdA.matrixV().col(8)(8);


    Matrix3d intrinsic_K;
    intrinsic_K << K.at<double>(0,0),K.at<double>(0,1),K.at<double>(0,2),
                K.at<double>(1,0),K.at<double>(1,1),K.at<double>(1,2),
                K.at<double>(2,0),K.at<double>(2,1),K.at<double>(2,2);

    // Matrix3d invK_H = intrinsic_K.inverse()*H;
    Matrix3d invK_H = H;

    Matrix3d H_hat(H);
    H_hat.col(2) = H_hat.col(0).cross(H_hat.col(1));
    JacobiSVD<MatrixXd> svdH_hat(H_hat, ComputeThinU | ComputeThinV);

    R = svdH_hat.matrixU()*(svdH_hat.matrixV().transpose());

    T = invK_H.col(2)/(invK_H.col(0).norm());
    if(T(2) < 0)
        {
            R.col(0) = -R.col(0);
            R.col(1) = -R.col(1);

            T = -T;
        }


    pose_refinement(intrinsic_K, R, T, pts_3, pts_2, is_inliner);

    double error =  calculateReprojectionError(pts_3, pts_2, R, T);

    error = sqrt(error/pts_3.size());
    printf("frame %d, points %d, RMSE error: %f\n", frame_cnt, (int)pts_3.size(), error);
    mean_rmse += error;
    if(min_rmse > error)
    {
        min_rmse = error;
    }
    if(max_rmse < error)
    {
        max_rmse = error;
    }
    printf("Mean RMSE: %f, Max RMSE: %f, Min RMSE: %f\n\n", mean_rmse/frame_cnt, max_rmse, min_rmse);

    Quaterniond Q_yourwork;
    Q_yourwork = R;
    nav_msgs::Odometry odom_yourwork;
    odom_yourwork.header.stamp = frame_time;
    odom_yourwork.header.frame_id = "world";
    odom_yourwork.pose.pose.position.x = T(0);
    odom_yourwork.pose.pose.position.y = T(1);
    odom_yourwork.pose.pose.position.z = T(2);
    odom_yourwork.pose.pose.orientation.w = Q_yourwork.w();
    odom_yourwork.pose.pose.orientation.x = Q_yourwork.x();
    odom_yourwork.pose.pose.orientation.y = Q_yourwork.y();
    odom_yourwork.pose.pose.orientation.z = Q_yourwork.z();
    pub_odom_yourwork.publish(odom_yourwork);

}

cv::Point3f getPositionFromIndex(int idx, int nth)
{
    int idx_x = idx % 18, idx_y = idx / 18;
    double p_x = idx_x * MarkerWithMargin - (3 + 2.5 * 0.2) * MarkerSize;
    double p_y = idx_y * MarkerWithMargin - (12 + 11.5 * 0.2) * MarkerSize;
    return cv::Point3f(p_x + (nth == 1 || nth == 2) * MarkerSize, p_y + (nth == 2 || nth == 3) * MarkerSize, 0.0);
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    frame_cnt++;
    double t = clock();
    cv_bridge::CvImagePtr bridge_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    MDetector.detect(bridge_ptr->image, Markers);
    float probDetect = TheBoardDetector.detect(Markers, TheBoardConfig, TheBoardDetected, CamParam, MarkerSize);
    ROS_DEBUG("p: %f, time cost: %f\n", probDetect, (clock() - t) / CLOCKS_PER_SEC);

    vector<int> pts_id;
    vector<cv::Point3f> pts_3;
    vector<cv::Point2f> pts_2;
    for (unsigned int i = 0; i < Markers.size(); i++)
    {
        int idx = TheBoardConfig.getIndexOfMarkerId(Markers[i].id);

        char str[100];
        sprintf(str, "%d", idx);
        //cv::putText(bridge_ptr->image, str, Markers[i].getCenter(), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(-1));
        //for (unsigned int j = 0; j < 4; j++)
        //{
            //sprintf(str, "%d", j);
            //cv::putText(bridge_ptr->image, str, Markers[i][j], CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(-1));
        //}

        for (unsigned int j = 0; j < 4; j++)
        {
            pts_id.push_back(Markers[i].id * 4 + j);
            pts_3.push_back(getPositionFromIndex(idx, j));
            pts_2.push_back(Markers[i][j]);
        }
    }

    //begin your function
    if (pts_id.size() > 5)
        process(pts_id, pts_3, pts_2, img_msg->header.stamp);

    //cv::imshow("in", bridge_ptr->image);
    //cv::waitKey(2);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tag_detector");
    ros::NodeHandle n("~");

    ros::Subscriber sub_img = n.subscribe("image_raw", 100, img_callback);
    pub_odom_yourwork = n.advertise<nav_msgs::Odometry>("/tag_detector/odom_yourwork",10);
    pub_odom_ref = n.advertise<nav_msgs::Odometry>("odom_ref",10);
    //init aruco detector
    string cam_cal, board_config;
    n.getParam("cam_cal_file", cam_cal);
    n.getParam("board_config_file", board_config);
    CamParam.readFromXMLFile(cam_cal);
    TheBoardConfig.readFromFile(board_config);

    //init intrinsic parameters
    cv::FileStorage param_reader(cam_cal, cv::FileStorage::READ);
    param_reader["camera_matrix"] >> K;
    param_reader["distortion_coefficients"] >> D;

    //init window for visualization
    //cv::namedWindow("in", 1);

    ros::spin();
}
