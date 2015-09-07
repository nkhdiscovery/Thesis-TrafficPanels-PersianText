
#include "nkhCV.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QMessageBox>
#include <QString>
#include <QFileInfo>
#include <QDir>
#include <QStringList>
#include <time.h>
#include <limits.h>
#include <cstring>

using namespace cv;
nkhCV::nkhCV(QMainWindow *parent)
{
    ui = parent ;

}

/*******************Final Thesis*******************/

void nkhCV::nkhCIELAB(QString fileName)
{
    Mat img,cielab;
    if(!nkhImread(img,fileName))
        return;
    toCIELAB(img, cielab);
    //nkhImshow_Write(img,cielab,"Color:",fileName,"CIELAB");
}

void nkhCV::toCIELAB(cv::Mat &src, cv::Mat &dst){
    cvtColor(src, dst, CV_BGR2Lab);
}

void nkhCV::greenMask(QString fileName, bool video){
    Mat img,masked,nullImg;
    if(!nkhImread(img,fileName,true))
        return;
    colorMask(img, masked, "green", video);
}

void nkhCV::colorMask(Mat &src, Mat &dst, std::string color, bool video)
{
    Mat srcBack = src.clone();
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    int iLowH = 68;
    int iHighH = 93;

    int iLowS = 40;
    int iHighS = 255;

    int iLowV = 47;
    int iHighV = 174;

    int iLowB = 68;
    int iHighB = 93;

    int iLowG = 63;
    int iHighG = 255;

    int iLowR = 55;
    int iHighR = 174;

    int hueValue = 0;
    if(color=="green")
    {
        //imshow("green",src);
        hueValue = 255; //green h : 38-75
    }
    cvCreateTrackbar("LowH", "Control", &iLowH, hueValue); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, hueValue);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    //    cvCreateTrackbar("LowB", "Control", &iLowB, hueValue); //Hue (0 - 179)
    //    cvCreateTrackbar("HighB", "Control", &iHighB, hueValue);

    //    cvCreateTrackbar("LowG", "Control", &iLowG, 255); //Saturation (0 - 255)
    //    cvCreateTrackbar("HighG", "Control", &iHighG, 255);

    //    cvCreateTrackbar("LowR", "Control", &iLowR, 255); //Value (0 - 255)
    //    cvCreateTrackbar("HighR", "Control", &iHighR, 255);

    //    Mat normal;
    //    nkhImproc(src,normal,NORM_CMD);
    //medianBlur(src,src,9);

    Mat imgHSV;
    Mat dst2 ;
    VideoCapture cap("11.mp4");

    // fps counter begin
    time_t start, end;
    int counter = 0;
    double sec;
    double fps;
    // fps counter end

    //    VideoWriter outVid("out.mp4",cap.get(CV_CAP_PROP_FOURCC),cap.get(CV_CAP_PROP_FPS),
    //                       Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
    while(true)
    {
        // fps counter begin
        if (counter == 0){
            time(&start);
        }
        // fps counter end

        if(video)
            cap>>src ;

        if(src.empty())
            break;

        if(video)
            resize(src,src, cvSize(src.size().width/2,src.size().height/2));
        cvtColor(src, imgHSV, COLOR_BGR2HSV);
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), dst); //Threshold the image
        //        inRange(src, Scalar(iLowB, iLowG, iLowR), Scalar(iHighB, iHighG, iHighR), dst2); //Threshold the image


        //morphological opening (remove small objects from the foreground)
        erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        dilate( dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        //morphological closing (fill small holes in the foreground)
        dilate( dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );

        //        Mat resized;
        if(video)
            resize(dst,dst, cvSize(dst.size().width/2.0,dst.size().height/2.0));



        medianBlur(dst,dst,5);

        //nkhImshow_Write(src, dst, "HSV", "HSV","green");
        //        nkhImshow_Write(src, dst2, "RGB", "RGB","green");

        vector<vector<Point> > contours;
        vector<Vec4i> order;
        GaussianBlur( dst, dst, Size(5, 5), 1,1 );
        Mat nullImg ;
        int params[]={70,210,3};
        Mat edges = edgeDetect("","Edge Canny",EDGE_DETECT_CANNY,params, dst);
        findContours( edges, contours, order, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        Mat showContours;// Mat::zeros( edges.size(), CV_8UC3 );
        if(video)
            resize(src,showContours,cvSize(src.size().width/2.0,src.size().height/2.0));
        else
            showContours = src.clone();


        RNG rng(12345);

        vector<vector<Point> > contoursApprox;
        contoursApprox.resize(contours.size());
        for( size_t k = 0; k < contours.size(); k++ )
        {
            double epsilon = 0.05*arcLength(contours[k],true);
            approxPolyDP(Mat(contours[k]), contoursApprox[k], epsilon, true);
        }

        vector<Moments> mu(contoursApprox.size() );
        for( int i = 0; i < contoursApprox.size(); i++ )
        { mu[i] = moments( contoursApprox[i], false ); }


        vector<Point2f> mc( contoursApprox.size() );
        for( int i = 0; i < contoursApprox.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

        vector<RotatedRect> minRect( contours.size() );
        for( int i = 0; i< contoursApprox.size(); i++ )
        {
            long long carea = contourArea(contoursApprox[i]) ;
            long long area = dst.size().width*dst.size().height;
            minRect[i] = minAreaRect( Mat(contoursApprox[i]) );
            if( carea< (area)/900.0 || carea > (area)/3.0
                    || (int)mc[i].y > dst.size().height/2  || minRect[i].size.width < minRect[i].size.height
                    || (minRect[i].angle <-30 ))
                continue;

            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//            drawContours( showContours, contoursApprox, i, color, 2, 8, order, 0, Point() );
            Point2f rect_points[4]; minRect[i].points( rect_points );
            for( int j = 0; j < 4; j++ )
                line( showContours, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
        imshow("Contoures", showContours);

        nkhImshow_Write(dst, showContours, "Contoures", "cont","green");

        //        for( int i = 0; i< contours.size(); i++ )
        //        {
        //            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //            drawContours( showContours, contours, i, color, 2, 8, order, 0, Point() );
        //        }
        //        imshow("Contoures",showContours);

        // fps counter begin
        char* fpsString = new char[32];
        time(&end);
        counter++;
        sec = difftime(end, start);
        fps = counter/sec;
        if (counter > 30)
            std::sprintf(fpsString, "%.2f fps\n",fps);
        // overflow protection
        if (counter == (INT_MAX - 1000))
            counter = 0;
        // fps counter end

        putText(src, fpsString, cvPoint(30,30),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        // imshow("Original",src);
        imshow("Thresh",dst);
        // outVid<<showContours ;

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            break;
        }
        if(!video)
            src=srcBack.clone();
    }

}


/********************************************/

void nkhCV::normalizeHist(QString fileName,QString windowName)
{
    Mat img,normalizedImg;
    if(!nkhImread(img,fileName))
        return;
    nkhImproc(img,normalizedImg,NORM_CMD);
    nkhImshow_Write(img,normalizedImg,windowName,fileName,NORM_CMD);
}
void nkhCV::nkhEqualizeHist(QString fileName,QString windowName)
{
    Mat img,equalizedImg;
    if(!nkhImread(img,fileName))
        return;
    nkhImproc(img,equalizedImg,EQHIST_CMD);
    nkhImshow_Write(img,equalizedImg,windowName,fileName,EQHIST_CMD);
}
void nkhCV::showHist(QString fileName, QString windowName)
{
    Mat img,equalizedImg,normalizedImg;
    if(!nkhImread(img,fileName))
        return;
    nkhImproc(img,normalizedImg,NORM_CMD);
    nkhImproc(img,equalizedImg,EQHIST_CMD);
    Mat origHist,normHist,eqHist;

    nkhImproc(img,origHist,HIST_IMG_CMD);
    nkhImproc(normalizedImg,normHist,HIST_IMG_CMD);
    nkhImproc(equalizedImg,eqHist,HIST_IMG_CMD);

    nkhImshow_Write(img,origHist,"Original Hist","orig.jpg",HIST_IMG_CMD);
    nkhImshow_Write(normalizedImg,normHist,"Normalized Hist","normal.jpg",HIST_IMG_CMD);
    nkhImshow_Write(equalizedImg,eqHist,"Equalized Hist","equal.jpg",HIST_IMG_CMD);
}

bool nkhCV::nkhImread(Mat &dst, QString fileName,bool color /*=false*/)
{
    dst = imread(fileName.toStdString());
    if(dst.empty())
    {
        QMessageBox::information(ui,"Image Not Found!",
                                 QString(("Image" + fileName +
                                          " not found.\nPlease copy that to current directory.")),
                                 QMessageBox::Ok|QMessageBox::Default
                                 );

        return false;
    }
    else
    {
        if(!color)
            cvtColor( dst, dst, CV_BGR2GRAY );
        return true;
    }
}
void nkhCV::nkhImshow_Write(cv::Mat &orig, cv::Mat& proc,QString windowName,QString fileName,QString PROC_CMD)
{
    QString nameWindowOrig = windowName+QObject::tr(": Original") ;
    QString nameWindowProc = windowName+QObject::tr(": Processed") ;
    QFileInfo fileInfo(fileName);
    if( !orig.empty() )
    {
        namedWindow(nameWindowOrig.toStdString(),1);
        imshow(nameWindowOrig.toStdString(),orig);
    }

    namedWindow(nameWindowProc.toStdString(),1);
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    imshow(nameWindowProc.toStdString(),proc);
    imwrite((fileInfo.completeBaseName()+"-"+PROC_CMD +".png"/*+fileInfo.suffix()*/).toStdString()
            ,proc,compression_params);
    // cvWaitKey();
}
Mat nkhCV::nkhCalcHist(Mat &src)
{
    Mat hist;
    const int sizes[] = {256};
    const int channels[] = {0};
    float range[] = {0,256};
    const float *ranges[] = {range};
    calcHist(&src, 1, channels, Mat(), hist, 1, sizes, ranges);
    return hist;
}
void nkhCV::nkhDrawHist(Mat &hist, Mat &dst)
{
    int width = 256, height = 256, histBins = 256 ;
    dst = Mat1b::zeros(height,histBins);
    double maxH=0;
    minMaxLoc(hist, 0, &maxH);
    for( int i = 0; i < histBins; i++ )
    {
        float binVal =hist.at<float>(i);
        int intensity = (int)(binVal*0.9*height/maxH);
        line(dst,
             Point(i,height),
             Point(i,height-intensity),
             Scalar::all(255))
                ;
    }
}

void nkhCV::nkhImproc(cv::Mat &src, cv::Mat &dst, QString command)
{

    if(command==NORM_CMD)
    {
        normalize(src,dst,0,255,CV_MINMAX);
    }
    else if(command == EQHIST_CMD)
    {
        //cvtColor( src, src, CV_BGR2GRAY );
        equalizeHist(src,dst);
    }
    else if(command == HIST_IMG_CMD)
    {
        Mat hist = nkhCalcHist(src);
        nkhDrawHist(hist,dst);
    }
    else
    {
        return;
    }
}

void nkhCV::nkhFilter2D(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel)
{
    filter2D(src, dst, CV_16U , kernel );
}
Mat nkhCV::nkhSobelFilterX()
{
    float sobelKernel[] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1};

    // Sobel Kernel from Gary Bradski's Book, working fine.
    /*float sobelKernel[] = { 1, -2, 1,
                            2, -4, 2,
                            1, -2, 1};
                            */
    return Mat(3,3,CV_32F,sobelKernel).clone();
}
Mat MyNBabuFilterZero()
{
    float BabuKernel[] = { -100,-100,0,100,100,
                           -100,-100,0,100,100,
                           -100,-100,0,100,100,
                           -100,-100,0,100,100,
                           -100,-100,0,100,100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}

Mat MyNBabuFilterThirty()
{
    float BabuKernel[] = { -100,32,100,100,100,
                           -100,-78,92,100,100,
                           -100,-100,0,100,100,
                           -100,-100,-92,78,100,
                           -100,-100,100,-32,100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}

Mat MyNBabuFilterSixty()
{
    float BabuKernel[] = { 100,100,100,100,100,
                           -32,78,100,100,100,
                           -100,-92,0,92,100,
                           -100,-100,-100,-78,32,
                           -100,-100,-100,-100,-100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}

Mat MyNBabuFilterNinety()
{
    float BabuKernel[] = { 100,100,100,100,100,
                           100,100,100,100,100,
                           0,0,0,0,0,
                           -100,-100,-100,-100,-100,
                           -100,-100,-100,-100,-100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}

Mat MyNBabuFiltertwoxsixty()
{
    float BabuKernel[] = { -100,100,100,100,100,
                           -100,100,100,78,-32,
                           -100,92,0,-92,-100,
                           32,-78,-100,-100,-100,
                           -100,-100,-100,-100,-100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}

Mat MyNBabuFiltertwoxninety()
{
    float BabuKernel[] = { 100,100,100,32,-100,
                           100,100,92,-78,-100,
                           100,100,0,-100,-100,
                           100,78,-92,-100,-100,
                           100,-32,-100,-100,-100};
    return Mat(5,5,CV_32F,BabuKernel).clone();
}
Mat nkhCV::edgeDetect(QString fileName, QString windowName, QString edgeMethod, int params[],Mat &imgRef)
{
    Mat img;

    if(fileName=="")
        img=imgRef.clone();
    else if(!nkhImread(img,fileName))
        return img;
    if(edgeMethod == EDGE_DETECT_SOBEL)
    {

        Mat edgeImg,edgeImgX,edgeImgY;
        Mat edgeAbsX1,edgeAbsY1,edgeAbsX2,edgeAbsY2;

        Mat kernelX = nkhSobelFilterX();
        filter2D(img,edgeImgX,CV_16S,kernelX);
        convertScaleAbs(edgeImgX,edgeAbsX1);

        /* Uncomment to have full edges, it's like passing dx=2 to OpenCV's Sobel function
         * It's like using second order derivative in x direction.
         * BE CAREFULL that you're using kernel of
         * float sobelKernel[] = { -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1};

        flip(kernelX,kernelX,1);
        nkhFilter2D(img,edgeImgX,kernelX);
        convertScaleAbs(edgeImgX,edgeAbsX2);
        */

        kernelX = nkhSobelFilterX();
        Mat kernelY ;
        transpose(kernelX,kernelY);
        /* Uncomment to have full edges, it's like passing dx=2 to OpenCV's Sobel function
         * It's like using second order derivative in x direction.
        nkhFilter2D(img,edgeImgY,kernelY);
        convertScaleAbs(edgeImgY,edgeAbsY2);
        flip(kernelY,kernelY,0);
        */
        filter2D(img,edgeImgY,CV_16S,kernelY);
        convertScaleAbs(edgeImgY,edgeAbsY1);

        Mat edgeImg1, edgeImg2;
        //threshold(edgeAbsX1,edgeAbsX1,200,255,0);
        //threshold(edgeAbsY1,edgeAbsY1,200,255,0);
        addWeighted(edgeAbsX1, 0.5 , edgeAbsY1, 0.5 , 0 , edgeImg1);
        edgeImg = edgeImg1.clone();
        /* Uncomment to have full edges, it's like passing dx=2 to OpenCV's Sobel function
         * It's like using second order derivative in x direction.
        addWeighted(edgeAbsX2, 0.5 , edgeAbsY2, 0.5 , 0 , edgeImg2);
        addWeighted(edgeImg1, 0.5 , edgeImg2, 0.5 , 0 , edgeImg);
        */

        Mat nullImg;

        nkhImshow_Write(nullImg,edgeAbsX1,"sobelX",fileName,"sobelX");
        nkhImshow_Write(nullImg,edgeAbsY1,"sobelY",fileName,"sobelY");
        nkhImshow_Write(nullImg,edgeImg,windowName,fileName,EDGE_DETECT_SOBEL);
        return edgeImg.clone();
        /*
         * This is OpenCV's sample for Sobel function, copied from the OpenCV's Wiki,
         * if you wanna how it should be used.
         * Final edge result is the same as mine.
         *
          Mat grad_x, grad_y,grad;
          Mat abs_grad_x, abs_grad_y;

          /// Gradient X
          //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
          Sobel( img, grad_x, -1, 1, 0, 3);
          convertScaleAbs( grad_x, abs_grad_x );

          /// Gradient Y
          //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
          Sobel( img, grad_y, -1, 0, 1, 3 );
          convertScaleAbs( grad_y, abs_grad_y );
          /// Total Gradient (approximate)
          addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
          Mat nullImg;
          nkhImshow_Write(nullImg,grad,windowName,fileName,EDGE_DETECT_SOBEL);
         *
         */

    }
    else if(edgeMethod == EDGE_DETECT_NEVATIA_BABU)
    {
        Mat dst0, dst1 , dst2,dst3,dst4,dst5,dst6;
        Mat kernel = MyNBabuFilterZero();
        nkhFilter2D(img, dst1, kernel);
        kernel = MyNBabuFilterThirty();
        nkhFilter2D(img, dst2 , kernel);
        kernel = MyNBabuFilterSixty();
        nkhFilter2D(img, dst3, kernel);
        kernel = MyNBabuFilterNinety();
        nkhFilter2D(img, dst4 , kernel);
        kernel = MyNBabuFiltertwoxsixty();
        nkhFilter2D(img, dst5 , kernel);
        kernel = MyNBabuFiltertwoxninety();
        nkhFilter2D(img, dst6 , kernel);
        Mat im1 , im2 ,im3;
        addWeighted(dst1, 0.5 , dst2 , 0.5 , 0, im1) ;
        addWeighted(dst3, 0.5 , dst4 , 0.5 , 0, im2) ;
        addWeighted(dst5, 0.5 , dst6 , 0.5 , 0, im3) ;
        addWeighted(im1, 0.5 , im2 , 0.5 , 0, dst0) ;
        addWeighted(im3, 0.33 , dst0 , 0.67 , 0, dst0) ;
        Mat nullImg;
        //threshold( dst0, dst0, 90, 100,1 );
        nkhImshow_Write(nullImg,dst0,windowName,fileName,EDGE_DETECT_NEVATIA_BABU);
        return dst0.clone();
    }
    else if(edgeMethod == EDGE_DETECT_CANNY)
    {
        Mat edgesCanny;
        blur( img, edgesCanny, Size(3,3) );
        Canny( img, edgesCanny, params[0], params[1],params[2]);
        Mat nullImg ;
        QString thresholds = QString::number(params[0])
                +"-" + QString::number(params[1])+ "-k" + QString::number(params[2]);
        //nkhImshow_Write(nullImg,edgesCanny,windowName+" : "+thresholds,fileName,EDGE_DETECT_CANNY+thresholds);
        return edgesCanny.clone();
    }
}

void nkhCV::nkhHoughLines(QString fileName,int params[])
{
    Mat img;
    if(!nkhImread(img,fileName))
        return;
    Mat nullImg;
    Mat edges = edgeDetect(fileName,"Edge Canny",EDGE_DETECT_CANNY,params, nullImg);
    Mat temp = edges.clone();
    if(params[3]!=0)
        edges = addGaussianNoise(temp,params[4],params[3]);

    nkhImshow_Write(nullImg,edges,"Hough Input",fileName,"houghInp-Sig"+QString::number(params[3]));

    vector<Vec2f> detectedLines;
    HoughLines(edges, detectedLines, 0.2, CV_PI/180, 50, 0,0);
    cvtColor(img.clone(),img,CV_GRAY2BGR);

    //I got this from OpenCV's Doc. just a transfer between polar and cartesian.
    for( int i = 0; i < detectedLines.size(); i++ )
    {
        float R = detectedLines[i][0], theta = detectedLines[i][1];
        Point p1, p2;
        double a = cos(theta), b = sin(theta);
        double x = a*R, y = b*R;
        p1.x = cvRound(x + 1000*(-b));
        p1.y = cvRound(y + 1000*(a));
        p2.x = cvRound(x - 1000*(-b));
        p2.y = cvRound(y - 1000*(a));
        line( img, p1, p2, Scalar(250,225,0), 1, CV_AA);
    }

    nkhImshow_Write(nullImg,img,"Hough Lines",fileName,"houghLines-Sig"+QString::number(params[3]));
}

void nkhCV::nkhHoughCircles(QString fileName)
{
    Mat img;
    if(!nkhImread(img,fileName))
        return;
    Mat src = img.clone();
    GaussianBlur( img, img, Size(5, 5), 1,1 );
    Mat nullImg ;
    nkhImshow_Write(nullImg,img,"Hough Inupt",fileName,"hoghCircInput");
    vector<Vec3f> circles;
    HoughCircles( img, circles, CV_HOUGH_GRADIENT, 0.1, img.rows/20, 100, 39, 1,100 );
    cvtColor(src.clone(),src,CV_GRAY2BGR);
    for( int i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( src, center, 3, Scalar(250,250,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, Scalar(250,225,0), 3, 8, 0 );
    }

    nkhImshow_Write(nullImg,src,"Hough Circle",fileName,"hoghCirc");
}
void nkhCV::nkhFindContours(QString fileName,int params[])
{
    Mat img;
    if(!nkhImread(img,fileName))
        return;
    Mat src = img.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> order;
    GaussianBlur( img, img, Size(5, 5), 1,1 );
    Mat nullImg ;
    Mat edges = edgeDetect(fileName,"Edge Canny",EDGE_DETECT_CANNY,params, nullImg);
    findContours( edges, contours, order, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Mat showContours = Mat::zeros( edges.size(), CV_8UC3 );
    RNG rng(12345);
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( showContours, contours, i, color, 2, 8, order, 0, Point() );
    }
    nkhImshow_Write(nullImg,showContours,"Contours",fileName,"contours");
}

Mat nkhCV::addGaussianNoise(Mat &img, int mean, int sigma)
{
    Mat gaussian_noise = img.clone();
    randn(gaussian_noise,mean,sigma);
    threshold(gaussian_noise,gaussian_noise,200,255,1);
    addWeighted(img,0.5,gaussian_noise,0.5,0,img);
    return img.clone();
}

void nkhCV::nkhKmeans(QString fileName, int params[])
{
    Mat img , nullImage;
    if(!nkhImread(img,fileName,true))
        return;
    Mat dst =img.clone() ;

    /*
    Mat kernel = getGaborKernel(Size(7,7), 20, 10,10,10);
    //ToDo
    //make a list of kernels with directions an frequncies, maybe 12 kernels are enough.
    Mat src_f;// img converted to float
    img.convertTo(src_f,CV_32F);
    cv::filter2D(src_f, dst, CV_32F, kernel);
    //ToDo
    //divide image to regions, for each region, convolve with kernels, get the var and miu as region's feature
    */

    for(int i = 0 ; i < params[3] ; i++)
        medianBlur(dst,dst,params[2]);

    //nkhImshow_Write(img,dst,"Median Blur",fileName,"medianblur");

    int channels= dst.channels();
    int pointsNum = dst.rows * dst.cols ;
    Mat points(dst.total(), 3, CV_32F);

    points = dst.reshape(channels,pointsNum).clone() ;
    points.convertTo(points,CV_32F);
    Mat labels;
    Mat centers;

    /*
    QMessageBox::information(ui,"Err",
                             QString::number(points.dims) + " - " + QString::number(params[0]) ,
                             QMessageBox::Ok|QMessageBox::Default
                            );
    */
    kmeans(points,params[0],labels,
            cv::TermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER,
                             1000, 1),
            params[1],cv::KMEANS_PP_CENTERS, centers
            );

    // map the centers
    cv::Mat new_image(dst.size(), dst.type());
    /*
    for (int i=0 ; i<img.rows ; i++)
    {
        for(int j=0 ; j<img.cols ; j++)
        {
            new_image.at<Vec3b>(j,i)[0]=
                    centers.at<Vec2b>(labels.at<Vec2b>(i*img.cols+j)[0])[0];
            new_image.at<Vec3b>(j,i)[1]=
                    centers.at<Vec2b>(labels.at<Vec2b>(i*img.cols+j)[0])[1];
            new_image.at<Vec3b>(j,i)[2]=
                    centers.at<Vec2b>(labels.at<Vec2b>(i*img.cols+j)[0])[2];
        }
    }
    */
    //Got this from internet, I got an exeption from nowhere in my own method :|
    for( int row = 0; row != dst.rows; ++row){
        auto new_image_begin = new_image.ptr<uchar>(row);
        auto new_image_end = new_image_begin + new_image.cols * 3;
        auto labels_ptr = labels.ptr<int>(row * dst.cols);

        while(new_image_begin != new_image_end){
            int const cluster_idx = *labels_ptr;
            auto centers_ptr = centers.ptr<float>(cluster_idx);
            new_image_begin[0] = centers_ptr[0];
            new_image_begin[1] = centers_ptr[1];
            new_image_begin[2] = centers_ptr[2];
            new_image_begin += 3; ++labels_ptr;
        }
    }

    cv::Mat binary;
    Mat toCanny;
    new_image.convertTo(toCanny,CV_8U);
    cv::Canny(toCanny, binary, 30, 90);

    nkhImshow_Write(dst,new_image,"Original",fileName,"blured"+QString::number(params[2])+"-"+QString::number(params[3])+QString::number(params[0]));
    nkhImshow_Write(dst,binary,"Final",fileName,"final"+QString::number(params[2])+"-"+QString::number(params[3])+QString::number(params[0]));
    cvWaitKey();
}

void nkhCV::nkhSIFTMatch(QString fileName1, QString fileName2, int params[])
{
    Mat img1, img2, dst1, dst2, nullImage;
    if(!nkhImread(img1, fileName1) || !nkhImread(img2, fileName2))
        return;

    nkhImproc(img1, img1, NORM_CMD);
    nkhImproc(img2, img2, NORM_CMD);

    SiftFeatureDetector siftDetector;
    SIFT mysift(0, 3, 0.04, 10, 1);
    std::vector<cv::KeyPoint> siftKeypointsSrc, siftKeypointsCmp;

    Mat descriptorsSrc, descriptorsCmp;
    mysift(img1, Mat(), siftKeypointsSrc, descriptorsSrc);
    mysift(img2, Mat(), siftKeypointsCmp, descriptorsCmp);

    //siftDetector.detect(img1, siftKeypointsSrc);
    drawKeypoints(img1, siftKeypointsSrc, dst1);
    //siftDetector.detect(img2, siftKeypointsCmp);
    drawKeypoints(img2, siftKeypointsCmp, dst2);

    //SiftDescriptorExtractor siftExtractor;

    //siftExtractor.compute( img1, siftKeypointsSrc, descriptorsSrc );
    //siftExtractor.compute( img2, siftKeypointsCmp, descriptorsCmp );

    FlannBasedMatcher flannMatcher;
    std::vector< DMatch > featureMatches;
    flannMatcher.match( descriptorsSrc, descriptorsCmp, featureMatches );
    double maxDist = 0, minDist = params[0];
    for( int i = 0; i < descriptorsSrc.rows; i++ )
    {
        double dist = featureMatches[i].distance;
        if( dist < minDist )
            minDist = dist;
        if( dist > maxDist )
            maxDist = dist;
    }
    std::vector< DMatch > goodMatches;
    for( int i = 0; i < descriptorsSrc.rows; i++ )
    {
        if( featureMatches[i].distance <= max(2*minDist, 0.02) )
        {
            goodMatches.push_back( featureMatches[i]);
        }
    }
    Mat imgMatches;
    drawMatches( img1, siftKeypointsSrc, img2, siftKeypointsCmp,
                 goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    nkhImshow_Write(nullImage,dst1,"src sift",fileName1,"sift-src");
    nkhImshow_Write(nullImage,dst2,"cmp sift",fileName2,"sift-cmp");
    nkhImshow_Write(nullImage,imgMatches,"Final Matches",fileName1,"sift-match");

}

void nkhCV::nkhHOGPeople(int params[])
{
    hogDetectFlag = true ;
    hogProgress = 0 ;
    QDir myDir("./pedestrians128x64");
    QStringList filesList = myDir.entryList();
    int totalFound = 0;
    //falsePositive
    /*
     QMessageBox::information(ui,"File List",
                             QString(("sas"+filesList[2])),
                             QMessageBox::Ok|QMessageBox::Default
                             );
                             */
    int sample = 2 ;
    for (sample = 2 ; sample<filesList.size() && hogDetectFlag ; sample++)//skip . and ..
    {
        Mat img,srcColor;
        QString currFile(QString("./pedestrians128x64")+QString("/")+filesList[sample]);

        if(!nkhImread(srcColor, currFile,true))
            return;
        resize(srcColor,srcColor,Size(128,256));

        cvtColor(srcColor,img, CV_RGB2GRAY);

        HOGDescriptor hog;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

        vector<Rect> found, found_filtered;

        hog.detectMultiScale(img, found, 0 , Size(8,8), Size(32,32), 1.05, 2);

        /*
        QMessageBox::information(ui,"File List",
                                 (QString("sas")+ QString::number(filesList.size())),
                                 QMessageBox::Ok|QMessageBox::Default
                                 );
                                 */

        size_t i, j;
        if(found.size()>0)
            totalFound++;
        for (i=0; i<found.size(); i++)
        {

            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        for (i=0; i<found_filtered.size(); i++)
        {
            Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.06);
            r.height = cvRound(r.height*0.9);
            rectangle(srcColor, r.tl(), r.br(), cv::Scalar(219,252,0), 2);
        }
        imwrite("./output/"+filesList[sample].toStdString(), srcColor);
        hogProgress = sample ;

        imshow("Detecting People...",srcColor);
        cvWaitKey(30);
        emitHogProgress(100*hogProgress/((int)filesList.size()-2),(100*((double)totalFound/(sample-2))));
    }
    if(hogDetectFlag == false )
    {
        QMessageBox::information(ui,"Detection Aborted!",
                                 QString(("Detection process has been aborted.")),
                                 QMessageBox::Ok|QMessageBox::Default
                                 );
    }
    if(sample-2>0)
    {
        QMessageBox::information(ui,"Detection Completed!",
                                 QString(("With "+QString::number(sample-2)+
                                          " images, detection rate is "+ QString::number(100*((double)totalFound/(sample-2))) + " %")),
                                 QMessageBox::Ok|QMessageBox::Default
                                 );
    }
    else
    {

        QMessageBox::information(ui,"Detection Failed!",
                                 QString(("Make sure if your folder is not empty!")),
                                 QMessageBox::Ok|QMessageBox::Default
                                 );
    }
}
void nkhCV::emitHogProgress(int hogProg, int successRate)
{
    emit hogProgressChanged(hogProg);
    emit hogSuccessChanged(successRate);
}

void nkhCV::nkhChainCode(QString fileName)
{
    int params[]={200,255,3};
    Mat img;
    if(!nkhImread(img,fileName))
        return;

    GaussianBlur( img, img, Size(5, 5), 1,1 );
    Mat nullImg ;
    Mat edges = edgeDetect(fileName,"Edge Canny",EDGE_DETECT_CANNY,params, nullImg);
    QString chainCode("");
    IplImage IplEdges = edges;

    CvChain* chain=0;
    CvMemStorage* storage=0;
    storage=cvCreateMemStorage(0);
    cvFindContours( &IplEdges, storage, (CvSeq**)(&chain), sizeof(*chain),
                    CV_RETR_EXTERNAL, CV_CHAIN_CODE );
    while(chain!=NULL)
    {
        CvSeqReader reader;
        int i, total = chain->total;
        cvStartReadSeq((CvSeq*)chain,&reader,0);
        for(i=0;i<total;i++)
        {
            char code;
            CV_READ_SEQ_ELEM(code, reader);
            chainCode.append(QString::number((int)code));
        }
        chain=(CvChain*)chain ->h_next;
    }

    QFile file(QFileInfo(fileName).completeBaseName()+".chain");
    if (file.open(QIODevice::ReadWrite)) {
        QTextStream out(&file);
        out << chainCode;
    }

    QMessageBox::information(ui,"Freeman Chain Code",
                             QString(("The code has a lenght of "
                                      + QString::number(chainCode.length())
                                      + " digits and is written in "
                                      + QFileInfo(fileName).completeBaseName()+".chain")),
                             QMessageBox::Ok|QMessageBox::Default
                             );

}

void nkhCV::nkhFourier(QString fileName)
{
    int params[]={200,255,3};
    Mat img;
    if(!nkhImread(img,fileName))
        return;
    GaussianBlur( img, img, Size(5, 5), 1,1 );
    Mat nullImg ;
    Mat edges = edgeDetect(fileName,"Edge Canny",EDGE_DETECT_CANNY,params, nullImg);
    Mat optimalImg;
    int optimalRows = getOptimalDFTSize( edges.rows ),
            optimalCols = getOptimalDFTSize( edges.cols );
    copyMakeBorder(edges, optimalImg, 0, optimalRows - edges.rows, 0, optimalCols - edges.cols,
                   BORDER_CONSTANT, Scalar::all(0));
    Mat ftParts[] = {Mat_<float>(optimalImg), Mat::zeros(optimalImg.size(), CV_32F)};
    Mat ftMatrix;
    merge(ftParts, 2, ftMatrix);
    dft(ftMatrix, ftMatrix);
    split(ftMatrix, ftParts);// first part is real , second part is imaginary
    magnitude(ftParts[0], ftParts[1], ftParts[0]);
    Mat magnitudeImage = ftParts[0];

    magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2,
                                         magnitudeImage.rows & -2));
    int centerX = magnitudeImage.cols/2, centerY = magnitudeImage.rows/2;

    Mat ftq0(magnitudeImage, Rect(0, 0, centerX, centerY));
    Mat ftq1(magnitudeImage, Rect(centerX, 0, centerX, centerY));
    Mat ftq2(magnitudeImage, Rect(0, centerY, centerX, centerY));
    Mat ftq3(magnitudeImage, Rect(centerX, centerY, centerX, centerY));

    Mat temp; // swap topLeft with botRight
    ftq0.copyTo(temp);
    ftq3.copyTo(ftq0);
    temp.copyTo(ftq3);

    ftq1.copyTo(temp); // swap topRight with botLeft
    ftq2.copyTo(ftq1);
    temp.copyTo(ftq2);

    normalize(magnitudeImage, magnitudeImage, 0, 1, CV_MINMAX);
    nkhImshow_Write(img,magnitudeImage,"Fourier",fileName,"Fourier");
}

void nkhCV::nkhRectEll(QString fileName)
{
    Mat img, nullImg;
    if(!nkhImread(img,fileName))
        return;
    int params[]={200,255,3};
    Mat edges = edgeDetect(fileName,"Edge Canny",EDGE_DETECT_CANNY,params, nullImg);
    Mat threshold_output = edges;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;


    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( Mat(contours[i]) );

        if( contours[i].size() > 5 )
        {
            minEllipse[i] = fitEllipse( Mat(contours[i]) );
        }
    }
    RNG rng(12345);
    /// Draw contours + rotated rects + ellipses
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // contour
        drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        // ellipse
        ellipse( drawing, minEllipse[i], color, 2, 8 );
        // rotated rectangle
        Point2f rect_points[4]; minRect[i].points( rect_points );
        for( int j = 0; j < 4; j++ )
            line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

    }
    nkhImshow_Write(nullImg,drawing,"Rectangle and Ellipse",fileName,"RectEll");
}

void nkhCV::nkhSkel(QString fileName, bool invert)
{
    int params[]={200,255,3};
    Mat img;
    if(!nkhImread(img,fileName))
        return;


    Mat nullImg , pDst ,pSrc, image = img.clone();
    //img.convertTo(pDst,CV_32F);
    threshold(image, image, 200, 255, ((invert)? CV_THRESH_BINARY_INV : CV_THRESH_BINARY));

    pSrc = image.clone();
    thinningGuoHall(image);

    nkhImshow_Write(pSrc,image,"Skeleton",fileName,"Skel");
}

void nkhCV::thinningGuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                    (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void nkhCV::thinningGuoHall(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;


    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}
