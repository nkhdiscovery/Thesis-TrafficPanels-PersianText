#ifndef NKHCV_H
#define NKHCV_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/cv.h>
#include <QString>
#include <QMainWindow>
#include <QTextStream>

class nkhCV : public QObject //images used with these class are assumed to be grayscale
{
    Q_OBJECT
public:
    nkhCV(QMainWindow* parent);

    virtual ~nkhCV(){};
    void normalizeHist(QString fileName,QString windowName);
    void nkhEqualizeHist(QString fileName,QString windowName);
    void showHist(QString fileName,QString windowName);
    cv::Mat edgeDetect(QString fileName, QString windowName, QString edgeMethod, int params[], cv::Mat &imgRef);
    cv::Mat addGaussianNoise(cv::Mat &img,int mean,int sigma);
    void nkhHoughLines(QString fileName,int params[]);
    void nkhHoughCircles(QString fileName);
    void nkhFindContours(QString fileName,int params[]);
    void nkhKmeans(QString fileName, int params[]);
    void nkhSIFTMatch(QString fileName1, QString fileName2, int params[]);
    void nkhHOGPeople(int params[]);
    void nkhChainCode(QString fileName);
    void nkhFourier(QString fileName);
    void nkhRectEll(QString fileName);
    void nkhSkel(QString fileName,bool invert);

    //Final Thesis
    void nkhCIELAB(QString fileName);
    void greenMask(QString fileName, bool video);

    QString NORM_CMD = "normalHist";
    QString EQHIST_CMD = "equalHist";
    QString HIST_IMG_CMD = "showHist";
    QString EDGE_DETECT_SOBEL = "edgeSobel";
    QString EDGE_DETECT_SOBELX = "edgeSobelX";
    QString EDGE_DETECT_SOBELY = "edgeSobelY";
    QString EDGE_DETECT_NEVATIA_BABU = "edgeNevBabu";
    QString EDGE_DETECT_CANNY = "edgeCanny";

signals:
    void hogProgressChanged(int newValue);
    void hogSuccessChanged(int newValue);


private:
    QMainWindow* ui;
    void nkhImproc(cv::Mat &src, cv::Mat &dst, QString command);
    bool nkhImread(cv::Mat &dst, QString fileName, bool color=false);
    void nkhImshow_Write(cv::Mat &orig, cv::Mat& proc,QString windowName,QString fileName,QString PROC_CMD);
    cv::Mat nkhCalcHist(cv::Mat &src);
    void nkhDrawHist(cv::Mat &hist, cv::Mat &dst);
    void nkhFilter2D(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel);
    cv::Mat nkhSobelFilterX();
    void emitHogProgress(int hogProg, int hogSuccess);
    int hogProgress ;
    bool hogDetectFlag;
    void thinningGuoHall(cv::Mat& im);
    void thinningGuoHallIteration(cv::Mat& im, int iter);

    //final thesis
    void toCIELAB(cv::Mat &src, cv::Mat &dst);
    void colorMask(cv::Mat &src, cv::Mat &dst, std::string color, bool video);

};

#endif // NKHCV_H
