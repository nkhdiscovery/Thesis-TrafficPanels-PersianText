#-------------------------------------------------
#
# Project created by QtCreator 2014-05-20T14:55:49
#
#-------------------------------------------------

QT       += core gui

CONFIG += C++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Phase1-Widget
TEMPLATE = app

INCLUDEPATH +=/usr/include/opencv
INCLUDEPATH +=/usr/include/opencv2

LIBS += -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -ltbb /lib64/libXext.so /lib64/libX11.so /lib64/libICE.so /lib64/libSM.so /lib64/libGL.so /lib64/libGLU.so -lrt -lpthread -lm -ldl



SOURCES += main.cpp\
        mainwindow.cpp \
    nkhCV.cpp

HEADERS  += mainwindow.h \
    nkhCV.h

FORMS    += mainwindow.ui
