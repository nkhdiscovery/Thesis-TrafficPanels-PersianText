#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "nkhCV.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cvObj=new nkhCV(this);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{

    QString fileName1 = ui->lineEdit_SIFTsrc->text() ;
    cvObj->greenMask(fileName1,false);

}

void MainWindow::on_btn_fourier_clicked()
{
    QString fileName1 = ui->lineEdit_SIFTsrc->text() ;
    cvObj->greenMask(fileName1, true);
}

void MainWindow::on_pushButton_4_clicked()
{
    QString fileName1 = ui->lineEdit_SIFTsrc->text() ;
    cvObj->nkhRectEll(fileName1);
}

void MainWindow::on_pushButton_2_clicked()
{
    bool invert = (ui->checkBox->isChecked())? true : false ;
    QString fileName1 = ui->lineEdit_SIFTsrc->text() ;
    cvObj->nkhSkel(fileName1, invert);
}
