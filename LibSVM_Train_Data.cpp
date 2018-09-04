// LibSVM_Train_Data.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

using namespace cv;
using namespace std;

void getRotationInvariantLBPFeature(Mat src_image, Mat dst_image,int radius,int neighbors);
void calcFeatures(const Mat &imgSrc, vector<float> &features);
void generate_sample_list(int posNum, int negNum);
void generateTrainingData(int nClass, int nDims, int posNum, int negNum);

//LIBSVM ��ѵ�����ݸ�ʽ�Ͳ������ݸ�ʽ��һ����

int _tmain(int argc, _TCHAR* argv[])
{
	int PosSampleNum = 875;					//����������
	int NegSampleNum = 2912;					//����������
	int nSamples = PosSampleNum + NegSampleNum;	//��������
	int nWidth = 24;							//�������
	int nHeight = 12;							//�����߶�
	int nDims = nWidth * nHeight;				//����ά��
	int nClass = 2;								//�������

	generateTrainingData(nClass, nDims, PosSampleNum, NegSampleNum);

	return 0;
}


void calcFeatures(const Mat &imgSrc, vector<float> &features)
{
	if (imgSrc.empty())
	{
		cout << "Invalid Input!" << endl;
		return ;
	}

	Mat gray_image(imgSrc.size(), CV_8UC1);
	cvtColor(imgSrc, gray_image, CV_BGR2GRAY);

	Mat lbp_image(imgSrc.size(), CV_8UC1);
	getRotationInvariantLBPFeature(gray_image, lbp_image, 3, 8);

	for (int i = 0; i < lbp_image.rows; i++)
	{
		for (int j = 0; j < lbp_image.cols; j++)
		{
			features.push_back((float)lbp_image.at<uchar>(i, j));
		}
	}
}


//��ת����Բ��LBP�������㣬����ʱĬ��neighbors=8��radius >= 1 ��Ϊ����,�뾶ԽС��ͼ������Խ��ϸ��
void getRotationInvariantLBPFeature(Mat src_image, Mat dst_image,int radius,int neighbors)
{
	for(int k = 0; k < neighbors; k++)
	{
		//���������������ĵ������ƫ����rx��ry
		float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
		float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));

		//Ϊ˫���Բ�ֵ��׼��
		//�Բ�����ƫ�����ֱ��������ȡ��
		int x1 = static_cast<int>(floor(rx));
		int x2 = static_cast<int>(ceil(rx));
		int y1 = static_cast<int>(floor(ry));
		int y2 = static_cast<int>(ceil(ry));

		//������ƫ����ӳ�䵽0-1֮��
		float tx = rx - x1;
		float ty = ry - y1;

		//����0-1֮���x��y��Ȩ�ؼ��㹫ʽ����Ȩ�أ�Ȩ�����������λ���޹أ��������Ĳ�ֵ�й�
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		//ѭ������ÿ������
		for(int i = radius; i < src_image.rows - radius; i++)
		{
			for(int j = radius; j < src_image.cols - radius; j++)
			{
				//����������ص�ĻҶ�ֵ
				uchar center = src_image.at<uchar>(i,j);

				//����˫���Բ�ֵ��ʽ�����k��������ĻҶ�ֵ
				float neighbor = src_image.at<uchar>(i + x1, j + y1) * w1 + src_image.at<uchar>(i + x1, j + y2) * w2 
					+ src_image.at<uchar>(i + x2, j + y1) * w3 + src_image.at<uchar>(i + x2, j + y2) * w4;

				//LBP����ͼ���ÿ���ھӵ�LBPֵ�ۼӣ��ۼ�ͨ���������ɣ���Ӧ��LBPֵͨ����λȡ��
				dst_image.at<uchar>(i - radius,j - radius) |= (neighbor > center) << (neighbors - k - 1);
			}
		}
	}

	//������ת���䴦��
	for(int i = 0; i < dst_image.rows; i++)
	{
		for(int j = 0; j < dst_image.cols; j++)
		{
			uchar currentValue = dst_image.at<uchar>(i, j);
			uchar minValue = currentValue;

			for(int k = 1; k < neighbors; k++)
			{
				//ѭ������
				uchar temp = (currentValue >> (neighbors - k)) | (currentValue << k);
				if(temp < minValue)
				{
					minValue = temp;
				}
			}
			dst_image.at<uchar>(i, j) = minValue;
		}
	}
}


void generate_sample_list(int posNum, int negNum)
{
	char imageName[100];
	FILE* pos_fp;
	pos_fp = fopen("PositiveSamplesList.txt","wb+");
	for (int i = 1; i <= posNum; i++)
	{
		sprintf(imageName,"%d.jpg",i);
		fprintf(pos_fp,"%s\r\n",imageName);
	}
	fclose(pos_fp);

	FILE* neg_fp;
	neg_fp = fopen("NegativeSamplesList.txt","wb+");
	for (int i = 1; i <= negNum; i++)
	{
		sprintf(imageName,"%d.jpg",i);
		fprintf(neg_fp,"%s\r\n",imageName);
	}
	fclose(neg_fp);
}


void generateTrainingData(int nClass, int nDims, int posNum, int negNum)
{
	int number = 0;
	int nCount = 0;
	Mat input_image;
	vector<float> features;
	vector<float> labels;

	generate_sample_list(posNum, negNum);//�������������ļ����б�
	string ImgName;//ͼƬ��(����·��)
	ifstream finPos("PositiveSamplesList.txt");//������ͼƬ���ļ����б�
	ifstream finNeg("NegativeSamplesList.txt");//������ͼƬ���ļ����б�

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����

	for(int i = 0; i < posNum && getline(finPos,ImgName); i++)
	{
		ImgName = "..\\pos-2\\" + ImgName;//������������·����
		input_image = imread(ImgName);//��ȡͼƬ
		calcFeatures(input_image, features);
	}
	cout << "Finished processing positive samlpes !" << endl;

	for (int j = 0; j < negNum && getline(finNeg,ImgName); j++)
	{
		ImgName = "..\\neg-2\\" + ImgName;//������������·����
		input_image = imread(ImgName);//��ȡͼƬ
		calcFeatures(input_image, features);
	}
	cout << "Finished processing negative samlpes !" << endl;

	//write the feature data into a txt file, the format must refer to libliner's reference 
	FILE * fp;
	fp = fopen("samples.txt","wb+");//����һ��txt�ļ�������д�����ݵģ�ÿ��д������׷�ӵ��ļ�β

	for (int m = 0; m < (posNum + negNum); m++)
	{
		if (m < posNum)
		{
			int lable = 1;		//	positive sample lable 1
			fprintf(fp,"%d ",lable);
		}
		else
		{
			int lable = -1;		//	negative sample lable -1
			fprintf(fp,"%d ",lable); 
		}

		for(int n = 0; n < nDims; n++)
		{
			fprintf(fp,"%d:%f ",(n+1),features.at(m * nDims + n));
		}
		fprintf(fp,"\r\n");
	}
	fclose(fp);

	cout << "Generate Training Data Complete!" << endl << endl;
}

