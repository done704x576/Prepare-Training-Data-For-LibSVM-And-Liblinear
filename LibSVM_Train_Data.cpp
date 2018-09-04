// LibSVM_Train_Data.cpp : 定义控制台应用程序的入口点。
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

//LIBSVM 的训练数据格式和测试数据格式是一样的

int _tmain(int argc, _TCHAR* argv[])
{
	int PosSampleNum = 875;					//正样本个数
	int NegSampleNum = 2912;					//负样本个数
	int nSamples = PosSampleNum + NegSampleNum;	//样本总数
	int nWidth = 24;							//样本宽度
	int nHeight = 12;							//样本高度
	int nDims = nWidth * nHeight;				//特征维数
	int nClass = 2;								//总类别数

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


//旋转不变圆形LBP特征计算，声明时默认neighbors=8。radius >= 1 且为整数,半径越小，图像纹理越精细。
void getRotationInvariantLBPFeature(Mat src_image, Mat dst_image,int radius,int neighbors)
{
	for(int k = 0; k < neighbors; k++)
	{
		//计算采样点对于中心点坐标的偏移量rx，ry
		float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
		float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));

		//为双线性插值做准备
		//对采样点偏移量分别进行上下取整
		int x1 = static_cast<int>(floor(rx));
		int x2 = static_cast<int>(ceil(rx));
		int y1 = static_cast<int>(floor(ry));
		int y2 = static_cast<int>(ceil(ry));

		//将坐标偏移量映射到0-1之间
		float tx = rx - x1;
		float ty = ry - y1;

		//根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		//循环处理每个像素
		for(int i = radius; i < src_image.rows - radius; i++)
		{
			for(int j = radius; j < src_image.cols - radius; j++)
			{
				//获得中心像素点的灰度值
				uchar center = src_image.at<uchar>(i,j);

				//根据双线性插值公式计算第k个采样点的灰度值
				float neighbor = src_image.at<uchar>(i + x1, j + y1) * w1 + src_image.at<uchar>(i + x1, j + y2) * w2 
					+ src_image.at<uchar>(i + x2, j + y1) * w3 + src_image.at<uchar>(i + x2, j + y2) * w4;

				//LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
				dst_image.at<uchar>(i - radius,j - radius) |= (neighbor > center) << (neighbors - k - 1);
			}
		}
	}

	//进行旋转不变处理
	for(int i = 0; i < dst_image.rows; i++)
	{
		for(int j = 0; j < dst_image.cols; j++)
		{
			uchar currentValue = dst_image.at<uchar>(i, j);
			uchar minValue = currentValue;

			for(int k = 1; k < neighbors; k++)
			{
				//循环左移
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

	generate_sample_list(posNum, negNum);//生成正负样本文件名列表
	string ImgName;//图片名(绝对路径)
	ifstream finPos("PositiveSamplesList.txt");//正样本图片的文件名列表
	ifstream finNeg("NegativeSamplesList.txt");//负样本图片的文件名列表

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人

	for(int i = 0; i < posNum && getline(finPos,ImgName); i++)
	{
		ImgName = "..\\pos-2\\" + ImgName;//加上正样本的路径名
		input_image = imread(ImgName);//读取图片
		calcFeatures(input_image, features);
	}
	cout << "Finished processing positive samlpes !" << endl;

	for (int j = 0; j < negNum && getline(finNeg,ImgName); j++)
	{
		ImgName = "..\\neg-2\\" + ImgName;//加上正样本的路径名
		input_image = imread(ImgName);//读取图片
		calcFeatures(input_image, features);
	}
	cout << "Finished processing negative samlpes !" << endl;

	//write the feature data into a txt file, the format must refer to libliner's reference 
	FILE * fp;
	fp = fopen("samples.txt","wb+");//创建一个txt文件，用于写入数据的，每次写入数据追加到文件尾

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

