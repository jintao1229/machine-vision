#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stack>
#include <math.h> 
using namespace cv;
using namespace std;

#define Pai  3.1415926
// this is a path of image
const char * dir = "C:/Users/45479/Desktop/progress/";

int compute_hist(Mat & dst_result_R, int&, int&);
Mat retinex(Mat src, int num_X, int num_Y);
int enhance_gray(Mat & src, int x1, int x2);

uchar sort_K(int array[], int K);
int isodata(const std::vector<int>& data);
int*  compute_hist(Mat & dst_result_R);
vector <int>  compute_hist2(Mat & dst_result_R);
int HuangLKThreshold(int* data);
int Percentile(int data[]);

Scalar getMSSIM(Mat  inputimage1, Mat inputimage2);
double getMSE(Mat  inputimage1, Mat inputimage2);
double getPSNR(double MSE);

int find_mincircle_mintubao(const char * dir_img);
vector<float> find_circle(Mat image);

vector<vector<float>>  get_eigenvalue(Mat bin, Mat src, vector<float> cir);

int main(int argc, char** argv)
{
	double t0 = getTickCount();
	std::vector<String> files;
	string filepath_Depth = dir;
	glob(filepath_Depth, files, false);
	size_t count = files.size();
	cout << "目录下一共有 "<<count << "张图片"<< endl<<endl;
	int tmp = 0;
	for (int i = 0; i < count; i++) {
		const char * dir_work = files[i].c_str();
		int progress = tmp;
		cout <<dir_work << " is starting!" << endl;
		Mat src = imread(dir_work, 0);
		if (src.empty()) {
			cout << "src is empty" << endl;
			system("pause");
		}
		double t = getTickCount();
		int num_X = src.rows, num_Y = src.cols;
		Mat img_enhance = retinex(src, num_X, num_Y);

		double t1 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "##去光照算法 -" << progress << "- 结束，花费时间： " << t1 << "ms" << endl;
		progress++;

		char chImageName[64] = { 0 };
		sprintf_s(chImageName, 64, "C:/Users/45479/Desktop/%d.bmp", progress);
		int x1 = 255, x2 = 255;
		compute_hist(img_enhance, x1, x2);
		enhance_gray(img_enhance, x1, x2);
		//imwrite(chImageName, img_enhance);

		double t2 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "###对比度增强 -"<< progress <<"- 结束，花费时间： " << t2-t1 << "ms" << endl;
		progress++;
		
		Mat src_Roi(img_enhance, Rect(460, 80, 1030, 1030));
		GaussianBlur(src_Roi, src_Roi, Size(5, 5), 1.0);
		if (src_Roi.empty()) {
			cout << "empty" << endl;
		}
		int * data = NULL;
		data = compute_hist(src_Roi);
		vector<int> data2 = compute_hist2(src_Roi);
		Mat bin_image(src_Roi.size(), CV_8UC1);
		//int h = HuangLKThreshold(data);
		int p = Percentile(data);
		//int iso = isodata(data2);
		delete data;
		
		double t3 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "###获取阈值（统计） -" << progress << "- 结束，花费时间： " << t3 - t2 << "ms" << endl;
		progress++;

		Mat useful_point(src.size(), CV_8UC1);
		vector<float> cir = find_circle(img_enhance);
		Point center((int)cir[1], (int)cir[0]);
		float radius = cir[2];
		//circle(useful_point, center, radius*0.57, Scalar(255, 255, 255), 2, 8, 0);
		//circle(useful_point, center, radius*0.94, Scalar(255, 255, 255), 2, 8, 0);
		
		double t4 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "###最小外接圆 -" << progress << "- 结束，花费时间： " << t4 - t3 << "ms" << endl;
		cout << "   圆心位置在 (" << cir[0] << "," << cir[1] << ") 圆半径是： " << radius << endl;
		progress++;
		
		for (int r = 0; r < src.rows; r++) {
			int leny = (r - center.y)*(r - center.y);
			for (int c = 0; c < src.cols; c++) {
				//useful_point.at<uchar>(r, c) = 100;
				int lenx = (c - center.x)*(c - center.x);
				int len = lenx + leny;
				if (len > radius*0.57*radius *0.57 && len < radius *0.94*radius *0.94) {
					if(src.at<uchar>(r, c) > p)
						useful_point.at<uchar>(r, c) = 255;
				}
			}
		}
		char chImageName1[64] = { 0 };
		sprintf_s(chImageName1, 64, "C:/Users/45479/Desktop/%d.bmp", progress);
		//imwrite(chImageName1, useful_point);
		double t5 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "###图像二值化部分提取 -" << progress << "- 结束，花费时间： " << t5 - t4 << "ms" << endl;
		progress++;

		vector< vector<float> > par = get_eigenvalue(useful_point, src, cir);
		double t6 = (getTickCount() - t) * 1000 / getTickFrequency();
		cout << "###图像特征值的提取 -" << progress << "- 结束，花费时间： " << t6 - t5 << "ms" << endl;
		progress++;

		cout << "   区域总个数：" << par.size() << endl;
		for (int siz = 0; siz < par.size(); siz++) {
			vector<float> eigen = par[siz];
			cout << "   面积 = " << eigen[0] << "\t周长 = " << eigen[1] << "\t  延展率 = " << eigen[2] << "  \t  前景均值 = " << eigen[3] << "   \t  灰度偏移比 = " << eigen[4] << endl;
			//cout << " " << eigen[0] << "\t" << eigen[1] << "\t" << eigen[2] << "\t" << eigen[3] << "\t" << eigen[4] << endl;
		}
		progress++;


		//设置矩阵A。  添加矩阵B   最终计算矩阵W    A * W = B
		Mat A(10, 6, CV_32FC1);
		Mat W(6, 1, CV_32FC1);
		Mat B(10, 1, CV_32FC1);
		if (par.size() > 10) {
			for (int i = 0; i < 10; i++) {
				vector<float> t = par[i];
				for (int j = 0; j < 6; j++) {
					A.at<float>(i, j) = t[j];
				}
			}
			B.at<float>(0, 0) = 1.0;
			B.at<float>(1, 0) = 1.0;
			B.at<float>(2, 0) = 1.0;
			B.at<float>(3, 0) = 0.0;
			B.at<float>(4, 0) = 0.0;
			B.at<float>(5, 0) = 1.0;
			B.at<float>(6, 0) = 0.0;
			B.at<float>(7, 0) = 1.0;
			B.at<float>(8, 0) = 1.0;
			B.at<float>(9, 0) = 1.0;
			Mat tmp;
			Mat A_t = A.t();
			cv::invert(A_t*A, tmp, DECOMP_LU);
			W = tmp * A_t*B;
			//cout << fixed <<"最终函数为： y = " << W.at<float>(0, 0) << " * x1 + " << W.at<float>(1, 0) << " * x2 + " << W.at<float>(2, 0) << " * x3 + " << W.at<float>(3, 0) << " * x4 + " << W.at<float>(4, 0) << " * x5 + " << W.at<float>(5, 0);
		}
		


		tmp = tmp + 10;
		cout << endl;
	}

	double total = (getTickCount() - t0) * 1000 / getTickFrequency();
	cout << "***总处理时间 : " << total <<"ms"<< endl;
	getchar();
	getchar();
	return 0;
}
/*
*@brief computer the first point and second point
@param dst_result_R	 input image of hist
@param x1		first point 0
@param x2		second point 255
*/
int compute_hist(Mat & dst_result_R, int &x1, int &x2) {
	MatND dstHist;
	int channels = 0;
	int histSize[] = { 256 };
	float midRanges[] = { 0, 256 };
	const float *ranges[] = { midRanges };
	calcHist(&dst_result_R, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);
	//Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	double g_dHistMaxValue;
	minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
	int mat[256];
	for (int i = 0; i < 256; i++)
	{
		int value = cvRound(dstHist.at<float>(i) * 256 / g_dHistMaxValue);
		mat[i] = value;
	}
	int i = 0;
	x2 = x2 - 60;
	while (mat[i] < 80) {
		i++;
	}
	int k = i;
	uchar min_sum = 255;
	for (k = i; k < i + 15; k++) {
		if (min_sum > mat[k]) {
			min_sum = mat[k];
			x1 = k;
		}
	}
	return 0;
}


/*
*@brief  return the enhance amge, using Retinex 
@param src	 input image t
@param num_X		rows of image
@param num_Y		cols of image
*/
Mat retinex(Mat src, int num_X, int num_Y) {
	/*将原图像转换为双精度图片*/
	Mat dst_double(src.size(), CV_64FC1);
	src.convertTo(dst_double, CV_64FC1, 1 / 255.0);

	/*双精度图片进行取对数，转换至对数域*/
	Mat dst_double_log(dst_double.size(), dst_double.type());   /*log (S)*/
	for (int i = 0; i < num_X; i++) {
		double *pData = dst_double_log.ptr<double>(i);
		double *pSrc = dst_double.ptr<double>(i);
		for (int j = 0; j < num_Y; j++) {
			pData[j] = log(pSrc[j] + 1);
		}
	}

	/*双精度图片对其进行快速傅里叶变换，根据光照模型（高斯）提取log（L）*/
	Mat gauss_mask(src.size(), CV_64FC1);
	double max_gauss = 0;
	double sigma = 200.0;
	double si = 2 * sigma*sigma;
	double dis_square = 205625;
	for (int i = 0; i < src.rows; i++) {
		double *pData2 = gauss_mask.ptr<double>(i);
		for (int j = 0; j < src.cols; j++) {
			double tmp = (i - 600)*(i - 600) + (j - 960)*(j - 960);
			if (tmp < dis_square) {
				tmp = dis_square - tmp;
			}
			else {
				tmp -= dis_square;
			}
			pData2[j] = exp(-tmp / si);
			if (max_gauss < pData2[j]) {
				max_gauss = pData2[j];
			}
		}
	}

	for (int i = 0; i < src.rows; i++) {
		double *pData2 = gauss_mask.ptr<double>(i);
		for (int j = 0; j < src.cols; j++) {
			//归一化至0-255 范围内
			//pData2[j] = pData2[j] * 255 / max_gauss;  
			pData2[j] = pData2[j] * 0.5 / max_gauss;
		}
	}
	Mat dst_gauss_log(gauss_mask.size(), CV_64FC1);   	/*log (L)*/
	for (int i = 0; i < num_X; i++) {
		double *pData = dst_gauss_log.ptr<double>(i);
		double *pGauss = gauss_mask.ptr<double>(i);
		double *pDouble_R = dst_double_log.ptr<double>(i);
		for (int j = 0; j < num_Y; j++) {
			pData[j] = log(pGauss[j] * pDouble_R[j] + 1);
		}
	}

	/*log(R) = log(S) - log(L)*/
	Mat dst_result_log(gauss_mask.size(), CV_64FC1);   	/*log (R)*/
	Mat dst_double_R(gauss_mask.size(), CV_64FC1);
	double max_dst_result_R = 0.0;
	for (int i = 0; i < num_X; i++) {
		double *pData = dst_result_log.ptr<double>(i);
		double *pGauss = dst_gauss_log.ptr<double>(i);
		double *pSrc = dst_double_log.ptr<double>(i);

		double *pres_R = dst_double_R.ptr<double>(i);
		for (int j = 0; j < num_Y; j++) {
			pData[j] = pSrc[j] - pGauss[j];
			pres_R[j] = exp(pData[j]) - 1;
			if (max_dst_result_R < pres_R[j]) {
				max_dst_result_R = pres_R[j];
			}
		}
	}

	/*实域R 归一化显示*/
	Mat dst_result_R(gauss_mask.size(), CV_8UC1);
	for (int i = 0; i < num_X; i++) {
		uchar *pData = dst_result_R.ptr<uchar>(i);
		double *pSrc = dst_double_R.ptr<double>(i);
		for (int j = 0; j < num_Y; j++) {
			pData[j] = (uchar)(pSrc[j] * 255.0 / max_dst_result_R);
		}
	}
	return dst_result_R;
}

/*
*@brief using first point and second point ,enhance image
@param dst_result_R	 input image of hist
@param x1		first point 0
@param x2		second point 255
*/
int enhance_gray(Mat & src, int x1, int x2) {
	for (int i = 0; i < src.rows; i++) {
		uchar * pData = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			if (pData[j] < x1) {
				pData[j] = 0;
			}
			else if (pData[j] > x2) {
				pData[j] = 255;
			}
			else {
				pData[j] = (pData[j] - x1) * 255 / (x2 - x1);
			}
		}
	}
	return 0;
}





int partition(int arr[], int left, int right)  //找基准数 划分
{
	int i = left + 1;
	int j = right;
	int temp = arr[left];
	while (i <= j)
	{
		while (arr[i] < temp)
		{
			i++;
		}
		while (arr[j] > temp)
		{
			j--;
		}
		if (i < j)
			swap(arr[i++], arr[j--]);
		else i++;
	}
	swap(arr[j], arr[left]);
	return j;

}

void quick_sort(int arr[], int left, int right)
{
	if (left > right)
		return;
	int j = partition(arr, left, right);
	quick_sort(arr, left, j - 1);
	quick_sort(arr, j + 1, right);
}

/*
*@brief Returns the Kth value
@param array[]	 input array
@param K		find Kth
*/
uchar sort_K(int array[], int K) {
	quick_sort(array, 0, 8);
	return array[K - 1];
}






/*核心理论
（1） 找到最小值与最大值的中值K
（2） 计算小于K值的灰度均值和大于K值的灰度均值的中值，更新K
（3） 迭代多次步骤（2），最终求得K
*@brief Returns the threshold
@param data	 input hist
*/
int isodata(const std::vector<int>& data) {
	if (data.empty()) {
		return -1;
	}
	int min_value = 255, max_value = 0, min = 0x7fffffff, max = 0;
	for (int i = 0; i < data.size(); i++) {
		if (data[i] < min) {
			min = data[i];
			min_value = i;

		}
		if (data[i] > max) {
			max = data[i];
			max_value = i;
		}
	}
	if (min_value >= max_value) {
		return -1;
	}
	int mid_value = (min_value + max_value) / 2;
	int mid_value2 = mid_value;
	do {
		mid_value = mid_value2;
		int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
		for (int i = 0; i < mid_value; i++) {
			sum1 += data[i] * (i + 1);
			sum2 += data[i];
		}
		min_value = sum1 / sum2 - 1;

		for (int j = 255; j >= mid_value; j--) {
			sum3 += data[j] * (j + 1);
			sum4 += data[j];
		}
		max_value = sum3 / sum4 - 1;

		mid_value2 = (min_value + max_value) / 2;
	} while (mid_value2 != mid_value);

	return mid_value;
}



/*
*@brief Returns hist
@param dst_result_R	 input image
*/
int*  compute_hist(Mat & dst_result_R) {
	//Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	int  *mat = new int[256];
	for (int t = 0; t < 256; t++) {
		mat[t] = 0;
	}
	for (int r = 0; r < dst_result_R.rows; r++) {
		for (int c = 0; c < dst_result_R.cols; c++) {
			int v = dst_result_R.at<uchar>(c, r);
			mat[v] = mat[v] + 1;
		}
	}
	int max = 0;
	for (int t = 0; t < 256; t++) {
		if (mat[t] > max)
			max = mat[t];
	}
	for (int t = 0; t < 256; t++) {
		mat[t] = mat[t] * 255 / max;
	}
	vector <int> hist;
	for (int i = 0; i < 256; i++)
	{
		hist.push_back(mat[i]);
		//line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - mat[i]), Scalar(255, 0, 0));
	}
	mat[0] = 0;
	return mat;
}



/*
*@brief Returns hist
@param dst_result_R	 input image
*/
vector <int>  compute_hist2(Mat & dst_result_R) {
	//Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	int  mat[256];
	for (int t = 0; t < 256; t++) {
		mat[t] = 0;
	}
	for (int r = 0; r < dst_result_R.rows; r++) {
		for (int c = 0; c < dst_result_R.cols; c++) {
			int v = dst_result_R.at<uchar>(c, r);
			mat[v] = mat[v] + 1;
		}
	}
	int max = 0;
	for (int t = 0; t < 256; t++) {
		if (mat[t] > max)
			max = mat[t];
	}
	for (int t = 0; t < 256; t++) {
		mat[t] = mat[t] * 255 / max;
	}
	mat[0] = 0;
	vector <int> hist;
	for (int i = 0; i < 256; i++)
	{
		hist.push_back(mat[i]);
		//line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - mat[i]), Scalar(255, 0, 0));
	}
	return hist;
}


/*
按照论文结构计算不同阈值t下的【小于t的均值为mu_0 大于t的均值为mu_1】
最后根据（4）隶属度（6）模糊的熵（8）最小化模糊性的测量
计算不同阈值t下的模糊性的测量 ，当（8）最小时为阈值t
*@brief Returns threshold
@param data	 hist
*/
int HuangLKThreshold(int* data) {
	int threshold = -1;
	int ih, it;
	int first_bin;
	int last_bin;
	int sum_pix;
	int num_pix;
	double term;
	double ent;  // entropy 
	double min_ent; // min entropy 
	double mu_x;
	first_bin = 0;
	for (ih = 0; ih < 256; ih++) {
		if (data[ih] != 0) {
			first_bin = ih;
			break;
		}
	}

	/* Determine the last non-zero bin */
	last_bin = 255;
	for (ih = 255; ih >= first_bin; ih--) {
		if (data[ih] != 0) {
			last_bin = ih;
			break;
		}
	}
	term = 1.0 / (double)(last_bin - first_bin);
	double mu_0[256] = { 0.0 };
	sum_pix = num_pix = 0;
	for (ih = first_bin; ih < 256; ih++) {
		sum_pix += ih * data[ih];
		num_pix += data[ih];
		/* NUM_PIX cannot be zero ! */
		mu_0[ih] = sum_pix / (double)num_pix;
	}

	double mu_1[256] = { 0.0 };
	sum_pix = num_pix = 0;
	for (ih = last_bin; ih > 0; ih--) {
		sum_pix += ih * data[ih];
		num_pix += data[ih];
		/* NUM_PIX cannot be zero ! */
		mu_1[ih - 1] = sum_pix / (double)num_pix;
	}

	/* Determine the threshold that minimizes the fuzzy entropy */
	threshold = -1;
	min_ent = 65535;
	for (it = 0; it < 256; it++) {
		ent = 0.0;
		for (ih = 0; ih <= it; ih++) {
			/* Equation (4) in Ref. 1 */
			mu_x = 1.0 / (1.0 + term * abs(ih - mu_0[it]));
			if (!((mu_x < 1e-06) || (mu_x > 0.999999))) {
				/* Equation (6) & (8) in Ref. 1 */
				ent += data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x));
			}
		}

		for (ih = it + 1; ih < 256; ih++) {
			/* Equation (4) in Ref. 1 */
			mu_x = 1.0 / (1.0 + term * abs(ih - mu_1[it]));
			if (!((mu_x < 1e-06) || (mu_x > 0.999999))) {
				/* Equation (6) & (8) in Ref. 1 */
				ent += data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x));
			}
		}
		/* No need to divide by NUM_ROWS * NUM_COLS * LOG(2) ! */
		if (ent < min_ent) {
			min_ent = ent;
			threshold = it;
		}
	}
	return threshold;
}

double partialSum(int y[], int j) {
	double x = 0;
	for (int i = 0; i <= j; i++)
		x += y[i];
	return x;
}

/*
统计方法生成阈值
*@brief Returns threshold
@param data	 hist
*/
int Percentile(int data[]) {
	int threshold = -1;
	double sum_gray = 0;
	double total = partialSum(data, 255);
	double temp = 0.01;
	double per = total * temp;
	//cout << total << endl;
	for (int i = 255; i > 0; i--) {
		sum_gray += data[i];
		//cout << "sum = " << sum_gray<<endl;
		if (sum_gray < per) {
			threshold = i;
		}
	}
	return threshold;
}





/*
ssim评价
scalar 中仅有第一个val可用。
*@brief return MSSIM value
@param inputimage1	 src image
@param inputimage2	 dst image
*/
Scalar getMSSIM(Mat  inputimage1, Mat inputimage2)
{
	Mat i1 = inputimage1;
	Mat i2 = inputimage2;
	const double C1 = 6.5025, C2 = 58.5225;
	Mat I1, I2;
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);
	Mat I2_2 = I2.mul(I2);
	Mat I1_2 = I1.mul(I1);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
	return mssim;
}


/*
MSE评价
*@brief return MSE value
@param inputimage1	 src image
@param inputimage2	 dst image
*/
double getMSE(Mat  inputimage1, Mat inputimage2) {
	double MSE = 0;
	for (int r = 0; r < inputimage1.rows; r++) {
		for (int c = 0; c < inputimage2.cols; c++) {
			int t = inputimage1.at<uchar>(r, c) - inputimage2.at<uchar>(r, c);
			MSE = MSE + (double)t*t;
		}
	}
	MSE /= (inputimage1.rows*inputimage1.cols);
	return MSE;
}

/*
PSNR评价
*@brief return PSNR value
@param inputimage1	 src image
@param inputimage2	 dst image
*/
double getPSNR(double MSE) {
	double PSNR = 0;
	PSNR = 20 * log(255.0 / sqrt(MSE));
	return PSNR;
}






/*
寻找最小凸包以及最小外接圆，最终找到环形区域
*@brief find mincircle and find mintubao
@param dir_img	 path of image
*/
int find_mincircle_mintubao(const char * dir_img)
{
	cv::Mat img = imread(dir_img, 0);
	cv::Mat dst2 = Mat::zeros(img.size(), CV_8UC3);
	cv::Mat dst_huan_bin = Mat::zeros(img.size(), CV_8UC1);
	cv::Mat dst_huan_src = Mat::zeros(img.size(), CV_8UC1);
	/*均值滤波*********************begin*/
	cv::Mat blu_img;
	cv::Size size_3;
	size_3.height = 3;
	size_3.width = 3;
	blur(img, blu_img, size_3);
	//imshow("blur", blu_img);
	/*均值滤波*********************end*/

	//waitKey(10);


	/*找出凸包并计算凸包面积************************begin*/
	std::vector<Point2i> t;
	for (int i = 0; i < blu_img.rows; i++) {
		for (int j = 0; j < blu_img.cols; j++) {
			unsigned char f = blu_img.at<uchar>(i, j);
			Point k = { i,j };
			if (f > 90) {
				t.push_back(k);
				dst2.at<Vec3b>(i, j)[0] = 255;
			}
			else {
				dst2.at<Vec3b>(i, j)[0] = 0;
			}
		}
	}
	vector<Point> hull;
	convexHull(t, hull, false);
	vector<Point> tubao(hull.size());
	for (int i = 0; i < hull.size(); i++)
	{
		tubao[i].y = hull[i].x;
		tubao[i].x = hull[i].y;
	}
	Point pt0, pt;
	for (int i = 0; i < tubao.size(); i++)
	{
		pt0 = tubao[i];
		//cout << pt0 <<endl;
		if (pt0 == tubao[tubao.size() - 1]) {
			pt = tubao[0];
			line(dst2, pt0, pt, Scalar(0, 255, 0), 1, CV_AA);
			break;
		}

		pt = tubao[i + 1];
		line(dst2, pt0, pt, Scalar(0, 255, 0), 1, CV_AA);
		pt0 = pt;
	}

	Mat dst3 = dst2.clone();
	long int area_tubao_out = 0;
	for (int i = 0; i < dst3.rows; i++) {
		bool in_tubao = false;
		for (int j = 0; j < dst3.cols; j++) {
			if (dst2.at<Vec3b>(i, j)[1] == 0 && !in_tubao) {
				dst3.at<Vec3b>(i, j)[1] = 255;
			}
			else {
				in_tubao = true;
			}
		}
	}
	for (int i = dst3.rows - 1; i >= 0; i--) {
		bool in_tubao = false;
		for (int j = dst3.cols - 1; j >= 0; j--) {
			if (dst2.at<Vec3b>(i, j)[1] == 0 && !in_tubao) {
				dst3.at<Vec3b>(i, j)[1] = 255;
			}
			else {
				in_tubao = true;
			}
		}
	}
	for (int i = 0; i < dst3.rows; i++) {
		for (int j = 0; j < dst3.cols; j++) {
			if (dst3.at<Vec3b>(i, j)[1] == 255) {
				area_tubao_out++;
			}
		}
	}
	int area_tubao = 2304000 - area_tubao_out;
	cout << "凸包的整体面积为：" << area_tubao << endl;
	/*找出凸包并计算凸包面积************************begin*/



	/*最小外接圆****************************begin*/
	Point2f center = { 100,200 };
	float radius = 100;
	minEnclosingCircle(t, center, radius);
	Point cen = Point(center.y, center.x);


	circle(dst2, cen, radius, Scalar(0, 0, 255), 1, 8, 0);
	circle(dst2, cen, radius*0.56, Scalar(0, 255, 255), 1, 8, 0);
	circle(dst2, cen, radius*0.96, Scalar(0, 255, 255), 1, 8, 0);

	//cout << img.cols;
	for (int r = 0; r < img.rows; r++) {
		int leny = (r - cen.y)*(r - cen.y);
		for (int c = 0; c < img.cols; c++) {
			int lenx = (c - cen.x)*(c - cen.x);
			int len = lenx + leny;
			if (len > radius*0.56*radius*0.56 && len < radius*0.95*radius*0.95) {
				dst_huan_bin.at<uchar>(r, c) = dst2.at<Vec3b>(r, c)[0];
				dst_huan_src.at<uchar>(r, c) = img.at<uchar>(r, c);
			}
		}
	}
	imshow("src", dst_huan_bin);
	imshow("src2", dst_huan_src);
	waitKey();
	int area_cir = 3.1415926*radius*radius;
	//imshow("dst2", dst2);
	//waitKey(1000);
	cout << "最小外接圆的整体面积为：" << area_cir << endl;
	/*最小外接圆******************************end*/

	return area_cir - area_tubao;
}


/*
修改 find_mincircle_mintubao 仅寻找环形区域
*@brief return  point of center ,radius of the two circles
@param image	 input image
*/
vector<float> find_circle(Mat image) {
	vector<float> res;
	std::vector<Point2i> t;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			unsigned char f = image.at<uchar>(i, j);
			Point k = { i,j };
			if (f > 150) {
				t.push_back(k);
			}
		}
	}
	Point2f center = { 100,200 };
	float radius = 100;
	minEnclosingCircle(t, center, radius);
	Point cen = Point(center.y, center.x);
	res.push_back(center.x);
	res.push_back(center.y);
	res.push_back(radius);
	return res;
}




/*
对二值化的图像寻找内部有效点
*@brief return  point in img
@param img	 input image
*/
Point find_one(Mat &img) {
	Point t;

	for (int i = 1; i < img.rows - 2; i++) {
		for (int j = 1; j < img.cols - 2; j++) {
			if (img.at<uchar>(i, j) == 255) {
				t.x = i;
				t.y = j;
				return t;
			}
		}
	}
	t.x = -1;
	t.y = -1;
	return t;
}

/*
根据内部有效点，利用种子扩张的方法提取当前区域的 点集
*@brief return  numbers of points 
@param img	 input image
@param p	 seed
@param u     label
@param area_cur   cur area points
*/
int seed_fill(Mat & img, Point p, uchar u, vector<Point>& area_cur) {
	stack<Point> st;
	st.push(p);
	int num = 0;
	while (!st.empty()) {
		Point p1 = st.top();
		int curx = p1.x;
		int cury = p1.y;
		area_cur.push_back(p1);

		img.at<uchar>(curx, cury) = u;
		st.pop();
		num++;
		if (img.at<uchar>(curx - 1, cury) == 255) {
			st.push(Point(curx - 1, cury));
		}
		if (img.at<uchar>(curx + 1, cury) == 255) {
			st.push(Point(curx + 1, cury));
		}
		if (img.at<uchar>(curx, cury - 1) == 255) {
			st.push(Point(curx, cury - 1));
		}
		if (img.at<uchar>(curx, cury + 1) == 255) {
			st.push(Point(curx, cury + 1));
		}
	}
	return num;
}

 
/*
根据点集计算 周长
*@brief return  circumference
@param area_cur   cur area points
*/
int getCircumference(vector<Point> area_cur) {
	//数据点展示到图像中，利用原图片 - 腐蚀变换 得到最终周长
	Mat src(1920,1200,CV_8UC1);
	int num = 0;
	int size = area_cur.size();
	for (int i = 0; i < size; i++) {
		Point cur = area_cur[i];
		src.at<uchar>(cur.x, cur.y) = 255;
	}
	//imshow("src", src);

	Mat src2(src.size(), src.type());
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(src, src2, element);
	//imshow("src2", src2);
	Mat dst(src.size(), src.type());
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			dst.at<uchar>(r, c) = src.at<uchar>(r, c) - src2.at<uchar>(r, c);
		}
	}
	//imshow("dst", dst);
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (dst.at<uchar>(r, c) != 0)
				num++;
		}
	}
	//waitKey();
	return num;
}

/*
根据点集计算 获取区域延展率  width / height
*@brief return  延展率
@param area_cur   cur area points
*/
float gets(vector<Point> area_cur) {
	RotatedRect rect = minAreaRect(area_cur);
	float res = (float)rect.size.width / rect.size.height;
	return res;
}


/*
根据点集计算 获取前景区域灰度均值
*@brief return  灰度均值
@param area_cur   cur area points
@param src   src image
*/
float getave(vector<Point> area_cur, Mat src) {
	float total = 0;
	int num = area_cur.size();

	for (int i = 0; i < num; i++) {
		Point cur = area_cur[i];
		total = total + src.at<uchar>(cur.x, cur.y);
	}

	return total/num;
}

/*
根据点集计算 获取区域灰度偏移比
*@brief return  灰度偏移比
@param area_cur   cur area points
@param src   src image
*/
float getd(vector<Point> area_cur,Mat src, vector<float> cir) {

	Point cen((int)cir[1], (int)cir[0]);
	float radius = cir[2];

	//环形区域均值
	float total = 0;
	int sum = 1;
	for (int r = 0; r < src.rows; r++) {
		int leny = (r - cen.y)*(r - cen.y);
		for (int c = 0; c < src.cols; c++) {
			int lenx = (c - cen.x)*(c - cen.x);
			int len = lenx + leny;
			if (len > radius*0.56*radius*0.56 && len < radius*0.95*radius*0.95) {
				total = total + src.at<uchar>(r, c);
				sum++;
			}
		}
	}
	int all_ave = total / sum;

	//前景均值
	float ave = 0;
	ave = getave(area_cur, src);

	return (ave - all_ave)/ all_ave;
}




/*
根据根据输入原图和二值图 计算特征值
*@brief return  eigenvalue collection
@param bin   bin image
@param src   src image
*/
vector<vector<float>>  get_eigenvalue(Mat bin,Mat src ,vector<float> cir) {
	Point p;
	uchar label_find = 1;
	vector<vector<float>> res;
	Mat bin_image = bin.clone();
	int num = 200;
	while (num--) {
		//获取区域面积 01
		int area = 0;
		//获取区域周长 02
		int circumference = 0;
		//获取区域延展率 03
		float s = 0.0;
		//获取前景区域灰度均值 04
		float ave = 0.0;
		//获取区域灰度偏移比 05
		float d = 0.0;
		vector<Point> area_cur;
		p = find_one(bin_image);
		if (p.x != -1) {
			area = seed_fill(bin_image, p, label_find, area_cur);
			if (area < 5) {
				continue;
			}
		}
		else {
			break;
		}
		circumference = getCircumference(area_cur);
		s = gets(area_cur);
		d = getd(area_cur, src,cir);
		ave = getave(area_cur, src);
		vector<float> eigen;
		eigen.push_back(area);
		eigen.push_back(circumference);
		eigen.push_back(s);
		eigen.push_back(ave);
		eigen.push_back(d);
		res.push_back(eigen);
	}
	return res;
}
