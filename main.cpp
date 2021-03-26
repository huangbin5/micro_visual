#include <iostream>
#include <opencv2/opencv.hpp>
#include "data.cpp"

using namespace std;
using namespace cv;

const string BaseDir = "../img/";
const int Direct[][2] = {{-1, 0},
                         {0,  1},
                         {1,  0},
                         {0,  -1}};
string imgName;
int minX = 2000, maxX = 0, minY = 2000, maxY = 0;
bool printLog = true;

// 连通域区域
Rect Regions[10000];
// 连通域编号
Mat RegionNum;
int Number = 0;
// 切割得到的图案坐标(64, 64)
Rect Patterns[64][64];
vector<Mat> Template[7];
char Character[] = "SOCLT+-";
// 图案识别的结果(分别以数字和字符表示)
int Result[64][64];
char CharacterResult[64][64];

// 更新4个"顶点"的坐标
void updateContour(int y, int x) {
    minX = min(minX, x);
    maxX = max(maxX, x);
    minY = min(minY, y);
    maxY = max(maxY, y);
}

// 读取图像并转换为灰度图
Mat readGray(const string &name) {
    // 读取图像
    Mat img = imread(BaseDir + "" + name);
    // 将图像转化为灰度图
    cvtColor(img, img, COLOR_BGR2GRAY);
    imwrite(BaseDir + "s_gray.jpg", img);
    if (printLog)
        cout << "readGray done" << endl;
    return img;
}

// 开运算
Mat open(const Mat &img, int w) {
    Mat res(img);
    Mat ele = getStructuringElement(MORPH_RECT, Size(w, w));
    morphologyEx(img, img, MORPH_OPEN, ele);
    imwrite(BaseDir + "s_open.jpg", img);
    if (printLog)
        cout << "open done" << endl;
    return res;
}

// 闭运算
Mat close(const Mat &img, int w) {
    Mat res(img);
    Mat ele = getStructuringElement(MORPH_RECT, Size(w, w));
    morphologyEx(img, img, MORPH_CLOSE, ele);
    imwrite(BaseDir + "s_close.jpg", img);
    if (printLog)
        cout << "close done" << endl;
    return res;
}

// 搜索连通域并去除面积小于阈值的连通域
void domain(Mat &img, int minSize) {
    RegionNum = Mat::zeros(img.size(), CV_32SC1);
    Number = 0;
    // 将最外围灰度值置为0，以减少后面越界的判断
    for (int i = 0; i < img.rows; ++i)
        img.at<uchar>(i, 0) = img.at<uchar>(i, img.cols - 1) = 0;
    for (int j = 0; j < img.cols; ++j)
        img.at<uchar>(0, j) = img.at<uchar>(img.rows - 1, j) = 0;
    // flag：0 未访问；-1 在队列中；-2 已访问
    Mat flag = Mat::zeros(img.size(), CV_32SC1);
    for (int i = 1; i < img.rows - 1; ++i)
        for (int j = 1; j < img.cols - 1; ++j)
            if (flag.at<int>(i, j) == 0 && img.at<uchar>(i, j) == 255) {
                minX = 2000, maxX = 0, minY = 2000, maxY = 0;
                ++Number;
                queue<pair<int, int>> candidate;
                candidate.push(make_pair(i, j));
                int cnt = 0;
                while (!candidate.empty()) {
                    pair<int, int> cur = candidate.front();
                    candidate.pop();
                    int y = cur.first, x = cur.second;
                    flag.at<int>(y, x) = -2;
                    updateContour(y, x);
                    RegionNum.at<int>(y, x) = Number;
                    ++cnt;
                    for (auto k : Direct) {
                        int ny = y + k[0], nx = x + k[1];
                        if (flag.at<int>(ny, nx) == 0 && img.at<uchar>(ny, nx) == 255) {
                            candidate.push(make_pair(ny, nx));
                            flag.at<int>(ny, nx) = -1;
                        }
                    }
                }
                if (cnt < minSize) {
                    --Number;
                    candidate.push(make_pair(i, j));
                    while (!candidate.empty()) {
                        pair<int, int> cur = candidate.front();
                        candidate.pop();
                        int y = cur.first, x = cur.second;
                        if (img.at<uchar>(y, x) == 0) continue;
                        img.at<uchar>(y, x) = 0;
                        RegionNum.at<int>(y, x) = 0;
                        for (auto k : Direct) {
                            int ny = y + k[0], nx = x + k[1];
                            if (img.at<uchar>(ny, nx) == 255)
                                candidate.push(make_pair(ny, nx));
                        }
                    }
                } else
                    Regions[Number] = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
            }
    imwrite(BaseDir + "s_domain.jpg", img);
    if (printLog)
        cout << "domain done" << endl;
}

Mat readHSV(const string &name) {
    // 读取图像
    Mat img = imread(BaseDir + "" + name);
    // 将图像转化为HSV
    cvtColor(img, img, COLOR_BGR2HSV);
    imwrite(BaseDir + "s_hsv.jpg", img);
    if (printLog)
        cout << "readHSV done" << endl;
    return img;
}

void removeDark(Mat &grayImg, Mat &hsvImg) {
    int thresh = 40;
    for (int i = 0; i < hsvImg.rows; ++i)
        for (int j = 0; j < hsvImg.cols; ++j) {
            if (hsvImg.at<Vec3b>(i, j)[2] < thresh * 255 / 100)
                grayImg.at<uchar>(i, j) = 0;
        }
    imwrite(BaseDir + "s_dark.jpg", grayImg);
    if (printLog)
        cout << "removeDark done" << endl;
}

// 提取图像区域，为方便处理可以加几个像素的黑色边框
Mat extract(Mat &grayImg, Mat &hsvImg) {
    int left = 400, right = 1500;
    // 去掉两边大片的无关区域
    for (int i = 0; i < grayImg.rows; ++i) {
        for (int j = 0; j < left; ++j)
            grayImg.at<uchar>(i, j) = 0;
        for (int j = right; j < grayImg.cols; ++j)
            grayImg.at<uchar>(i, j) = 0;
        for (int j = left; j < right; ++j)
            if (grayImg.at<uchar>(i, j) != 0)
                grayImg.at<uchar>(i, j) = 255;
    }
    // 闭运算，让待选区域连通
    close(grayImg, 12);
    // 只保留一个连通域
    domain(grayImg, 10000);
    // 涂黑背景，保留7像素边框进行剪裁
    minX = 2000, maxX = 0, minY = 2000, maxY = 0;
    for (int i = 0; i < grayImg.rows; ++i) {
        for (int j = 0; j < grayImg.cols; ++j) {
            if (grayImg.at<uchar>(i, j) == 0)
                hsvImg.at<Vec3b>(i, j)[2] = 0;
            else {
                updateContour(i, j);
                break;
            }
        }
        for (int j = grayImg.cols - 1; j >= 0; --j) {
            if (grayImg.at<uchar>(i, j) == 0)
                hsvImg.at<Vec3b>(i, j)[2] = 0;
            else {
                updateContour(i, j);
                break;
            }
        }
    }
    for (int j = 0; j < grayImg.cols; ++j) {
        for (int i = 0; i < grayImg.rows; ++i) {
            if (grayImg.at<uchar>(i, j) == 0)
                hsvImg.at<Vec3b>(i, j)[2] = 0;
            else {
                updateContour(i, j);
                break;
            }
        }
        for (int i = grayImg.rows - 1; i >= 0; --i) {
            if (grayImg.at<uchar>(i, j) == 0)
                hsvImg.at<Vec3b>(i, j)[2] = 0;
            else {
                updateContour(i, j);
                break;
            }
        }
    }
    Rect area(minX - 7, minY - 7, maxX - minX + 15, maxY - minY + 15);
    Mat res = hsvImg(area);
    cvtColor(res, res, COLOR_HSV2BGR);
    cvtColor(res, res, COLOR_BGR2GRAY);
    imwrite(BaseDir + "s_extract.jpg", res);
    if (printLog)
        cout << "extract done" << endl;
    return res;
}

Mat threshBright(const Mat &img) {
    Mat res = img.clone();
    for (int y = 7; y < img.rows - 7; ++y)
        for (int x = 7; x <= img.cols - 7; ++x)
            if (img.at<uchar>(y, x) > 0) {
                int sum = 0, cnt = 0;
                for (int i = -7; i <= 7; ++i)
                    for (int j = -7; j <= 7; ++j) {
                        uchar v = img.at<uchar>(y + i, x + j);
                        if (v > 0) {
                            sum += v;
                            ++cnt;
                        }
                    }
                double avg = (double) sum / cnt;
                if (img.at<uchar>(y, x) < avg)
                    res.at<uchar>(y, x) = 0;
                else
                    res.at<uchar>(y, x) = 255;
            }
    imwrite(BaseDir + "s_bright.jpg", res);
    if (printLog)
        cout << "threshBright done" << endl;
    return res;
}

// 识别目标
int recognize(Mat target) {
    // 先将目标缩放至13x13
    int row = target.rows, col = target.cols;
    if (row != 13) {
        int up = abs(row - 13) / 2, down = up;
        if ((row - 13) % 2 == 1) {
            int upNum = 0, downNum = 0;
            for (int j = 0; j < col; ++j) {
                if (target.at<uchar>(0, j) == 255)
                    ++upNum;
                if (target.at<uchar>(row - 1, j) == 255)
                    ++downNum;
            }
            (row > 13) ^ (upNum > downNum) ? ++up : ++down;
        }
        if (row > 13)
            target = target(Rect(0, up, col, 13));
        else {
            Mat tmp = target.clone();
            target = Mat(13, col, CV_8UC1);
            for (int j = 0; j < col; ++j) {
                for (int i = 0; i < up; ++i)
                    target.at<uchar>(i, j) = tmp.at<uchar>(0, j);
                for (int i = up; i < up + row; ++i)
                    target.at<uchar>(i, j) = tmp.at<uchar>(i - up, j);
                for (int i = up + row; i < 13; ++i)
                    target.at<uchar>(i, j) = tmp.at<uchar>(row - 1, j);
            }
        }
    }
    if (col != 13) {
        int left = abs(col - 13) / 2, right = left;
        if ((col - 13) % 2 == 1) {
            int leftNum = 0, rightNum = 0;
            for (int i = 0; i < 13; ++i) {
                if (target.at<uchar>(i, 0) == 255)
                    ++leftNum;
                if (target.at<uchar>(i, col - 1) == 255)
                    ++rightNum;
            }
            (col > 13) ^ (leftNum > rightNum) ? ++left : ++right;
        }
        if (col > 13)
            target = target(Rect(left, 0, 13, 13));
        else {
            Mat tmp = target.clone();
            target = Mat(13, 13, CV_8UC1);
            for (int i = 0; i < 13; ++i) {
                for (int j = 0; j < left; ++j)
                    target.at<uchar>(i, j) = tmp.at<uchar>(i, 0);
                for (int j = left; j < left + col; ++j)
                    target.at<uchar>(i, j) = tmp.at<uchar>(i, j - left);
                for (int j = left + col; j < 13; ++j)
                    target.at<uchar>(i, j) = tmp.at<uchar>(i, col - 1);
            }
        }
    }
    // 再跟模板一个个比较
    int maxArea = 0, maxIndex = -1;
    for (int h = 0; h < 7; ++h)
        for (int k = 0; k < Template[h].size(); ++k) {
            int cnt = 0;
            for (int i = 0; i < 13; ++i)
                for (int j = 0; j < 13; ++j)
                    if (target.at<uchar>(i, j) == Template[h][k].at<uchar>(i, j))
                        ++cnt;
            if (cnt > maxArea) {
                maxArea = cnt;
                maxIndex = h;
            }
        }
    return maxIndex;
}

void printResult() {
    for (auto &row : CharacterResult) {
        for (char ele : row)
            cout << ele << " ";
        cout << endl;
    }
}

/**
 * 图案寻找算法
 * 错位：相邻图案错位最多6像素(图案大小一半)
 * 大小：标准大小12，间隔3，最大限制15
 */
void search(Mat &img) {
    int len = min(img.rows, img.cols);
    // 寻找左上角第一个图案，按副对角线方向扫描
    for (int k = 0; k < len; ++k) {
        int flag = false;
        for (int i = 0; i <= k; ++i) {
            int j = k - i;
            if (img.at<uchar>(i, j) == 255) {
                Patterns[0][0] = Regions[RegionNum.at<int>(i, j)];
                flag = true;
                break;
            }
        }
        if (flag) break;
    }
    imwrite(BaseDir + "split/0_0.jpg", img(Patterns[0][0]));
    Result[0][0] = recognize(img(Patterns[0][0]));
    CharacterResult[0][0] = Character[Result[0][0]];
    // 一个个图案遍历
    int row = 0, col = 0;
    while (row < 63 || col < 63) {
        int left, up, right, down;
        // 确定下一个图案的大致范围
        if (col < 63) {
            ++col;
            Rect area = Patterns[row][col - 1];
            left = area.x + area.width + 1, right = left + 1 + 12 + 2;
            up = area.y - 2, down = area.y + area.height + 2;
        } else {
            col = 0, ++row;
            Rect area = Patterns[row - 1][col];
            left = area.x - 2, right = area.x + area.width + 2;
            up = area.y + area.height + 1, down = up + 1 + 12 + 2;
        }
        // 找到目标连通域
        map<int, int> showTimes;
        for (int i = up; i <= down; ++i)
            for (int j = left; j <= right; ++j)
                if (img.at<uchar>(i, j) == 255) {
                    int number = RegionNum.at<int>(i, j);
                    if (showTimes.count(number) == 0)
                        showTimes[number] = 1;
                    else
                        ++showTimes[number];
                }
        int maxTimes = 0, maxTimesNum = 0;
        for (auto &it : showTimes)
            if (it.second > maxTimes) {
                maxTimesNum = it.first;
                maxTimes = it.second;
            }
        if (maxTimesNum == 0) {
            cout << "寻找连通域(" << row << ", " << col << ")出错" << endl;
            return;
        } else {
            bool hasCut = false;
            Rect region = Regions[maxTimesNum];
            // 设置切割阈值
            int thresh = row > 0 ? Patterns[row - 1][col].width / 2 : 6;
            if (region.x + region.width >= right + thresh) {
                hasCut = true;
                // 切断
                for (int i = up; i <= down; ++i)
                    if (RegionNum.at<int>(i, right - 1) == maxTimesNum) {
                        RegionNum.at<int>(i, right - 1) = 0;
                        img.at<uchar>(i, right - 1) = 0;
                    }
                // 建立新的连通域
                for (int i = up; i <= down; ++i)
                    if (RegionNum.at<int>(i, right) == maxTimesNum) {
                        minX = 2000, maxX = 0, minY = 2000, maxY = 0;
                        ++Number;
                        queue<pair<int, int>> candidate;
                        candidate.push(make_pair(i, right));
                        while (!candidate.empty()) {
                            pair<int, int> cur = candidate.front();
                            candidate.pop();
                            int y = cur.first, x = cur.second;
                            updateContour(y, x);
                            RegionNum.at<int>(y, x) = Number;
                            for (auto k : Direct) {
                                int ny = y + k[0], nx = x + k[1];
                                if (RegionNum.at<int>(ny, nx) == maxTimesNum)
                                    candidate.push(make_pair(ny, nx));
                            }
                        }
                        Regions[Number] = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
                        break;
                    }
            }
            thresh = col > 0 ? Patterns[row][col - 1].height / 2 : 6;
            if (region.y + region.height >= down + thresh) {
                hasCut = true;
                for (int j = left; j <= right; ++j)
                    if (RegionNum.at<int>(down - 1, j) == maxTimesNum) {
                        RegionNum.at<int>(down - 1, j) = 0;
                        img.at<uchar>(down - 1, j) = 0;
                    }
                for (int j = left; j <= right; ++j)
                    if (RegionNum.at<int>(down, j) == maxTimesNum) {
                        minX = 2000, maxX = 0, minY = 2000, maxY = 0;
                        ++Number;
                        queue<pair<int, int>> candidate;
                        candidate.push(make_pair(down, j));
                        while (!candidate.empty()) {
                            pair<int, int> cur = candidate.front();
                            candidate.pop();
                            int y = cur.first, x = cur.second;
                            updateContour(y, x);
                            RegionNum.at<int>(y, x) = Number;
                            for (auto k : Direct) {
                                int ny = y + k[0], nx = x + k[1];
                                if (RegionNum.at<int>(ny, nx) == maxTimesNum)
                                    candidate.push(make_pair(ny, nx));
                            }
                        }
                        Regions[Number] = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
                        break;
                    }
            }
            // 重新计算连通域坐标
            if (hasCut) {
                minX = 2000, maxX = 0, minY = 2000, maxY = 0;
                for (int i = up; i < down + 6; ++i)
                    for (int j = left; j < right + 6; ++j)
                        if (RegionNum.at<int>(i, j) == maxTimesNum)
                            updateContour(i, j);
                region = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
            }
            // 过小或过大则参考邻居大小进行切割
            if (region.width <= 8 || region.width > 16 || region.height <= 8 || region.height > 16) {
                // 好在4个角都不是'-'
                region.width = region.height = 12;
                if (row > 0) {
                    region.x = Patterns[row - 1][col].x;
                    if (row > 1)
                        region.x = 2 * Patterns[row - 1][col].x - Patterns[row - 2][col].x;
                    region.width = Patterns[row - 1][col].width;
                }
                if (col > 0) {
                    region.y = Patterns[row][col - 1].y;
                    if (col > 1)
                        region.y = 2 * Patterns[row][col - 1].y - Patterns[row][col - 2].y;
                    region.height = Patterns[row][col - 1].height;
                }
                if (row == 0) {
                    if (col > 0)
                        region.x = Patterns[row][col - 1].x + Patterns[row][col - 1].width + 3;
                    region.width = region.height;
                }
                if (col == 0) {
                    if (row > 0)
                        region.y = Patterns[row - 1][col].y + Patterns[row - 1][col].height + 3;
                    region.height = region.width;
                }
            }
            Patterns[row][col] = region;
            imwrite(BaseDir + "split/" + to_string(row) + "_" + to_string(col) + ".jpg",
                    img(Patterns[row][col]));
            Result[row][col] = recognize(img(Patterns[row][col]));
            CharacterResult[row][col] = Character[Result[row][col]];
        }
    }
}

// 制作13x13的标准图案
void makeTemplate() {
    Size size(13, 13);
    // S
    Mat pattern = Mat::zeros(size, CV_8UC1);
    for (int i = 0; i < pattern.rows; ++i)
        for (int j = 0; j < pattern.cols; ++j)
            pattern.at<uchar>(i, j) = 255;
    Template[0].push_back(pattern.clone());
    // O
    for (int i = 3; i < 10; ++i)
        for (int j = 3; j < 10; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[1].push_back(pattern.clone());
    // C:4
    for (int i = 3; i < 10; ++i)
        for (int j = 10; j < 13; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[2].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
        Template[2].push_back(pattern.clone());
    }
    // L:4
    for (int i = 0; i < 10; ++i)
        for (int j = 10; j < 13; ++j)
            pattern.at<uchar>(i, j) = 0;
    Template[3].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_COUNTERCLOCKWISE);
        Template[3].push_back(pattern.clone());
    }
    // T:4
    for (int i = 3; i < 13; ++i) {
        for (int j = 0; j < 3; ++j)
            pattern.at<uchar>(i, j) = 0;
        for (int j = 5; j < 8; ++j)
            pattern.at<uchar>(i, j) = 255;
    }
    Template[4].push_back(pattern.clone());
    for (int i = 0; i < 3; ++i) {
        rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
        Template[4].push_back(pattern.clone());
    }
    // +
    for (int i = 0; i < 13; ++i) {
        if (i >= 5 && i < 8) continue;
        for (int j = 0; j < 3; ++j)
            pattern.at<uchar>(i, j) = 0;
        for (int j = 5; j < 8; ++j)
            pattern.at<uchar>(i, j) = 255;
    }
    Template[5].push_back(pattern.clone());
    // -:2
    for (int i = 0; i < 13; ++i) {
        if (i >= 5 && i < 8) continue;
        for (int j = 5; j < 8; ++j)
            pattern.at<uchar>(i, j) = 0;
    }
    Template[6].push_back(pattern.clone());
    rotate(pattern, pattern, ROTATE_90_CLOCKWISE);
    Template[6].push_back(pattern.clone());
    // 保存
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < Template[i].size(); ++j)
            imwrite(BaseDir + "template/" + to_string(i) + to_string(j) + ".jpg", Template[i][j]);
}

// 计算准确率；显示错误识别的图案
void check(Mat &img) {
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
            if (img.at<uchar>(i, j) == 255)
                img.at<uchar>(i, j) = 127;
    int cnt = 0;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
            if (Result[i][j] == sevenPatterns[i][j])
                ++cnt;
            else {
                Rect region = Patterns[i][j];
                for (int y = 0; y < region.height; ++y)
                    for (int x = 0; x < region.width; ++x)
                        if (img.at<uchar>(region.y + y, region.x + x) == 127)
                            img.at<uchar>(region.y + y, region.x + x) = 255;
            }
    imwrite(BaseDir + "check/" + imgName, img);
    if (printLog)
        cout << "check done" << endl;
    cout << imgName << "准确率为" << (double) cnt / 64 / 64 * 100 << "%" << endl;
}

void hsvSolution() {
    Mat grayImg = readGray(imgName);
    Mat hsvImg = readHSV(imgName);
    // 先用40亮度的阈值去除外围背景
    removeDark(grayImg, hsvImg);
    // 选定图像区域：左400右1500，去除背景噪点
//    mark(darkImg);
    Mat cutImg = extract(grayImg, hsvImg);
    // 对亮度局部阈值化
    Mat threshImg = threshBright(cutImg);
    // 处理局部噪点
    open(threshImg, 2);
    close(threshImg, 2);
    domain(threshImg, 10);
    search(threshImg);
    printResult();
    check(threshImg);
}

int main() {
    printLog = true;
    makeTemplate();
//    imgName = "6.jpg";
//    hsvSolution();
    for (int i = 1; i <= 6; ++i) {
        imgName = to_string(i) + ".jpg";
        hsvSolution();
    }
    return 0;
}