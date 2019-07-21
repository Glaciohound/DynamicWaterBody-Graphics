/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: Mon Dec 10 08:18:00 2018
 ************************************************************************/

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include "image.h"
#include "BVH.h"
#include <stdio.h>   
#include <stdlib.h>   

/* ------------- Utility --------------*/
using namespace std;
static const float pi = 3.14159265358979;
static float Max=1e10, zero_threshold=1e-6, triangle_threshold = 1.1, medium_threshold=1e-4;
#define PI ((float)3.14159265358979)
#define ALPHA ((float)0.85)
int pvalue[61]={
	2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79, 83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181, 191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283
};

inline int reverse(const int i,const int p) {
	if (i==0)
        return i;
    else
        return p-i;
}
float hal(const int b, int j) {
	const int p = pvalue[b]; 
	float h = 0.0, f = 1.0 / (float)p, fct = f;
	while (j > 0) {
		h += reverse(j % p, p) * fct; j /= p; fct *= f;
	}
	return h;
}
List* ListAdd(HPoint *i,List* h){
	List* p=new List;
	p->id=i;
	p->next=h;
	return p;
}

inline unsigned int Scene::f_hash(const int ix, const int iy, const int iz) {
	return (unsigned int)((ix*73856093)^(iy*19349663)^(iz*83492791))%num_hash;
}
inline float toInt(float x){ return (unsigned char)(pow(1-exp(-x),1/2.2)*255+.5); }
Color toInt(Vector3d v){
    return Color(toInt(v.x), toInt(v.y), toInt(v.z));
}
inline int resize(int x){
    return x;
}

/* ------------ Color Zone -------------*/

int big = 200, small = 10;
vector<Color> colorList = {
    Color(big, big, big),
    Color(big, small, small),
    Color(small, small, big),
    Color(120, 120, 120),
    Color(150, 150, 200),
    Color(255, 255, 255),
    Color(28, 28, 28),
};
vector<string> colorListIndex={
    "white", "red", "blue", "gray", "water_blue", "bright", "dark"
};
Color get_color(string color_name){
    int color_index=-1;
    for (int i=0; i!=colorList.size(); i++){
        if (colorListIndex[i]==color_name)
            color_index = i;
    }
    if (color_index !=-1)
        return colorList[color_index];
}
bool has_color(string color_name){
    int color_index=-1;
    for (int i=0; i!=colorList.size(); i++)
        if (colorListIndex[i]==color_name)
            color_index = i;
    if (color_index !=-1)
        return true;
    else 
        return false;
}
Light::Light(Vector3d f, Vector3d d, float r, string name):
    from(f), direction(d.normalized()), range(r), color(name){
        fin = direction.get_FinVector();
        handle = direction.cross(fin);
    }

/* ------------ Image Zone -------------*/

Image::Image(int w, int h, string command): 
    feature(command), color(Color(0, 0, 0)),
    width(w), height(h),
    image(width, vector<Color>(height)),
    radius(min(w, h)/2),
    measurement(w, vector<Vector3d>(h)),
    values(width, vector<float>(height)){ }

void Image::output(string outfile_name){
    FILE *f;
    int filesize = 54 + 3*width*height;
    int w=width, h=height;
    int l = outfile_name.length();
    if (outfile_name.substr(l-4, 4)!=".bmp")
        outfile_name += ".bmp";

    unsigned char bmpfileheader[14] =
        {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] =
        {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] =
        {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    f = fopen(outfile_name.c_str(),"w");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    unsigned char output_string[3*width*height];
    for (int i=0; i!=width; i++)
        for (int j=0; j!=height; j++){
            output_string[3*(j*w+i)+2] = (unsigned char)image[i][j].x;
            output_string[3*(j*w+i)+1] = (unsigned char)image[i][j].y;
            output_string[3*(j*w+i)+0] = (unsigned char)image[i][j].z;
        }
    for(int i=0; i<h; i++)
    {
        fwrite(output_string+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }
    fclose(f);
}
bool Image::from_file(string infile_name)
{
    feature = "image";
    if (infile_name.substr(infile_name.length()-4, 4)!=".bmp")
        infile_name += ".bmp";
    FILE *f=fopen(infile_name.c_str(),"rb");

    unsigned char head[40];
    fseek(f, 14, 0);
    fread(head, 40, 1, f);
    width = head[4] + (head[5]<<8) + (head[6]<<16) + (head[7]<<24);
    height = head[8] + (head[9]<<8) + (head[10]<<16) + (head[11]<<24);
    radius = min(width, height)/2;
    image = vector<vector<Color>>(width, vector<Color>(height));

    vector<unsigned char>img(width * height * 3);
    int m = max(height, width);
    values = vector<vector<float>>(m, vector<float>(m, 0));
    DP_values = vector<vector<Pair>>(m, vector<Pair>(m));
    seam = vector<int>(max(width, height));
    unsigned char* input_string = &img[0];
    unsigned char bmppad[3];
    for(int i=0; i<height; i++)
    {
        fread(input_string + (width*(height-i-1)*3),3, width, f);
        fread(bmppad,1,(4-(width*3)%4)%4,f);
    }
    fclose(f);
    for (int i=0; i!=width; i++)
        for (int j=0; j!=height; j++)
            image[i][j] = Color(
                    img[3*(j*width+i)+2],
                    img[3*(j*width+i)+1],
                    img[3*(j*width+i)+0]);
    return true;
}
void Image::reset(int w, int h){
    *this = Image(w, h);
}
void Image::draw_point(float x, float y, Color color){
    int radius = min(width, height)/2;
    int i=x*height+width/2, j=y*height+height/2, w=width, h=height;
    draw_point(i, j, color);
}
void Image::draw_point(int i, int j, Color color){
    int w=width, h=height;
    if (i>=width || i<0 || j>=height || j<0)
        return;
    image[i][j] = color;
}
Color Image::pick_color(int i, int j){
    if (feature == "mono" || feature == "mirror" || feature == "glass"){
        return color;
    }
    int w=width, h=height;
    if (i>=width || i<0 || j>=height || j<0)
        return Color(0, 0, 0);
    return image[i][j];
}
Color Image::pick_color(float x, float y){
    return pick_color((int)(x*width), (int)(y*height));
}
void Image::draw_from_measurement(){
    for(int i = 0; i< width; i++)
        for (int j=0; j < height; j++)
            draw_point(i, j, toInt(measurement[i][j]));
}


/* --------------------- Seam Carving Zone ----------------------------*/
void Image::calculate_value(string type){
    if (type == "horizontal"){
        for (int i=0; i!=width; i++)
            for (int j=0; j!=height; j++){
                values[i][j] = (pick_color(i, j) - pick_color(i+1, j)).norm("1");
            }
    }
    if (type == "vertical"){
        for (int i=0; i!=width; i++)
            for (int j=0; j!=height; j++){
                Color t = pick_color(i, j), n = pick_color(i, j+1);
                values[i][j] = pow((t-n).norm("2"), 2);
            }
    }
    if (type == "mix"){
        for (int i=0; i!=width; i++)
            for (int j=0; j!=height; j++){
                values[i][j] =
                    (pick_color(i, j) - pick_color(i, j+1)).norm("1") +
                    (pick_color(i, j) - pick_color(i+1, j)).norm("2");
            }
    }
}
void Image::calculate_seam(string direction){
    if (direction == "horizontal"){
        DP_values = vector<vector<Pair>>(width, vector<Pair>(height));
        seam = vector<int>(max(width, height));
        for (int i=0; i!=width; i++){
            for (int j=0; j!=height; j++){
                if (i==0){
                    DP_values[i][j] = Pair(-1, values[i][j]);
                    continue;
                }
                DP_values[i][j] = Pair(-1, 1e9);
                for (int k=max(j-1, 1); k<=min(j+1, height-2); k++){
                    Pair cand(k, values[i][j] + DP_values[i-1][k].v);
                    if (cand.v < DP_values[i][j].v)
                        DP_values[i][j] = cand;
                }
            }
        }
        Pair cand(-1, 1e9);
        for (int j=1; j!=height-1; j++){
            if (DP_values[width-1][j].v < cand.v)
                cand = Pair(j, DP_values[width-1][j].v);
        }
        seam[width-1] = cand.i;
        for (int i=width-2; i!=-1; i--)
            seam[i] = DP_values[i+1][seam[i+1]].i;
        seamsx.push_back(seam);
        for (int i=0; i!=width; i++)
            values[i][seam[i]] = 1e7;
    }
    else if (direction == "vertical"){
        DP_values = vector<vector<Pair>>(width, vector<Pair>(height));
        seam = vector<int>(max(width, height));
        for (int j=0; j!=height; j++){
            for (int i=0; i!=width; i++){
                if (j==0){
                    DP_values[i][j] = Pair(-1, values[i][j]);
                    continue;
                }
                DP_values[i][j] = Pair(-1, 1e9);
                for (int k=max(i-1, 1); k<=min(i+1, width-2); k++){
                    Pair cand(k, values[i][j] + DP_values[k][j-1].v);
                    if (cand.v < DP_values[i][j].v)
                        DP_values[i][j] = cand;
                }
            }
        }
        Pair cand(-1, 1e9);
        for (int i=1; i!=width-1; i++){
            if (DP_values[i][height-1].v < cand.v)
                cand = Pair(i, DP_values[i][height-1].v);
        }
        seam[height-1] = cand.i;
        for (int j=height-2; j!=-1; j--)
            seam[j] = DP_values[seam[j+1]][j+1].i;
        seamsy.push_back(seam);
        for (int j=0; j!=height; j++)
            values[seam[j]][j] += 1e7;
    }
}
bool Image::contains(string type, int i, int j){
    if (type == "x"){
        for (int k=0; k!=seamsx.size(); k++)
            if (seamsx[k][i] == j)
                return true;
    }
    else{
        for (int k=0; k!=seamsy.size(); k++)
            if (seamsy[k][i] == j)
                return true;
    }
    return false;
}
Image Image::cut_seam(string type){
    if (type == "horizontal"){
        Image output(width, height - seamsx.size(), "image");
        for (int i=0; i!=width; i++){
            int counter = 0;
            for (int j=0; j!=height; j++){
                if (contains("x", i, j))
                    counter ++;
                else
                    output.draw_point(i, j-counter, pick_color(i, j));
            }
        }
        return output;
    }
    else if (type == "vertical"){
        Image output(width - seamsy.size(), height, "image");
        for (int j=0; j!=height; j++){
            int counter = 0;
            for (int i=0; i!=width; i++){
                if (contains("y", j, i))
                    counter ++;
                else
                    output.draw_point(i-counter, j, pick_color(i, j));
            }
        }
        return output;
    } 
}
Image Image::expand_seam(string type){
    if (type == "horizontal"){
        Image output(width, height + seamsx.size(), "image");
        for (int i=0; i!=width; i++){
            int counter = 0;
            for (int j=0; j!=height; j++){
                if (contains("x", i, j)){
                    output.draw_point(i, j+counter, pick_color(i, j));
                    counter ++;
                    output.draw_point(i, j+counter, pick_color(i, j));
                }
                else
                    output.draw_point(i, j+counter, pick_color(i, j));
            }
        }
        return output;
    }
    else if (type == "vertical"){
        Image output(width + seamsy.size(), height, "image");
        for (int j=0; j!=height; j++){
            int counter = 0;
            for (int i=0; i!=width; i++){
                if (contains("y", j, i)){
                    output.draw_point(i+counter, j, pick_color(i, j));
                    counter ++;
                    output.draw_point(i+counter, j, pick_color(i, j));
                }
                else
                    output.draw_point(i+counter, j, pick_color(i, j));
            }
        }
        return output;
    } 
}
Image Image::display_seam(string type){
    Image output = *this;
    if (type == "horizontal"){
        for (int n=0; n!=seamsx.size(); n++)
            for (int i=0; i!=width; i++){
                output.draw_point(i, seamsx[n][i], Color(255, 0, 0));
            }
        return output;
    }
    else if (type == "vertical"){
        for (int n=0; n!=seamsy.size(); n++)
            for (int j=0; j!=width; j++){
                output.draw_point(seamsy[n][j], j, Color(255, 0, 0));
            }
        return output;
    } else if (type == "mix"){
        for (int n=0; n!=seamsx.size(); n++)
            for (int i=0; i!=width; i++){
                output.draw_point(i, seamsx[n][i], Color(255, 0, 0));
            }
        for (int n=0; n!=seamsy.size(); n++)
            for (int j=0; j!=width; j++){
                output.draw_point(seamsy[n][j], j, Color(255, 0, 0));
            }
        return output;
    }
}
Image Image::display_energy(){
    Image output = *this;
    for (int i=0; i!=width; i++)
        for (int j=0; j!=height; j++)
            output.draw_point(i, j, Color((int)values[i][j], (int)values[i][j], (int)values[i][j]));
    return output;
}
Image Image::transpose(){
    Image output(height, width, "image");
    for (int i=0; i!=width; i++)
        for (int j=0; j!=height; j++)
            output.draw_point(j, i, pick_color(i, j));
    int m = max(width, height);
    output.values = vector<vector<float>>(m, vector<float>(m, 0));
    output.DP_values = vector<vector<Pair>>(m, vector<Pair>(m));
    output.seam = vector<int>(m);
    return output;
}
void Image::protect(int x1, int x2, int y1, int y2){
    for (int i=x1; i!=x2; i++)
        for (int j=y1; j!=y2; j++){
            values[i][j] = 1e10;
            Color c = pick_color(i, j);
            draw_point(i, j, Color(c.x/2, c.z/2, c.y/2));
        }
}
void Image::remove(int x1, int x2, int y1, int y2){
    for (int i=x1; i!=x2; i++)
        for (int j=y1; j!=y2; j++){
            values[i][j] = -1e10;
            Color c = pick_color(i, j);
            draw_point(i, j, Color(c.x/2, c.z/2, c.y/2));
        }
}
Image Image::denoise(int size, float alpha){
    Image output = *this;
    cout<<"Denoise"<<endl;
    ProgressBar pbar(height*width);
    int num = pow(size*2+1, 2) - 1, large = 10000;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i=0; i<=width-1; i++)
        for (int j=0; j<=height-1; j++){
            pbar.update();
            float sum = 1 - num * alpha;
            Color result = pick_color(i, j) * (sum * large);
            for (int x = max(i-size, 0); x<=min(i+size, width-1); x++)
                for (int y = max(j-size, 0); y<=min(j+size, height-1); y++)
                    if (x!=i || y!=j){
                        result = result + pick_color(x, y) * (alpha * large);
                        sum += alpha;
                    }
            result = result * (1/sum/large);
            output.draw_point(i, j, result);
        }
    return output;
}


/* ------------ NurbsCurve Zone -------------*/

int NurbsCurve::k_default = 5;
int NurbsCurve::n_default = 100;
NurbsCurve::NurbsCurve(int n, int k):n(n), k(min(k, n-1)), ControlPoints(n) {
}
NurbsCurve::NurbsCurve(vector<Vector3d> pointlist, int k):
    n(pointlist.size()), k(k), ControlPoints(pointlist){
    this->k = min(this->n-1, this->k);
}
NurbsCurve::NurbsCurve(string command, int n, int k):n(n), k(min(n-1, k)), ControlPoints(n){
    if (command=="x_axis")
        *this = NurbsCurve(vector<Vector3d>{Vector3d(0, 0, 0), Vector3d(1, 0, 0)}, 1);
    else if (command=="y_axis")
        *this = NurbsCurve(vector<Vector3d>{Vector3d(0, 0, 0), Vector3d(0, 1, 0)}, 1);
    else if (command=="z_axis")
        *this = NurbsCurve(vector<Vector3d>{Vector3d(0, 0, 0), Vector3d(0, 0, 1)}, 1);
    else if (command=="circle"){
        for (int i=0; i!=n; i++)
            ControlPoints[i] = Vector3d(cos(2*pi*i/(n-1)), sin(2*pi*i/(n-1)), 0);
    }
}
NurbsCurve::NurbsCurve(float (*f)(float),bool polar, int n, int k): NurbsCurve(n, k){
    for (int i=0; i!=n; i++)
        if (polar){
            float theta = 2*pi*i/(n-1), r = f(theta);
            ControlPoints[i] = Vector3d(r*cos(theta), r*sin(theta), 0);
        }
        else
            ControlPoints[i] = Vector3d(f(i*1./(n-1)), 0, i*1./(n-1));
}
NurbsCurve operator*(float c, NurbsCurve curve){
    NurbsCurve output = curve;
    for (int i=0; i!=curve.n; i++)
        output.ControlPoints[i] = c * output.ControlPoints[i];
    return output;
}
NurbsCurve NurbsCurve::operator*(float c){
    return c*(*this);
}
float NurbsCurve::get_max(){
    float output = -100;
    for (int i=0; i!=n; i++)
        output = max(ControlPoints[i][2], output);
    return output;
}
float NurbsCurve::get_min(){
    float output = 100;
    for (int i=0; i!=n; i++)
        output = min(ControlPoints[i][2], output);
    return output;
}
static vector<bezier_coefficient_item> bezier_coeffs;
bezier_coefficient_item::bezier_coefficient_item(int n, int k):
    n(n), k(k), coeffs(k+1, vector<vector<float>>(n+k+2, vector<float>(4))){
    if (n==0)
        return;
    vector<float> t_(n+k+2, 0.0);
    for (int i=0; i!=n+k+2; i++)
        t_[i] = min(max(i-k, 0), n-k);
    for (int p=1; p!=k+1; p++){
        for (int i=0; i!=n+k+1-p; i++){
           if (t_[p+i] - t_[i] != 0){
                coeffs[p][i][0] = 1 / (t_[p+i] - t_[i]);
                coeffs[p][i][1] = t_[i] * coeffs[p][i][0];
            }
            else{
                coeffs[p][i][0] = 0;
                coeffs[p][i][1] = 0;
            }
            if (t_[p+i+1] - t_[i+1] != 0){
                coeffs[p][i][2] = 1 / (t_[p+i+1] - t_[i+1]);
                coeffs[p][i][3] = t_[i+p+1] * coeffs[p][i][2];
            }
            else{
                coeffs[p][i][2] = 0;
                coeffs[p][i][3] = 0;
            }
        }
    }
}
Matrix Bezier(int n, int k, float t){
    if (t<0 || t>1){
        cout<<t<<' '<<"t out of range"<<endl;
    }
    int index = -1;
    if (bezier_coeffs.size() == 0)
        bezier_coeffs.reserve(20);
    for (int i=0; i!=bezier_coeffs.size(); i++)
        if (bezier_coeffs[i].n == n && bezier_coeffs[i].k == k){
            index = i;
            break;
        }
    if (index == -1){
        bezier_coeffs.push_back(bezier_coefficient_item(n, k));
        index = bezier_coeffs.size()-1;
    }
    bezier_coefficient_item& coeffs = bezier_coeffs[index];
    float values[2*(k+1)] = {0};
    if (t>=1)
        t = 1-zero_threshold;
    t = t * (n-k);
    int t0 = int(t) - k, t1 = int(t);
    if (k==2){
        float  A = coeffs.coeffs[1][t1+1][3] - t * coeffs.coeffs[1][t1+1][2],
                B = coeffs.coeffs[1][t1+2][0] * t - coeffs.coeffs[1][t1+2][1],
                C = k*(n-k);
        values[0] = (coeffs.coeffs[2][t1][3] - t * coeffs.coeffs[2][t1][2]) * A,
        values[1] = (coeffs.coeffs[2][t1+1][3] - t * coeffs.coeffs[2][t1+1][2]) * B +
                    (coeffs.coeffs[2][t1+1][0] * t - coeffs.coeffs[2][t1+1][1]) * A,
        values[2] = (coeffs.coeffs[2][t1+2][0] * t - coeffs.coeffs[2][t1+2][1]) * B;
        values[3] = -C * A * coeffs.coeffs[2][t1][2];
        values[4] = C * (A * coeffs.coeffs[2][t1+1][0] - B * coeffs.coeffs[2][t1+1][2]);
        values[5] = C * B * coeffs.coeffs[2][t1+2][0];
        return Matrix(values, 2, 3);
    }
    vector<float> weights(k+2, 0.0),
        weights_p(k+2, 0.0),
        t_(2*k+2, 0.0);
    weights_p[k] = 1;
    for (int i=0; i!=2*k+2; i++)
        t_[i] = min(max(t0+i, 0), n-k);
    for (int i=1; i!=k+1; i++){
        weights.assign(k+2, 0.0);
        for (int j=0; j!=k+1; j++){
            weights[j] = weights_p[j]*(t*coeffs.coeffs[i][t0+j+k][0] - coeffs.coeffs[i][t0+j+k][1]) +
                weights_p[j+1]*(coeffs.coeffs[i][t0+j+k][3] - t*coeffs.coeffs[i][t0+j+k][2]);
        }
        if (i!=k)
            swap(weights, weights_p);
    }
    for (int i=0; i!=k+1; i++){
        values[i] = weights[i];
        values[i + (k+1)] = k*(n-k)* (weights_p[i]*coeffs.coeffs[k][i+t0+k][0] -
                weights_p[i+1]*coeffs.coeffs[k][i+t0+k][2]);
    }
    return Matrix(values, 2, k+1);
}
Matrix NurbsCurve::getPoint(float t){
    Matrix parameters = Bezier(n, k, t);
    Vector3d output, derivative;
    if (t == 1)
        t -= zero_threshold;
    else if (t == 0)
        t +=  zero_threshold;
    int t0 = t*(n-k);
    for (int i=0; i!=k+1; i++) {
        output += parameters[0][i] * ControlPoints[i+t0];
        derivative += parameters[1][i] * ControlPoints[i+t0];
    }
    return Matrix({output, derivative});
}


float NurbsCurve::get_t_from_height(float h){
    float ret = 0;
    for (int i=0; i!=10; i++){
        float hp = getPoint(ret)[0][2];
        if (h >= hp)
            ret += pow(0.5, i+1);
        else
            ret -= pow(0.5, i+1);
    }
    return ret;
}

/* ------------ NurbsSurface Zone -------------*/

int NurbsSurface::k_default = 4;
int NurbsSurface::m_default = 30;
int NurbsSurface::n_default = 30;
NurbsSurface::NurbsSurface(int m, int n, string command, int k): m(m), n(n), k(min(min(m-1, n-1), k)), image(), ControlPoints(m, vector<Vector3d>(n)){
    if ( has_color(command) ){
        image.feature = "mono";
        image.color = get_color(command);
    }
    else if (command == "mirror"){
        image.color = get_color("bright");
        image.feature = command;
    }
    else if (command == "glass"){
        image.color = get_color("bright");
        image.feature = command;
    }
    else{
        image.feature = "image";
        image.from_file(command);
    }
}
void NurbsSurface::cylinder_surface(NurbsCurve spine, NurbsCurve base, NurbsCurve width, Vector3d fin){
    float mx = width.get_max(), mn = width.get_min(), height = mx-mn;
    for (int i=0; i!=n; i++){
        Vector3d support = width.getPoint(1.*i/(n-1))[0];
        float t = (support[2] - mn) / height,
               w = support[0];
        if (t>1) t = 1;
        if (t<0) t = 0;
        Matrix getpoint = spine.getPoint(t);
        Vector3d point = getpoint[0],
                 direction = getpoint[1];
        direction = direction.normalized();
        fin = direction.get_FinVector(fin);
        Vector3d y = direction.cross(fin);
        for (int j=0; j!=m; j++){
            ControlPoints[j][i] = point +
                ((Vector3d)base.getPoint(1.*j/(m-1))[0]).project(fin, y, direction)*w;
        }
    }
}
void NurbsSurface::rocket_body(NurbsCurve body_shape, NurbsCurve wing_shape, float (*f_wing)(float, float, float)){
    float height = body_shape.get_max() - body_shape.get_min();
    for (int i=0; i!=n; i++){
        Vector3d wingpoint = wing_shape.getPoint(1.*i/(n-1))[0];
        float wing_span = wingpoint[0],
               h = wingpoint[2]*height,
               t = body_shape.get_t_from_height(h);
        Vector3d body_point = body_shape.getPoint(t)[0];
        float body_width = body_point[0];
        if (t>1) t = 1;
        if (t<0) t = 0;
        for (int j=0; j!=m; j++){
            float theta = 2*pi*j/(m-1), r = f_wing(theta, wing_span, body_width);
            ControlPoints[j][i] = Vector3d(r*cos(theta), r*sin(theta), h);
        }
    }
}
void NurbsSurface::binary_expand(NurbsCurve curve_x, NurbsCurve curve_y){
    Vector3d from = curve_y.getPoint(0)[0],
             basePoint;
    for (int i=0; i!=m; i++){
        basePoint = curve_x.getPoint(1.*i/(m-1))[0];
        for (int j=0; j!=n; j++){
            ControlPoints[i][j] = basePoint + curve_y.getPoint(1.*j/(n-1))[0] - from;
        }
    }
}
float wave_function(float x, float y, float wave_length, float height, float move){
    return  sin((0.7*x+0.7*y + move/10)/wave_length)*height +
            sin((-0.95*x+0.2*y + move/10)/wave_length)*height; +
            sin((0.2*x-0.95*y + move/10)/wave_length)*height;
}
void NurbsSurface::water_expand(NurbsCurve curve_x, NurbsCurve curve_y, float wave_length, float height, float move){
    Vector3d from = curve_y.getPoint(0)[0],
             basePoint;
    for (int i=0; i!=m; i++){
        basePoint = curve_x.getPoint(1.*i/(m-1))[0];
        for (int j=0; j!=n; j++){
            Vector3d direct = basePoint + curve_y.getPoint(1.*j/(n-1))[0] - from;
            ControlPoints[i][j] = direct + Vector3d(0, 0, wave_function(direct.x, direct.y, wave_length, height, move));
        }
    }
}
void Scene::sketch(NurbsSurface surface, Color color){
    stringstream s;
    int m = surface.m, n = surface.n;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++){
            Vector3d point = surface.ControlPoints[i][j];
            sketch(point, color);
            s.str("");
            s << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << endl;
            obj_file[0].push_back(s.str());
            if (i != m-1 && j != n-1){
                s.str("");
                s << "f " << v_count+i*n+j+1 << ' ' << v_count+i*n+j+n+1 << ' ' << v_count+i*n+j+n+2 << ' ' << v_count+i*n+j+2 << endl;
                obj_file[4].push_back(s.str());
            }
        }
    v_count += m*n;
}
void NurbsSurface::build_patches(){
    ProgressBar pbar((m-1)*(n-1));
    for (int j=0; j<=m-2; j++)
        for (int k=0; k!=n-1; k++){
            pbar.update();
            patchList.push_back(Patch(this, 1.*j/(m-1), 1.*(j+1)/(m-1), 1.*k/(n-1), 1.*(k+1)/(n-1), true));
            patchList.push_back(Patch(this, 1.*j/(m-1), 1.*(j+1)/(m-1), 1.*k/(n-1), 1.*(k+1)/(n-1), false));
        }
}
Matrix NurbsSurface::getPoint(float u, float v){
    Matrix parm_u = Bezier(m, k, u),
           parm_v = Bezier(n, k, v);
    if (u == 1)
        u -= zero_threshold;
    if (v == 1)
        v -= zero_threshold;
    int u0 = u*(m-k), v0=v*(n-k);
    if (k==2){
        Vector3d    &v00 = ControlPoints[u0][v0], &v01 = ControlPoints[u0][v0+1], &v02 = ControlPoints[u0][v0+2],
                    &v10 = ControlPoints[u0+1][v0], &v11 = ControlPoints[u0+1][v0+1], &v12 = ControlPoints[u0+1][v0+2],
                    &v20 = ControlPoints[u0+2][v0], &v21 = ControlPoints[u0+2][v0+1], &v22 = ControlPoints[u0+2][v0+2];
        vector<float> &wu = parm_u[0], &wv = parm_v[0], &pu = parm_u[1], &pv = parm_v[1];
        Vector3d result =       wu[0]*wv[0]*v00 + wu[0]*wv[1]*v01 + wu[0]*wv[2]*v02 +
                                wu[1]*wv[0]*v10 + wu[1]*wv[1]*v11 + wu[1]*wv[2]*v12 +
                                wu[2]*wv[0]*v20 + wu[2]*wv[1]*v21 + wu[2]*wv[2]*v22,
                 derivative_u = pu[0]*wv[0]*v00 + pu[0]*wv[1]*v01 + pu[0]*wv[2]*v02 +
                                pu[1]*wv[0]*v10 + pu[1]*wv[1]*v11 + pu[1]*wv[2]*v12 +
                                pu[2]*wv[0]*v20 + pu[2]*wv[1]*v21 + pu[2]*wv[2]*v22,
                 derivative_v = wu[0]*pv[0]*v00 + wu[0]*pv[1]*v01 + wu[0]*pv[2]*v02 +
                                wu[1]*pv[0]*v10 + wu[1]*pv[1]*v11 + wu[1]*pv[2]*v12 +
                                wu[2]*pv[0]*v20 + wu[2]*pv[1]*v21 + wu[2]*pv[2]*v22;
        return Matrix({result, derivative_u, derivative_v});
    }
    Vector3d result, derivative_u, derivative_v;
    for (int i=0; i!=k+1; i++)
        for (int j=0; j!=k+1; j++){
            result += parm_u[0][i] * parm_v[0][j] * ControlPoints[i+u0][j+v0];
            derivative_u += parm_u[1][i] * parm_v[0][j] * ControlPoints[i+u0][j+v0];
            derivative_v += parm_u[0][i] * parm_v[1][j] * ControlPoints[i+u0][j+v0];
        }
    return Matrix({result, derivative_u, derivative_v});
}
Matrix NurbsSurface::intersect(Ray view, Vector u_range, Vector v_range){

    float u = (u_range[0] + u_range[1])/2,
          v = (v_range[0] + v_range[1])/2,
          margin = (u_range[1] - u_range[0])*1;
    Vector3d direction = view.direction.normalized();
    int num_iter = 10;
    Vector3d n1 = direction.get_FinVector(),
             n2 = direction.cross(n1);
    float d1 = 0 - view.from.dot(n1),
           d2 = 0 - view.from.dot(n2);
    bool found = false;
    Matrix getpoint;
    Vector3d S, S_u, S_v;
    for (int i=0; i!=num_iter && not found; i++){
        getpoint = getPoint(u, v);
        S = getpoint[0];
        S_u = getpoint[1];
        S_v = getpoint[2];

        Vector R(2);
        Matrix J(2, 2);
        R[0] = n1.dot(S) + d1;
        R[1] = n2.dot(S) + d2;
        J[0][0] = n1.dot(S_u);
        J[0][1] = n1.dot(S_v);
        J[1][0] = n2.dot(S_u);
        J[1][1] = n2.dot(S_v);

        Matrix inverse;
        try{
            inverse = J.inverse2();
        }
        catch(exception e){
            return Matrix(3, 3, -1);
        }
        Vector delta = inverse * R;
        u -= delta[0];
        v -= delta[1];
        found = R.norm() < medium_threshold ;
        if (u<u_range[1]+margin && u>u_range[0]-margin && v>v_range[0]-margin && v<v_range[1]+margin){
            u = min(max(u, (float)0.), (float)1.);
            v = min(max(v, (float)0.), (float)1.);
        } 
        else break;
    }
    if (found){
        float t = (S-view.from).dot(direction);
        if (t>0)
            return Matrix({Vector3d(t, u, v), S, S_u.cross(S_v).normalized()});
        else return Matrix(3, 3, -1);
    }
    return Matrix(3, 3, -1);
}


/* ------------ Patch Zone -------------*/

Patch::Patch(NurbsSurface* surface, float xmin, float xmax, float ymin, float ymax, bool upper):
    surface(surface),
    u_range(vector<float>{xmin, xmax}),
    v_range(vector<float>{ymin, ymax}) {
        if (not upper){
            p1 = surface->getPoint(xmin, ymin);
            p2 = surface->getPoint(xmax, ymin);
            p3 = surface->getPoint(xmin, ymax);
        }
        else{
            p1 = surface->getPoint(xmax, ymin);
            p2 = surface->getPoint(xmax, ymax);
            p3 = surface->getPoint(xmin, ymax);
        }
        k = (p2 - p1).cross(p3 - p1);
        int m = surface->m, n = surface->n, k = surface->k;
        int imin = xmin*(m - k),
            imax = xmax*(m - k) + k-1,
            jmin = ymin*(n - k),
            jmax = ymax*(n - k) + k-1;
        bBox = Box(surface->ControlPoints[imin][jmin]);
        for (int i=imin; i<=imax; i++)
            for (int j=jmin; j<=jmax; j++)
                bBox.include(surface->ControlPoints[i][j]);
        auto tmp = surface->getPoint((u_range[0]+u_range[1])/2, (v_range[0]+v_range[1])/2);
        centroid = tmp[0];
        norm = ((Vector3d)tmp[1]).cross(tmp[2]).normalized();
}
Box& Patch::getBox(){
    return bBox;
}
Vector3d& Patch::getCentroid(){
    return centroid;
}
bool Patch::intersect(Ray view, IntersectionInfo<Patch>* info){
    float t = (centroid - view.from).dot(view.direction);
    if (u_range[1]-u_range[0] <= triangle_threshold){
        float t = ((p1 - view.from).dot(k)) / (view.direction.dot(k));
        Vector3d p = view.from + t * view.direction,
                 v1 = p1 - p,
                 v2 = p2 - p,
                 v3 = p3 - p,
                 x1 = v1.cross(v2),
                 x2 = v2.cross(v3),
                 x3 = v3.cross(v1);
        if (x1.dot(x2)<0 || x2.dot(x3)<0 || x3.dot(x1)<0)
            return false;
        else{
            info->object = this;
            info->tuv = Vector3d(t, (u_range[0] + u_range[1])/2, (v_range[0] + v_range[1])/2);
            info->hit = centroid;
            info->norm = norm;
            info->intersected = true;
            Color color = surface->image.pick_color(info->tuv[1], info->tuv[2]);
            info->color = Vector3d::max(Vector3d(color.x/255., color.y/255., color.z/255.), Vector3d(0, 0, 0));
            return true;
        }
    }
    else{
        Matrix tuple = surface->intersect(view, u_range, v_range);
        if (tuple[0][0] > 0){
            info->object = this;
            info->tuv = tuple[0];
            info->hit = tuple[1];
            info->norm = tuple[2];
            info->intersected = true;
            Color color = surface->image.pick_color(info->tuv[1], info->tuv[2]);
            info->color = Vector3d::max(Vector3d(color.x/255., color.y/255., color.z/255.), Vector3d(0, 0, 0));
            return true;
        }
        else return false;
    }
}
ostream& operator<<(ostream& stream, const Patch& patch){
    return stream<<"bbox: "<<endl<<patch.bBox<<"centroid: "<<endl<<patch.centroid<<"uv: "<<endl<<patch.u_range<<patch.v_range<<endl;
}

/* ------------ Scene Zone -------------*/

Scene::Scene(Vector3d v1, Vector3d v2, vector<Light> lights, int w, int h, float d, long num_sample, int multi_select, int num_thread):
    image(w, h), ViewYBase(), obj_file(6), v_count(0), View(v1, v2.normalized(), NULL), lights(lights), num_sample(num_sample), multi_select(multi_select), num_thread(num_thread), scale(d){
    ViewYBase = View.direction.get_FinVector();
    ViewXHandle = View.direction.cross(ViewYBase);
}
void Scene::sketch(string s){
    for (int i=0; i!=surfaces.size(); i++)
        sketch(surfaces[i]);
    ofstream fout;
    fout.open(s+".obj");
    for (int i=0; i!=5; i++){
        int size = obj_file[i].size();
        if (size==0)
            continue;
        for (int j=0; j!=size; j++)
            fout << obj_file[i][j];
        fout << endl;
    }
    fout.close();
}
void Scene::sketch(NurbsCurve curve, int num, Color color){
    stringstream s;
    for (int i=0; i!=num; i++){
        Vector3d point = curve.getPoint(1.0*i/(num-1))[0];
        sketch(point, color);
    }
}
void Scene::sketch(Vector3d v, Color color){
    Vector3d v_r = v - View.from;
    float z = v_r.dot(View.direction);
    if (z<0) return;
    else v_r = v_r - z*View.direction;
    float y = v_r.dot(ViewYBase),
           x = -v_r.cross(ViewYBase).dot(View.direction);
}
void Scene::output(string s, int up_to_now){
	List* lst=hitpoints; 
    image.measurement.assign(image.width, vector<Vector3d>(image.height));
	while (lst != NULL) {
		HPoint* hp=lst->id;
		lst=lst->next;
		int i=hp->pix, j=i/image.width;
        i = i%image.width;
        image.measurement[i][j]=image.measurement[i][j]+hp->flux*(1.0/(PI*hp->r2*up_to_now))/multi_select;
	}
    image.draw_from_measurement();
    image.output(s);
}
Color Scene::trace_ray(float x, float y){
    Vector3d direction = View.direction + x/scale*ViewXHandle +
                            y/scale*ViewYBase,
             result(-1, 0, 0);
    Color color;
    Ray ray(View.from, direction.normalized(), NULL);
    IntersectionInfo<Patch> intersection = bvh.intersect(ray, false);

    if (intersection.intersected)
        color = intersection.object->surface->image.pick_color(intersection.tuv[1], intersection.tuv[2]);

    return color;
}
Vector3d Scene::shoot_ray(float x, float y){
    return (View.direction + x/scale*ViewXHandle +
                            y/scale*ViewYBase).normalized();
}
void Scene::render(string file_name, int seed){
    cout<<endl<<"[== HitPointPass ==]"<<endl;
    ProgressBar pbar(image.width);
	//#pragma omp parallel for schedule(dynamic, 1)
	for (int x=0; x<image.width; x++){
		for (int y=0; y<image.height; y++)
            for (int j=0; j!=multi_select; j++)
            {
                int k = x+y*image.width;
                Vector3d d = shoot_ray(2.*(x+hal(13, k))/image.width-1, -2.*(y+hal(16, k))/image.height+1);
                trace(Ray(View.from, d, NULL), 0, true, Vector3d(), Vector3d(2, 2, 2), 0, x+y*image.width);
		}
        pbar.update();
	}
	build_hash_grid(image.width,image.height); 
	
    int num_photon=num_sample/num_thread; 
	Vector3d vw(1,1,1);
    cout<<endl<<"[== PhotonPass ==]"<<endl;
    pbar = ProgressBar(num_photon);
    int num_lights = lights.size();
	#pragma omp parallel for schedule(dynamic, 1)
	for(int i=0;i<num_photon;i++) {
		float p=100.*(i+1)/num_photon;
		int m=num_thread*i; 
		Ray r(Vector3d(0, 0, 0), Vector3d(0, 0, 0), NULL); 
		Vector3d f;
        if ((i+1) % 1000 == 0){
            output(file_name, m);
            cout<<endl<<"Output image at "<<i<<" photons."<<endl;
        }
		for(int j=0;j<num_thread;j++){
			genp(&r,&f,m+j, (m+j)%num_lights, seed); 
			trace(r,0,false,f,vw,m+j, -1);
		}
        pbar.update();
	}
    output(file_name, num_sample);
}
void Scene::build_bvh(){
    int l = surfaces.size();
    patchList.reserve(10000000);
    cout<<endl<<"[== Creating BVH ==]"<<endl;
    for (int i=0; i!=l; i++){
        surfaces[i].build_patches();
        int patch_size = surfaces[i].patchList.size();
        for (int j=0; j!=patch_size; j++){
            patchList.push_back(&surfaces[i].patchList[j]);
        }
    }
    cout<<"[==== Done ====]"<<endl;
    bvh.set_bricks(&patchList, 1);
}

void Scene::build_hash_grid(const int w, const int h) {
	hpbbox.reset();
	List *lst = hitpoints;
	while (lst != NULL) {
		HPoint *hp=lst->id; 
		lst=lst->next; 
		hpbbox.fit(hp->pos);
	}

	Vector3d ssize = hpbbox.max - hpbbox.min;
	float irad = ((ssize.x + ssize.y + ssize.z) / 3.0) / ((w + h) / 2.0) * 2.0;

	hpbbox.reset(); 
	lst = hitpoints; 
	int vphoton = 0; 
	while (lst != NULL) {
		HPoint *hp = lst->id; 
		lst = lst->next;
		hp->r2=irad *irad; 
		hp->n = 0; 
		hp->flux = Vector3d();
		vphoton++; 
		hpbbox.fit(hp->pos-irad); 
		hpbbox.fit(hp->pos+irad);
	}

	hash_s=1.0/(irad*2.0); 
	num_hash = vphoton; 

	hash_grid=new List*[num_hash];
	for (unsigned int i=0; i<num_hash;i++) hash_grid[i] = NULL;
	lst = hitpoints; 
	while (lst != NULL) { 
		HPoint *hp = lst->id; 
		lst = lst->next;
		Vector3d BMin = ((hp->pos - irad) - hpbbox.min) * hash_s;
		Vector3d BMax = ((hp->pos + irad) - hpbbox.min) * hash_s;
		for (int iz = abs(int(BMin.z)); iz <= abs(int(BMax.z)); iz++)
			for (int iy = abs(int(BMin.y)); iy <= abs(int(BMax.y)); iy++)
				for (int ix = abs(int(BMin.x)); ix <= abs(int(BMax.x)); ix++) {
					int hv=f_hash(ix,iy,iz); 
					hash_grid[hv]=ListAdd(hp,hash_grid[hv]);
				}
	}
}

void Scene::genp(Ray* pr, Vector3d* f, int i, int j, int seed) {
    Light light = lights[j];
    Color lc = get_color(light.color);
    Vector3d v(lc.x/255., lc.y/255., lc.z/255.);
    float r = seed / 150.;
    Vector3d vp = v.proceed(r);
	*f = Vector3d(3000, 3000, 3000)*(PI*4.0) * vp;
	float p=2.*PI*hal(0,i/lights.size()), t=2.*acos(sqrt(1.-hal(1,i/lights.size())));
    pr->direction = cos(p)*sin(t)*light.fin + sin(p)*sin(t)*light.handle + cos(t)*light.direction; 
    pr->from = light.from;
}

void Scene::trace(const Ray &r,int dpt,bool m,const Vector3d &fl, const Vector3d &adj, int i, int pixel_index) 
{
    IntersectionInfo<Patch> info = bvh.intersect(r, false);
	dpt++;
    if(!info.intersected||(dpt>=20)){
        return;
    }
 
	int d3=dpt*3;
    const Patch* obj = info.object;
	Vector3d x = r.from+r.direction*info.tuv[0],
             n = info.norm,
             f = info.color;
	Vector3d nl=n.dot(r.direction)<0?n:n*-1; 
	float p=f.x>f.y&&f.x>f.z?f.x:f.y>f.z?f.y:f.z;


	if (obj->surface->image.feature=="mono" || obj->surface->image.feature=="image") {
		float r1=2.*PI*hal(d3-1,i),r2=hal(d3+0,i);
		float r2s=sqrt(r2);
		Vector3d w=nl,u=((fabs(w.x)>.1?Vector3d(0,1):Vector3d(1))%w).normalized();
		Vector3d v=w%u, d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).normalized();
		if (m) {
			HPoint* hp=new HPoint; 
			hp->f=f.mul(adj); 
			hp->pos=x;
			hp->nrm=n; 
			hp->pix = pixel_index; 
			hitpoints = ListAdd(hp, hitpoints);
		} 
		else 
		{
			Vector3d hh = (x-hpbbox.min) * hash_s;
			int ix = abs(int(hh.x)), iy = abs(int(hh.y)), iz = abs(int(hh.z));
			#pragma omp critical
			{
				List* hp = hash_grid[f_hash(ix, iy, iz)]; 
				while (hp != NULL) {
					HPoint *hitpoint = hp->id; 
					hp = hp->next; 
					Vector3d v = hitpoint->pos - x;
					if ((hitpoint->nrm.dot(n) > 1e-3) && (v.dot(v) <= hitpoint->r2)) {
						float g = (hitpoint->n*ALPHA+ALPHA) / (hitpoint->n*ALPHA+1.0);
						hitpoint->r2=hitpoint->r2*g; 
						hitpoint->n++;
						hitpoint->flux=(hitpoint->flux+hitpoint->f.mul(fl)*(1./PI))*g;
					}
				}
			}
			if (hal(d3+1,i)<p) trace(Ray(x,d, obj->surface),dpt,m,f.mul(fl)*(1./p),adj,i, pixel_index);
		}
	} else if (obj->surface->image.feature=="mirror") {
		trace(Ray(x, r.direction-n*2.0*n.dot(r.direction), obj->surface), dpt, m, f.mul(fl), f.mul(adj),i, pixel_index);

	} else if (obj->surface->image.feature=="glass"){
		Ray lr(x,r.direction-n*2.0*n.dot(r.direction), obj->surface); 
		bool into = (n.dot(nl)>0.0);
		float nc = 1.0, nt=1.33, nnt = into?nc/nt:nt/nc, ddn = r.direction.dot(nl), cos2t;

		if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0) return trace(lr,dpt,m,fl,adj,i, pixel_index);

		Vector3d td = (r.direction*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).normalized();
		float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:td.dot(n));
		float Re=R0+(1-R0)*c*c*c*c*c,P=Re;Ray rr(x,td, obj->surface);
        Vector3d fa=f.mul(adj);
		if (m) {
			trace(lr,dpt,m,fl,fa*Re,i, pixel_index);
			trace(rr,dpt,m,fl,fa*(1.0-Re),i, pixel_index);
		} else {
			(hal(d3-1,i)<P)?trace(lr,dpt,m,fl,fa,i, pixel_index):trace(rr,dpt,m,fl,fa,i, pixel_index);
		}
	} else throw invalid_argument("invalid image feature");
}
