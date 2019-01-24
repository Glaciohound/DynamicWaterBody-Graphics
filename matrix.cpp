/*************************************************************************
	> File Name: matrix.cpp
	> Author: Han Chi
	> Created Time: Wed May  2 20:10:51 2018
 ************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "matrix.h"
#include <vector>
#include <stdlib.h>

using namespace std;

static double Max=1e10, zero_threshold=1e-5;

bool negligible(double d, double th){
    if (th<0) th=zero_threshold;
    return abs(d)<=th;
}
bool approx(double a, double b, double th){
    if (th<0) th=zero_threshold;
    return abs(a-b)<=th;
}
double frand(){
    return rand() % 10000 / (double)9999;
}
void strip(string& s, char c){
    s = s.substr(s.find(c) + 1);
}

Matrix::Matrix(int m, int n, double d):m(m), n(n), M(m, vector<double>(n, d)){ }
Matrix::Matrix(double* input, int m, int n): Matrix(m, n, 0.0){
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            M[i][j]=input[i*n+j];
}
Matrix::Matrix(vector<Vector3d> input): Matrix(input.size(), 3, 0.0){
    for (int i=0; i!=m; i++){
        M[i][0] = input[i].x;
        M[i][1] = input[i].y;
        M[i][2] = input[i].z;
    }
}
void Matrix::switch_row(int x, int y){
    swap(M[x], M[y]);
}
void Matrix::reset(double value){
    (*this) = Matrix(m, n, value);
}
void Matrix::random(double value){
    srand(time(NULL));
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            M[i][j]=(rand()%1000)*value/1000;
}

bool Matrix::is_diagonal() const{
    return (is_upper() && is_lower());
}
bool Matrix::is_upper() const{
    if (m!=n) return false;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=i; j++)
            if (abs(M[i][j])>zero_threshold) return false;
    return true;
}
bool Matrix::is_lower() const{
    if (m!=n) return false;
    for (int i=0; i!=m; i++)
        for (int j=i+1; j!=n; j++)
            if (abs(M[i][j])>zero_threshold) return false;
    return true;
}
bool Matrix::is_singular() const{
    try {
        inverse();
    }
    catch(const char* error){
        return true;
    }
    return false;
}
bool Matrix::is_vector() const{
    return m==1;
}
double Matrix::mode(string type) const{
    if (type=="2"){
        Matrix cp = *this;
        return sqrt(dot(cp));
    }
    else if (type=="xy"){
        return sqrt(M[0][0]*M[0][0]+M[0][1]*M[0][1]);
    }
    else return 1;
}
double Matrix::norm(string type) const{
    return mode(type);
}
Matrix Matrix::normalized(string type) const{
    return (*this) / mode(type);
}
double Matrix::cond(string type) const{
    return (*this).inverse().mode(type)*(*this).mode(type);
}
double Matrix::get_min() const{
    double output = 1e7;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            output = std::min(output, M[i][j]);
    return output;
}
double Matrix::get_max() const{
    double output = -1e7;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            output = std::max(output, M[i][j]);
    return output;
}
bool exist(int* choice, int i, int n){
    if (choice[i]==n)
        return false;
    for (int j=0; j!=i; j++)
        if (choice[j]==choice[i]) return true;
    return false;
}
double Matrix::determinant() const{
    int counter=0, i=0;
    double sum=0., element;
    int choice[m];
    for (int j=0; j!=m; j++)
        choice[j]=-1;
    while(i!=-1){
        choice[i]++;
        while (exist(choice, i, m))
            choice[i]++;
        if (choice[i]==m){
            choice[i]=-1;
            i--;
            continue;
        }
        else if(i!=m-1)
            i++;
        else{
            counter++;
            if ((counter%4)<=1) element=1;
            else element=-1;
            for (int j=0; j!=m; j++)
                element*=M[j][choice[j]];
            sum+=element;
        }
    }
    return sum;
}
double Matrix::main_eigenvalue(double threshold, int* counter){
    double last=-1.;
    bool finished=false;
    Matrix x(m,1,1.);
    while (!finished){
        x = x * (*this);
        double now=x.mode("infty");
        if (abs(now-last)<threshold)
            finished=true;
        x=x/now;
        if (counter!=NULL)
            (*counter)++;
        last=now;
    }
    return last;
}

Matrix Matrix::inverse2() const{
    Matrix ret = Matrix(2, 2, 0.0);
    double denom = M[0][0]*M[1][1]-M[0][1]*M[1][0];
    ret[0][0] = M[1][1] / denom;
    ret[0][1] = -M[0][1] / denom;
    ret[1][0] = -M[1][0] / denom;
    ret[1][1] = M[0][0] / denom;
    return ret;
}
Matrix Matrix::inverse() const{
    if (m==2 && n==2){
        double denom = 1/(M[0][0]*M[1][1]-M[0][1]*M[1][0]);
        double value[4] = { M[1][1] * denom, -M[0][1] * denom,
                -M[1][0] * denom, M[0][0] * denom };
        return Matrix(value, 2, 2);
    }
    Matrix result=Matrix::Identity(m), copy=*this;
    if (is_diagonal()){
        for (int i=0; i!=m; i++){
            Assert(!negligible(M[i][i]), "inverse: singular\n");
            result[i][i]=1.0/M[i][i];
        }
        return result;
    }
    for (int i=0; i!=m; i++){
        for (int k=i; k!=m; k++)
            if (!negligible(copy[k][i])){
                if (k!=i){
                    result.switch_row(i, k);
                    copy.switch_row(i, k);
                }
                Vector v=copy.pick_col(i)/copy[i][i];
                v[i]=1.0-1.0/copy[i][i];
                result=result-v*result.pick_row(i).transpose();
                copy=copy-v*copy.pick_row(i).transpose();
                break;
            }
        Assert(true, "inverse: singular\n");
    }
    return result;
}
double Matrix::dot(Matrix a) const{
    double sum=0.;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            sum+=a[i][j]*M[i][j];
    return sum;
}


Matrix Matrix::operator*(Matrix A) const{
    if (n==A.n && A.m == 1)
        return (*this)*A.transpose();
    Matrix result(m,A.n,0.0);
    if (is_diagonal()){
        for (int i=0; i!=m; i++)
            for (int j=0; j!=A.n; j++)
                result[i][j]=M[i][i]*A[i][j];
        return result;
    }
    if (A.is_diagonal()){
        for (int j=0; j!=n; j++)
            for (int i=0; i!=m; i++)
                result[i][j]=A[j][j]*M[i][j];
        return result;
    }
    for (int i=0; i!=m; i++)
        for (int j=0; j!=A.n; j++)
            for (int k=0; k!=n; k++)
                result[i][j]+=M[i][k]*A[k][j];
    return result;
}
Matrix Matrix::operator*(double k) const{
    Matrix result = *this;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            result.M[i][j] *= k;
    return result;
}
Matrix Matrix::operator/(double k) const{
    Matrix result = *this;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            result.M[i][j] /= k;
    return result;
}
Matrix Matrix::operator-(const Matrix& A) const{
    Matrix result = *this;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            result.M[i][j] -= A.M[i][j];
    return result;
}
Matrix Matrix::operator+(const Matrix& A) const{
    Matrix result = *this;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            result.M[i][j] += A.M[i][j];
    return result;
}
void Matrix::operator+=(Matrix A){
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            M[i][j] += A.M[i][j];
}
double Matrix::operator^(Matrix& A) const{
    return this->dot(A);
}
bool Matrix::operator==(Matrix& A) const{
    if (m != A.m || n != A.n)
        return false;
    for (int i=0; i!=m; i++)
        for (int j=0; j!=n; j++)
            if (!approx(M[i][j], A[i][j]))
                return false;
    return true;
}
Matrix operator*(double k, Matrix from){
    Matrix result(from.m,from.n,0.0);
    int x=from.m, y=from.n;
    for (int i=0; i!=x; i++)
        for (int j=0; j!=y; j++)
            result[i][j]=from[i][j]*k;
    return result;
}
vector<double>& Matrix::operator[](int i){
    return M[i];
}
ostream &operator<<(ostream& stream, const Matrix& mat){
    stream<<"\33[2K\r";
    stream<<"Matrix "<<mat.m<<"x"<<mat.n<<" "<<":"<<endl;
    stream<< "[";
    for (int i=0; i!=mat.m; i++){
        if (i!=0)
            stream<<" ";
        stream<<"[";
        for (int j=0; j!=mat.n; j++){
            stream<<mat.M[i][j]*!negligible(mat.M[i][j]);
            if (j!=mat.n-1)
                stream<<", ";
        }
        stream<<"]";
        if (i!=mat.m-1) stream<<","<<endl;
        else stream<<"]"<<endl;
    }
    return stream;
}
void Matrix::print(){
    operator<<(cout, *this);
}


Vector Matrix::pick_col(int j) const{
    Vector result(m, 0.0);
    for (int i=0; i!=m; i++)
        result[i]=M[i][j];
    return result;
}
Vector Matrix::pick_row(int i) const{
    Vector result(n,0.0);
    for (int j=0; j!=n; j++)
        result[j]=M[i][j];
    return result;
}
Matrix Matrix::transpose() const{
    Matrix result(n,m,0.0);
    for (int i=0; i!=n; i++)
        for (int j=0; j!=m; j++)
            result.M[i][j]=M[j][i];
    return result;
}
Vector Matrix::pick_diagonal(){
    Vector result(m, 0.0);
    for (int i=0; i!=m; i++)
        result[i]=M[i][i];
    return result;
}
Matrix Matrix::Identity(int n){
    Matrix result(n,n,0.);
    for (int i=0; i!=n; i++)
        result[i][i]=1;
    return result;
}

/*---------------------------------------------------------
 *                  Vector Zone
 *-------------------------------------------------------*/

Vector::Vector(int m, double init): Matrix(1, m, init){
}
Vector::Vector(const Matrix A): Matrix(A){
    if (A.n == 1)
        (*this) = A.transpose();
}
Vector::Vector(const vector<double>& array): Matrix(1, array.size(), 0){
    for (int i=0; i!=n; i++)
        M[0][i] = array[i];
}
Vector3d Vector::to_3d(){
    return (Vector3d)(*this);
}
double& Vector::operator[](int i){
    return M[0][i];
}
Vector operator*(double c, Vector v){
    return c * v;
}
Matrix Vector::transpose(){
    return ((Matrix)(*this));
}
Vector Vector::min(Vector& v, double c){
    Vector output(v.n);
    for (int i=0; i!=v.n; i++)
        output[i] = std::min(v[i], c);
    return output;
}
Vector Vector::max(Vector& v, double c){
    Vector output(v.n);
    for (int i=0; i!=v.n; i++)
        output[i] = std::max(v[i], c);
    return output;
}
Vector Vector::max(Vector v1, Vector v2){
    Vector output(v1.n);
    for (int i=0; i!=v1.n; i++)
        output[i] = std::max(v1[i], v2[i]);
    return output;
}
Vector Vector::min(Vector v1, Vector v2){
    Vector output(v1.n);
    for (int i=0; i!=v1.n; i++)
        output[i] = std::min(v1[i], v2[i]);
    return output;
}
Vector operator*(Matrix M, Vector V){
    return M*((Matrix)V).transpose();
}
Matrix operator*(Vector V, Matrix M){
    return ((Matrix)V).transpose()*M;
}

/* -------------------- Vector3d Zone ------------------------*/

/*
#define loadps(mem)		_mm_load_ps((const double * const)(mem))
#define storess(ss,mem)	_mm_store_ss((double * const)(mem),(ss))
#define storeps(ss,mem) _mm_store_ps((double * const)(mem), (ss))
#define minss			_mm_min_ss
#define maxss			_mm_max_ss
#define minps			_mm_min_ps
#define maxps			_mm_max_ps
#define mulps			_mm_mul_ps
#define divps			_mm_div_ps
#define subps			_mm_sub_ps
#define addps			_mm_add_ps
#define rotatelps(ps)		_mm_shuffle_ps((ps),(ps), 0x39)
#define muxhps(low,high)	_mm_movehl_ps((low),(high))

static const double flt_plus_inf = -logf(0);
static const double __attribute__((aligned(16)))
  ps_cst_plus_inf[4] = {  flt_plus_inf,  flt_plus_inf,  flt_plus_inf,  flt_plus_inf },
  ps_cst_minus_inf[4] = { -flt_plus_inf, -flt_plus_inf, -flt_plus_inf, -flt_plus_inf };
Vector3d::Vector3d(double x, double y, double z): Vector(3, 0.0){
    M[0].reserve(4);
    const __m128
        temp = _mm_setr_ps(x, y, z, 0);
    storeps(temp, &M[0][0]);
}
Vector3d::Vector3d(const Matrix A): Vector(A){
    M[0].reserve(4);
}
Vector3d::Vector3d(const vector<double> array): Vector(3, 0.0){
    M[0].reserve(4);
    storeps(loadps(&array[0]), &M[0][0]);
}
Vector3d Vector3d::cross(Vector3d v){
    Vector3d ret;
    const __m128
        a = loadps(&M[0][0]),
        b = loadps(&v[0]),
        m1 =_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
        m2 = _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))),
        prod = subps(m1, m2);
    storeps(prod, &ret[0]);

    return ret;
}
Vector3d Vector3d::project(Vector3d& x, Vector3d& y, Vector3d& z){
    Vector3d ret;
    const __m128
        vx = _mm_set_ps1(M[0][0]),
        px = mulps(vx, loadps(&x[0])),
        vy = _mm_set_ps1(M[0][1]),
        py = mulps(vy, loadps(&y[0])),
        vz = _mm_set_ps1(M[0][2]),
        pz = mulps(vz, loadps(&z[0])),
        ps = addps(px, py);
    storeps(addps(ps, pz), &ret[0]);
    return ret;
}
double Vector3d::norm(){
    const __m128 v = loadps(&M[0][0]);
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(v, v, 0x71))); 
}
Vector3d Vector3d::normalized(){
    Vector3d ret;
    const __m128
        v = loadps(&M[0][0]),
        inverse_norm = _mm_rsqrt_ps(_mm_dp_ps(v, v, 0x77));
    storeps(_mm_mul_ps(v, inverse_norm), &ret[0]);
    return ret;
}
Vector3d Vector3d::get_FinVector(Vector3d guide){
    Vector3d norm = normalized();
    return (guide - guide.dot(norm)*norm).normalized();
}
Vector3d Vector3d::operator-(Vector3d v) const{
    Vector3d ret;
    const __m128
        v1 = loadps(&M[0][0]),
        v2 = loadps(&v[0]),
        s = subps(v1, v2);
    storeps(s, &ret[0]);
    return ret;
}
Vector3d Vector3d::operator+(Vector3d v) const{
    Vector3d ret;
    const __m128
        v1 = loadps(&M[0][0]),
        v2 = loadps(&v[0]),
        s = addps(v1, v2);
    storeps(s, &ret[0]);
    return ret;
}
double Vector3d::operator^(Vector3d v) const{
    return _mm_cvtss_f32(_mm_dp_ps(loadps(&v[0]),
                loadps(&M[0][0]), 0x71)); 
}
Vector3d Vector3d::operator/(double c) const{
    Vector3d ret;
    const __m128
        v1 = loadps(&M[0][0]),
        v2 = _mm_set_ps1(c),
        s = divps(v1, v2);
    storeps(s, &ret[0]);
    return ret;
}
Vector3d Vector3d::operator*(double c) const{
    Vector3d ret;
    const __m128
        v1 = loadps(&M[0][0]),
        v2 = _mm_set_ps1(c),
        s = mulps(v1, v2);
    storeps(s, &ret[0]);
    return ret;
}
void Vector3d::operator+=(Vector3d v){
    const __m128
        v1 = loadps(&M[0][0]),
        v2 = loadps(&v[0]),
        s = addps(v1, v2);
    storeps(s, &M[0][0]);
};
Vector3d operator*(double c, Vector3d v){
    return Vector3d(c*v[0], c*v[1], c*v[2]);
};
*/

Vector3d::Vector3d(double x, double y, double z):x(x), y(y), z(z){
}
Vector3d::Vector3d(Matrix A){
    if (A.n == 1)
        *this = Vector3d(A[0][0], A[1][0], A[2][0]);
    else
        *this = Vector3d(A[0][0], A[0][1], A[0][2]);
}
Vector3d::Vector3d(const vector<double> array): Vector3d(array[0], array[1], array[2]){
}
void Vector3d::print(){
    cout<<"Vector 3d: "<<x<<", "<<y<<", "<<z<<endl;
}
Vector3d Vector3d::cross(Vector3d v) const{
    return Vector3d(y*v.z - z*v.y,
                    -x*v.z + z*v.x,
                    x*v.y - y*v.x);
}
Vector3d Vector3d::mul(Vector3d v) const{
    return Vector3d(x*v.x, y*v.y, z*v.z);
}
Vector3d Vector3d::project(Vector3d& a, Vector3d& b, Vector3d& c){
    return x*a + y*b + z*c;
}
Vector3d Vector3d::get_FinVector(Vector3d guide){
    Vector3d norm = normalized();
    if (cross(guide).norm() != 0)
        return (guide - guide.dot(norm)*norm).normalized();
    if (x == 0 && y == 0)
        return Vector3d(0, 1, 0);
    else {
        Vector3d z = Vector3d(0, 0, 1);
        return (z - z.dot(norm) * norm).normalized();
    }
}
double Vector3d::norm(){
    return sqrt(x*x + y*y + z*z);
}
Vector3d Vector3d::normalized(){
    return (*this)/sqrt(x*x+y*y+z*z);
}
double Vector3d::get_min(){
    return std::min(std::min(x, y), z);
}
double Vector3d::get_max(){
    return std::max(std::max(x, y), z);
}
Vector3d Vector3d::operator-(Vector3d v) const{
    return Vector3d(x - v.x, y - v.y, z - v.z);
}
Vector3d Vector3d::operator+(Vector3d v) const{
    return Vector3d(x + v.x, y + v.y, z + v.z);
}
Vector3d Vector3d::operator*(Vector3d v) const{
    return Vector3d(x * v.x, y * v.y, z * v.z);
}
Vector3d Vector3d::operator/(Vector3d v) const{
    return Vector3d(x / v.x, y / v.y, z / v.z);
}
double Vector3d::operator^(Vector3d v) const{
    return x * v.x + y * v.y + z * v.z;
}
Vector3d Vector3d::operator%(Vector3d v) const{
    return cross(v);
}
Vector3d Vector3d::operator/(double c) const{
    return Vector3d(x / c, y / c, z / c);
}
Vector3d Vector3d::operator*(double c) const{
    return Vector3d(x * c, y * c, z * c);
}
Vector3d Vector3d::operator+(double c) const{
    return Vector3d(x + c, y + c, z + c);
}
Vector3d Vector3d::operator-(double c) const{
    return Vector3d(x - c, y - c, z - c);
}
void Vector3d::operator+=(Vector3d v){
    x += v.x;
    y += v.y;
    z += v.z;
}
Vector3d operator*(double c, Vector3d v){
    return Vector3d(c*v.x, c*v.y, c*v.z);
}
double& Vector3d::operator[](int i){
    switch (i){
        case 0: return x;
        case 1: return y;
        case 2: return z;
    }
}
double Vector3d::dot(Vector3d v) const{
    return x*v.x + y*v.y + z*v.z;
}
ostream& operator<<(ostream& stream, const Vector3d& v){
    stream<<"Vector: ["<<v.x<<", "<<v.y<<", "<<v.z<<"]"<<endl;
}
Vector3d Vector3d::min(Vector3d u, Vector3d v){
    return Vector3d(std::min(u.x, v.x), std::min(u.y, v.y), std::min(u.z, v.z));
}
Vector3d Vector3d::max(Vector3d u, Vector3d v){
    return Vector3d(std::max(u.x, v.x), std::max(u.y, v.y), std::max(u.z, v.z));
}
Vector3d Vector3d::min(Vector3d u, double c){
    return Vector3d(std::min(u.x, c), std::min(u.y, c), std::min(u.z, c));
}
Vector3d Vector3d::max(Vector3d u, double c){
    return Vector3d(std::max(u.x, c), std::max(u.y, c), std::max(u.z, c));
}
double Vector3d::get_x() const{
    return x;
}
double Vector3d::get_y() const{
    return y;
}
double Vector3d::get_z() const{
    return z;
}
double dec(double r){
    double pi = 3.14159265358979;
    return 0.5*(cos(r*pi)+1);
}
Vector3d Vector3d::proceed(double r){
    double xp, yp, zp;
    double f = 1.0/3;
    if (r<=f){
        xp = dec(3*r)*x + (1-dec(3*r))*z;
        yp = dec(3*r)*y + (1-dec(3*r))*x;
        zp = dec(3*r)*z + (1-dec(3*r))*y;
    }
    else if (r>f && r<=2*f){
        xp = dec(3*(r-f))*z + (1-dec(3*(r-f)))*y;
        yp = dec(3*(r-f))*x + (1-dec(3*(r-f)))*z;
        zp = dec(3*(r-f))*y + (1-dec(3*(r-f)))*x;
    }
    else {
        xp = dec(3*(r-2*f))*y + (1-dec(3*(r-2*f)))*x;
        yp = dec(3*(r-2*f))*z + (1-dec(3*(r-2*f)))*y;
        zp = dec(3*(r-2*f))*x + (1-dec(3*(r-2*f)))*z;
    }
    return Vector3d(xp, yp, zp);
}
