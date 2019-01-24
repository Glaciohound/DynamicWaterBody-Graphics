/*************************************************************************
	> File Name: matrix.h
	> Author: Han Chi
	> Created Time: Wed May  2 19:28:00 2018
 ************************************************************************/

#ifndef _MATRIX_H
#define _MATRIX_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <emmintrin.h> 
#include <smmintrin.h>
#include "utils.h"
using namespace std;

class Vector;
class Vector3d;


class Matrix{
    public:
    vector<vector<float>> M;
    int m, n;

    Matrix(int m=1, int n=1, float d=0.0);
    Matrix(float* input, int m=1, int n=1);
    Matrix(vector<Vector3d>);
    void input(float*, int=0, int=0);
    
    void switch_row(int, int);
    void reset(float=0.0);
    void random(float=1.0);
    float cond(string) const;
    float mode(string) const;
    float determinant() const;
    float main_eigenvalue(float=1e-6, int* =NULL);
    Matrix inverse() const;
    Matrix inverse2() const;
    bool is_upper() const;
    bool is_lower() const;
    bool is_diagonal() const;
    bool is_singular() const;
    bool is_vector() const;
    float norm(string="2") const;
    Matrix normalized(string="2") const;
    float get_min() const;
    float get_max() const;

    Vector pick_col(int) const;
    Vector pick_row(int) const;
    Matrix transpose() const;
    Vector pick_diagonal();
    Matrix to_diagonal_matrix();

    Matrix operator*(Matrix) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator+(const Matrix&) const;
    float operator^(Matrix&) const; // dot product of matrices
    bool operator==(Matrix&) const;
    Matrix operator/(float) const;
    Matrix operator*(float) const;
    void operator+=(Matrix);
    friend Matrix operator*(float, Matrix);
    vector<float>& operator[](int);
    float dot(Matrix) const;
    friend ostream &operator<<(ostream&, const Matrix&);
    void print();

    static Matrix Identity(int);
};

class Vector: public Matrix{
    public:
    Vector(int dimension=1, float init=0.0);
    Vector(const Matrix);
    Vector(const vector<float>&);
    Vector3d to_3d();
    float& operator[](int);
    friend Vector operator*(float, Vector);
    static Vector min(Vector&, float);
    static Vector max(Vector&, float);
    static Vector min(Vector, Vector);
    static Vector max(Vector, Vector);
    Vector div(Vector) const;
    Matrix transpose();

    friend Vector operator*(Matrix, Vector);
    friend Matrix operator*(Vector, Matrix);
};


class __attribute__((aligned(16))) Vector3d{
    public:
    float x, y, z, w;
    /*inline*/ Vector3d(float =0, float =0, float =0);
    /*inline*/ Vector3d(const string);
    /*inline*/ Vector3d(const Matrix);
    /*inline*/ Vector3d(const vector<float>);
    void print();
    /*inline*/ Vector3d cross(Vector3d) const;
    /*inline*/ Vector3d mul(Vector3d) const;
    /*inline*/ Vector3d project(Vector3d&, Vector3d&, Vector3d&);
    /*inline*/ Vector3d get_FinVector(Vector3d =Vector3d(0, 0, 1));
    /*inline*/ float norm();
    /*inline*/ float norm(string s);
    /*inline*/ float get_min();
    /*inline*/ float get_max();
    /*inline*/ Vector3d normalized();
    Vector3d proceed(float);

    /*inline*/ Vector3d operator-(Vector3d) const;
    /*inline*/ Vector3d operator+(Vector3d) const;
    float operator^(Vector3d) const; // dot product of matrices
    /*inline*/ Vector3d operator*(Vector3d) const;
    /*inline*/ Vector3d operator%(Vector3d) const;
    /*inline*/ Vector3d operator/(Vector3d) const;
    /*inline*/ Vector3d operator/(float) const;
    /*inline*/ Vector3d operator*(float) const;
    /*inline*/ Vector3d operator+(float) const;
    /*inline*/ Vector3d operator-(float) const;
    /*inline*/ void operator+=(Vector3d);
    friend Vector3d operator*(float, Vector3d);

    /*inline*/ float& operator[](int);
    friend ostream &operator<<(ostream&, const Vector3d&);
    /*inline*/ float dot(const Vector3d) const;
    /*inline*/ static Vector3d min(Vector3d, Vector3d);
    /*inline*/ static Vector3d max(Vector3d, Vector3d);
    /*inline*/ static Vector3d min(Vector3d, float);
    /*inline*/ static Vector3d max(Vector3d, float);
    /*inline*/ float get_x() const;
    /*inline*/ float get_y() const;
    /*inline*/ float get_z() const;
};

class Ray{
    public:
    void* surface;
    Vector3d from, direction;
    Ray(Vector3d f, Vector3d d, void* p): from(f), direction(d), surface(p){
    }
};

bool negligible(float, float=-1);
bool approx(float, float, float=-1);
float frand();

#endif
