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
    vector<vector<double>> M;
    int m, n;

    Matrix(int m=1, int n=1, double d=0.0);
    Matrix(double* input, int m=1, int n=1);
    Matrix(vector<Vector3d>);
    void input(double*, int=0, int=0);
    
    void switch_row(int, int);
    void reset(double=0.0);
    void random(double=1.0);
    double cond(string) const;
    double mode(string) const;
    double determinant() const;
    double main_eigenvalue(double=1e-6, int* =NULL);
    Matrix inverse() const;
    Matrix inverse2() const;
    bool is_upper() const;
    bool is_lower() const;
    bool is_diagonal() const;
    bool is_singular() const;
    bool is_vector() const;
    double norm(string="2") const;
    Matrix normalized(string="2") const;
    double get_min() const;
    double get_max() const;

    Vector pick_col(int) const;
    Vector pick_row(int) const;
    Matrix transpose() const;
    Vector pick_diagonal();
    Matrix to_diagonal_matrix();

    Matrix operator*(Matrix) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator+(const Matrix&) const;
    double operator^(Matrix&) const; // dot product of matrices
    bool operator==(Matrix&) const;
    Matrix operator/(double) const;
    Matrix operator*(double) const;
    void operator+=(Matrix);
    friend Matrix operator*(double, Matrix);
    vector<double>& operator[](int);
    double dot(Matrix) const;
    friend ostream &operator<<(ostream&, const Matrix&);
    void print();

    static Matrix Identity(int);
};

class Vector: public Matrix{
    public:
    Vector(int dimension=1, double init=0.0);
    Vector(const Matrix);
    Vector(const vector<double>&);
    Vector3d to_3d();
    double& operator[](int);
    friend Vector operator*(double, Vector);
    static Vector min(Vector&, double);
    static Vector max(Vector&, double);
    static Vector min(Vector, Vector);
    static Vector max(Vector, Vector);
    Vector div(Vector) const;
    Matrix transpose();

    friend Vector operator*(Matrix, Vector);
    friend Matrix operator*(Vector, Matrix);
};


class Vector3d{
    public:
    double x, y, z, w;
    /*inline*/ Vector3d(double =0, double =0, double =0);
    /*inline*/ Vector3d(const string);
    /*inline*/ Vector3d(const Matrix);
    /*inline*/ Vector3d(const vector<double>);
    void print();
    /*inline*/ Vector3d cross(Vector3d) const;
    /*inline*/ Vector3d mul(Vector3d) const;
    /*inline*/ Vector3d project(Vector3d&, Vector3d&, Vector3d&);
    /*inline*/ Vector3d get_FinVector(Vector3d =Vector3d(0, 0, 1));
    /*inline*/ double norm();
    /*inline*/ double get_min();
    /*inline*/ double get_max();
    /*inline*/ Vector3d normalized();
    Vector3d proceed(double);

    /*inline*/ Vector3d operator-(Vector3d) const;
    /*inline*/ Vector3d operator+(Vector3d) const;
    double operator^(Vector3d) const; // dot product of matrices
    /*inline*/ Vector3d operator*(Vector3d) const;
    /*inline*/ Vector3d operator%(Vector3d) const;
    /*inline*/ Vector3d operator/(Vector3d) const;
    /*inline*/ Vector3d operator/(double) const;
    /*inline*/ Vector3d operator*(double) const;
    /*inline*/ Vector3d operator+(double) const;
    /*inline*/ Vector3d operator-(double) const;
    /*inline*/ void operator+=(Vector3d);
    friend Vector3d operator*(double, Vector3d);

    /*inline*/ double& operator[](int);
    friend ostream &operator<<(ostream&, const Vector3d&);
    /*inline*/ double dot(const Vector3d) const;
    /*inline*/ static Vector3d min(Vector3d, Vector3d);
    /*inline*/ static Vector3d max(Vector3d, Vector3d);
    /*inline*/ static Vector3d min(Vector3d, double);
    /*inline*/ static Vector3d max(Vector3d, double);
    /*inline*/ double get_x() const;
    /*inline*/ double get_y() const;
    /*inline*/ double get_z() const;
};

class Ray{
    public:
    void* surface;
    Vector3d from, direction;
    Ray(Vector3d f, Vector3d d, void* p): from(f), direction(d), surface(p){
    }
};

bool negligible(double, double=-1);
bool approx(double, double, double=-1);
double frand();

#endif
