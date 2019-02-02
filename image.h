/*************************************************************************
	> File Name: image.h
	> Author: 
	> Mail: 
	> Created Time: Tue Dec 11 16:12:06 2018
 ************************************************************************/

#ifndef _IMAGE_H
#define _IMAGE_H

#include "matrix.h"
#include "utils.h"
#include <set>
#include "BVH.h"
#include <emmintrin.h>
using namespace std;

/* -------------- Utility ---------------*/

struct hash_box {Vector3d min, max; // axis aligned bounding box
	inline void fit(const Vector3d &p)
	{
		if (p.x<min.x)min.x=p.x; // min
		if (p.y<min.y)min.y=p.y; // min
		if (p.z<min.z)min.z=p.z; // min
		max.x=std::max(p.x, max.x);
		max.y=std::max(p.y, max.y);
		max.z=std::max(p.z, max.z);
	}
	inline void reset() {
		min=Vector3d(1e20,1e20,1e20); 
		max=Vector3d(-1e20,-1e20,-1e20);
	}
};
struct HPoint {
	Vector3d f,pos,nrm,flux; 
	float r2; 
	unsigned int n; // n = N / ALPHA in the paper
	int pix;
};
struct List {HPoint *id; List *next;};
List* ListAdd(HPoint *i,List* h);

typedef Vector3d Color;
Color get_color(string);
class Vector3d;
class NurbsCurve;
class NurbsSurface;
class Patch;
static Vector3d Vector3d_static;

struct Light{
    Vector3d from, direction, fin, handle;
    string color;
    float range;
    Light(Vector3d, Vector3d, float, string);
};

/* ------------------ Main Zone -------------*/

struct Pair{
    int i;
    float v;
    Pair(int i=0, float v=0):i(i), v(v){}
};
class Image{
    public:
        int width, height, radius;
        string feature;
        Color color;
        vector<vector<Color>> image;
        vector<vector<Vector3d>> measurement;
        vector<vector<float>> values;
        vector<vector<Pair>> DP_values;
        vector<int> seam;
        vector<vector<int>> seamsx;
        vector<vector<int>> seamsy;

        Image(int width=640, int height=480, string="mono");
        void output(string outfile_name);
        void reset(int width=640, int height=480);
        bool from_file(string infile_name);
        void draw_point(float =0, float =0, Color=get_color("white"));
        void draw_point(int =0, int =0, Color=get_color("white"));
        Color pick_color(int, int);
        Color pick_color(float, float);
        void draw_from_measurement();
        void calculate_value(string);
        void calculate_seam(string);
        Image display_seam(string);
        Image cut_seam(string);
        Image expand_seam(string);
        Image display_energy();
        Image transpose();
        bool contains(string, int, int);
        void protect(int, int, int, int);
        void remove(int, int, int, int);
        Image denoise(int size, float alpha);
};
class Scene{
    public:
    Ray View;
    vector<Light> lights;
    Vector3d ViewYBase;
    Vector3d ViewXHandle;
    long num_sample;
    int num_thread;
    int multi_select;
    float scale;
    Image image;
    vector<vector<string>> obj_file;
    vector<NurbsSurface> surfaces;
    int v_count;
    BVH<Patch> bvh;
    vector<Patch*> patchList;
    unsigned int num_hash, num_photon;
    float hash_s; List **hash_grid; List *hitpoints = NULL; hash_box hpbbox;

    Scene(Vector3d from, Vector3d direction, vector<Light> lights,
            int width, int height,
            float scale,
            long num_sample,
            int multi_select, int num_thread);
    void sketch(NurbsCurve, int =200, Color =get_color("white"));
    void sketch(Vector3d, Color color);
    void sketch(NurbsSurface, Color =get_color("white"));
    void sketch(string);
    void output(string outfile_name, int );
    void build_bvh();
    void render(string, int);
    Color trace_ray(float, float);
    Vector3d shoot_ray(float, float);

    inline unsigned int f_hash(const int ix, const int iy, const int iz);
    void build_hash_grid(const int w, const int h);
    void genp(Ray* pr, Vector3d* f, int i, int, int);
    void trace(const Ray &r,int dpt,bool m,const Vector3d &fl, const Vector3d &adj, int i, int pixel_index);
};

struct bezier_coefficient_item{
    int n, k;
    vector<vector<vector<float>>> coeffs;
    bezier_coefficient_item(int =0,int =0);
};
Matrix Bezier(int, int, float);
class NurbsCurve{
    public:
    static int k_default, n_default;
    vector<Vector3d> ControlPoints;
    int n, k;
    NurbsCurve(int n=1, int k=k_default);
    NurbsCurve(vector<Vector3d>, int k=k_default);
    NurbsCurve(string command, int n=n_default, int k=k_default);
    NurbsCurve(float (*)(float), bool polar_coordinates=false, int n=n_default, int k=k_default);
    NurbsCurve friend operator*(float, NurbsCurve);
    NurbsCurve operator*(float);
    Matrix getPoint(float t);
    float get_min();
    float get_max();
    float get_t_from_height(float);
    int get_n();
};

class NurbsSurface{
    public:
    static int k_default, n_default, m_default;
    int m, n, k;
    Image image;
    vector<vector<Vector3d>> ControlPoints;
    vector<Patch> patchList;
    NurbsSurface(int =m_default, int =n_default, string="", int =k_default);
    void cylinder_surface(NurbsCurve, NurbsCurve, NurbsCurve, Vector3d guide_direction=Vector3d(0, 0, 1));
    void rocket_body(NurbsCurve, NurbsCurve, float (*)(float, float, float));
    void binary_expand(NurbsCurve, NurbsCurve);
    void water_expand(NurbsCurve, NurbsCurve, float, float, float);
    void build_patches();
    Matrix getPoint(float u, float v);
    Matrix intersect(Ray view, Vector u_range=Vector(vector<float>({0, 1})), Vector v_range=Vector(vector<float>({0, 1})));
};

class Patch{
    public:
        NurbsSurface* surface;
        Vector u_range, v_range;
        Box bBox;
        Vector3d centroid, norm, p1, p2, p3, k;
        Patch(NurbsSurface*, float, float, float, float, bool);
        Box& getBox();
        Vector3d& getCentroid();
        bool intersect(Ray view, IntersectionInfo<Patch>*);
        
        friend ostream& operator<<(ostream&, const Patch&);
};

#endif
