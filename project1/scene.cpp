#include<iostream>
using namespace std;

#include "../matrix.h"
#include "../image.h"
#include <cmath>

using namespace std;

Scene* scene;
Scene scene_0(
    Vector3d(16, 0, -4),
    Vector3d(-1, 0, 0),
    vector<Light>{
        Light(Vector3d(-5, 0, 5), Vector3d(0, 0, -1), 1, "dark"),
        Light(Vector3d(13, -13, 5), Vector3d(-1, 0.5, -0.5), 0.4, "red"),
        Light(Vector3d(13, 13, 5), Vector3d(-1, -0.5, -0.5), 0.4, "blue")
        },
    1900, 1400,
    1.,
    100000000,
    1, 1000
);
Scene scene_(
        Vector3d()
        );

double f_wing(double theta, double wing, double radius){
    return radius + exp(10*cos(3*theta))*(wing-min(wing, radius))/22000;
}
double f_SoftTriangle(double theta){
    return 1 + pow(cos(3*theta)+3, 3)/120;
}
double f_egg(double x){
    return sqrt(1-pow(1-x*2,2));
}
double recip(double x){
    return 1/(x+0.1);
}
NurbsCurve circle("circle");
NurbsCurve egg(f_egg, false);


Vector3d norm(NurbsSurface* surface, double u, double v){
    Matrix M = surface->getPoint(u, v);
    Vector3d su = M.pick_col(1), sv = M.pick_col(2);
    return su.cross(sv);
}
void construct_0(int seed, int length){

    /*
    NurbsSurface body(50, 50, "world", 3);

    body.cylinder_surface(
            NurbsCurve(vector<Vector3d>{Vector3d(-7, 2, -2), Vector3d(-7, 2, 8)}),
            circle,
            egg*5,
            Vector3d(0, 1, 0));
    */

    string same_color = "gray";
    double up = 15, down = -15, left = -15, right = 15, front = -15, back = 20, ground = -1;

    NurbsSurface floor(2, 2, same_color, 1);
    floor.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, down), Vector3d(back, left, down)}),
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, down), Vector3d(front, right, down)}));

    NurbsSurface water(20, 20, "glass", 2);
    water.water_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, ground), Vector3d(back, left, ground)}),
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, ground), Vector3d(front, right, ground)}),
            0.8, 0.4, seed);

    NurbsSurface wall_front(2, 2, same_color, 1);
    wall_front.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, down), Vector3d(front, right, down)}),
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, down), Vector3d(front, left, up)}));

    NurbsSurface wall_behind(2, 2, same_color, 1);
    wall_behind.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(back, left, down), Vector3d(back, left, up)}),
            NurbsCurve(vector<Vector3d>{Vector3d(back, left, down), Vector3d(back, right, down)}));

    NurbsSurface wall_left(2, 2, same_color, 1);
    wall_left.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(back, left, down), Vector3d(front, left, down)}),
            NurbsCurve(vector<Vector3d>{Vector3d(back, left, down), Vector3d(back, left, up)}));

    NurbsSurface wall_right(2, 2, same_color, 1);
    wall_right.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(front, right, down), Vector3d(back, right, down)}),
            NurbsCurve(vector<Vector3d>{Vector3d(front, right, down), Vector3d(front, right, up)}));

    NurbsSurface wall_up(2, 2, same_color, 1);
    wall_up.binary_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, up), Vector3d(front, right, up)}),
            NurbsCurve(vector<Vector3d>{Vector3d(front, left, up), Vector3d(back, left, up)}));


    scene = &scene_0;
    scene->surfaces = {floor, wall_front, wall_left, wall_right, wall_up, wall_behind, water};
    scene->num_sample = length;
    scene->build_bvh();
}

void construct_1(int seed, int length){
    double size = 30, ground = -10;
    NurbsSurface water(400, 400, "glass", 2);
    water.water_expand(
            NurbsCurve(vector<Vector3d>{Vector3d(-size, -size, ground),
                                        Vector3d(size, -size, ground)}),
            NurbsCurve(vector<Vector3d>{Vector3d(-size, -size, ground), 
                                        Vector3d(-size, size, ground)}),
            1, 0.15, seed);

    NurbsSurface support(50, 50, "gray", 2);
    support.cylinder_surface(
            NurbsCurve(vector<Vector3d>{Vector3d(0, 0, -12), Vector3d(0, 0, 3)}),
            NurbsCurve(f_SoftTriangle, true)*2,
            NurbsCurve(recip, false));

    NurbsSurface rocket(100, 50, "white", 2);
    rocket.rocket_body(
            NurbsCurve(vector<Vector3d>{
                Vector3d(13, 0, 0),
                Vector3d(11, 0, 10),
                Vector3d(11.5, 0, 30),
                Vector3d(12, 0, 80),
                Vector3d(12, 0, 120),
                Vector3d(5, 0, 140),
                Vector3d(0, 0, 140),
                }, 3),
            NurbsCurve(vector<Vector3d>{
                Vector3d(0, 0, 0),
                Vector3d(10, 0, 0.01),
                Vector3d(30, 0, 0.02),
                Vector3d(30, 0, 0.04),
                Vector3d(30, 0, 0.08),
                Vector3d(10, 0, 0.30),
                Vector3d(0, 0, 0.95),
                Vector3d(0, 0, 1),
                }, 2),
            f_wing
            );

    NurbsSurface gym(20, 20, "white", 2);
    gym.cylinder_surface(
            NurbsCurve(vector<Vector3d>{
                Vector3d(0, 0, 0),
                Vector3d(0, 0, 170),
                }),
            NurbsCurve(f_SoftTriangle, true)*400,
            NurbsCurve(vector<Vector3d>{
                Vector3d(1, 0, 0),
                Vector3d(1, 0, 0.5),
                Vector3d(0.7, 0, 1),
                })
            );

    scene = &scene_0;
    scene->surfaces = {water, support, rocket};
    scene->sketch("rocket");
    exit(0);
    //scene->build_bvh();
}
