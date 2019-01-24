/*************************************************************************
	> File Name: BVH.cpp
	> Author: 
	> Mail: 
	> Created Time: Sun Jan 13 01:53:25 2019
 ************************************************************************/

#include<iostream>
using namespace std;

#include <algorithm>
#include "BVH.h"

Box::Box(Vector3d& mn, Vector3d& mx)
  : mn(mn), mx(mx) { extent = this->mx - this->mn;}
Box::Box( Vector3d& p)
  : mn(p), mx(p) { extent = mx - mn;}
void Box::include( Vector3d& p) {
  mn = Vector3d::min(mn, p);
  mx = Vector3d::max(mx, p);
  extent = mx - mn;
}
void Box::include( Box& b) {
  mn = Vector3d::min(mn, b.mn);
  mx = Vector3d::max(mx, b.mx);
  extent = mx - mn;
}
uint32_t Box::maxDimension()  {
  uint32_t result = 0;
  if(extent[1] > extent[0]) {
    result = 1;
    if(extent[2] > extent[1]) result = 2;
  } else if(extent[2] > extent[0]) result = 2;
  return result;
}

double Box::surfaceArea()  {
  return 2.f*( extent[0]*extent[2] + extent[0]*extent[1] + extent[1]*extent[2] );
}

#define loadps(mem)		_mm_load_ps((const double * const)(mem))
#define storess(ss,mem)	_mm_store_ss((double * const)(mem),(ss))
#define minss			_mm_min_ss
#define maxss			_mm_max_ss
#define minps			_mm_min_ps
#define maxps			_mm_max_ps
#define mulps			_mm_mul_ps
#define divps			_mm_div_ps
#define subps			_mm_sub_ps
#define rotatelps(ps)		_mm_shuffle_ps((ps),(ps), 0x39)
#define muxhps(low,high)	_mm_movehl_ps((low),(high))	

static const double flt_plus_inf = -logf(0);
static const double __attribute__((aligned(16)))
  ps_cst_plus_inf[4] = {  flt_plus_inf,  flt_plus_inf,  flt_plus_inf,  flt_plus_inf },
  ps_cst_minus_inf[4] = { -flt_plus_inf, -flt_plus_inf, -flt_plus_inf, -flt_plus_inf };

bool Box::intersect(Ray& view, double *tnear, double *tfar)  {
    /*
    const __m128
          plus_inf = loadps(ps_cst_plus_inf),
          minus_inf	= loadps(ps_cst_minus_inf),
         
          box_min	= loadps(&mn.x),
          box_max	= loadps(&mx.x),
          pos	= loadps(&view.from.x),
          inv_dir	= loadps(&view.direction.x),
        
          l1 = divps(subps(box_min, pos), inv_dir),
          l2 = divps(subps(box_max, pos), inv_dir), 
          
          filtered_l1a = minps(l1, plus_inf),
          filtered_l2a = minps(l2, plus_inf), 
          filtered_l1b = maxps(l1, minus_inf),
          filtered_l2b = maxps(l2, minus_inf);

    __m128 lmax = maxps(filtered_l1a, filtered_l2a), 
           lmin = minps(filtered_l1b, filtered_l2b);

    const __m128 lmax0 = rotatelps(lmax), 
          lmin0 = rotatelps(lmin);
    lmax = minss(lmax, lmax0);
    lmin = maxss(lmin, lmin0);

    const __m128 lmax1 = muxhps(lmax,lmax),
          lmin1 = muxhps(lmin,lmin);
    lmax = minss(lmax, lmax1);
    lmin = maxss(lmin, lmin1);

    const bool ret = _mm_comige_ss(lmax, _mm_setzero_ps()) & _mm_comige_ss(lmax,lmin);

    storess(lmin, tnear);
    storess(lmax, tfar);

    return  ret; 
    */
    // originally used for double values
    double pinf = 1e7, minf = -1e7;
    Vector3d r_1 = ((Vector3d)(mx - view.from))/(view.direction), r_2 = ((Vector3d)(mn - view.from))/(view.direction),
             filtered_r1a = Vector3d::min(r_1, pinf), filtered_r2a=Vector3d::min(r_2, pinf),
             filtered_r1b = Vector3d::max(r_1, minf), filtered_r2b=Vector3d::max(r_2, minf);
    double lmax = Vector3d::max(filtered_r1a, filtered_r2a).get_min(),
           lmin = Vector3d::min(filtered_r1b, filtered_r2b).get_max();
    bool output = lmax >= lmin;
    if (output){
        *tnear = lmin;
        *tfar = lmax;
    }
    return output;
}
