/*************************************************************************
	> File Name: BVH.h
	> Author: 
	> Mail: 
	> Created Time: Sun Jan 13 01:32:00 2019
 ************************************************************************/

#ifndef _BVH_H
#define _BVH_H

#include <vector>
#include "matrix.h"
#include <stdint.h>
#include <emmintrin.h> 
#include <immintrin.h>

template <class T>
struct IntersectionInfo;
struct BVHnode;
template <class T>
struct BVHshot;


struct Box {
    Vector3d mn, mx, extent;
    Box() { }
    Box(Vector3d& mn, Vector3d& mx);
    Box(Vector3d& p);

    bool intersect(Ray& view, float *tnear, float *tfar) ;
    void include( Vector3d& p);
    void include( Box& b);
    uint32_t maxDimension() ;
    float surfaceArea() ;

    friend ostream& operator<<(ostream& stream, const Box& box){
        return stream<<"Box:\n"<<box.mn<<box.mx;
    }
};
template <class T>
class BVH {
    public:
    uint32_t nNodes, nLeafs, leafSize;
    std::vector<T*>* bricks;
    vector<BVHnode> flatTree;

    BVH();
    BVH(std::vector<T*>* objects, uint32_t leafSize=4);
    IntersectionInfo<T> intersect(Ray view, bool occlusion);
    void build();
    void set_bricks(std::vector<T*>* objects, uint32_t leafSize=4);
};

/* ---------------- Utility ---------------------- */

template <class T>
struct IntersectionInfo {
    T* object; 
    Vector3d hit, tuv, norm; 
    bool intersected = false;
    Vector3d color;
};

struct BVHnode {
    Box bbox;
    uint32_t start, nbricks, rightOffset;
    friend ostream& operator<<(ostream& stream, const BVHnode& node){
        return stream<<node.bbox<<node.start<<' '<<node.nbricks<<' '<<node.rightOffset<<endl; }
};

template <class T>
struct BVHshot {
    uint32_t i;
    float mint;
    BVHshot() { }
    BVHshot(int _i, float _mint) : i(_i), mint(_mint) { }
};

struct BVHentry {
  uint32_t parent;
  uint32_t start, end;
};

/* ----------------- BVH Zone -------------------- */

template <class T>
BVH<T>::BVH(){};
template <class T>
BVH<T>::BVH(std::vector<T*>* objects, uint32_t leafSize):
        nNodes(0), nLeafs(0), 
        leafSize(leafSize),
        bricks(objects){
    build();
}

template <class T>
IntersectionInfo<T> BVH<T>::intersect(Ray view, bool occlusion)  {
    IntersectionInfo<T> ret;
    ret.tuv[0] = 999999999.f;
    ret.object = NULL;
    float bbhits[4];
    int32_t closer, other;

    BVHshot<T> todo[64];
    int32_t stackptr = 0;

    todo[stackptr].i = 0;
    todo[stackptr].mint = -9999999.f;
    float self_threshold = 1e-4;

    while(stackptr>=0) {
        int ni = todo[stackptr].i;
        float near = todo[stackptr].mint;
        stackptr--;
        const BVHnode &node(flatTree[ ni ]);

        if (near > ret.tuv[0])
            continue;

        if (node.rightOffset == 0)
            for(uint32_t o=0;o<node.nbricks;++o) {
                T* obj = (*bricks)[node.start+o];
                IntersectionInfo<T> current;

                bool intersected = obj->intersect(view, &current);
                if (intersected && (obj->surface != view.surface ||
                            current.tuv[0] > self_threshold)){
                    if(occlusion)
                        return current;
                    if (current.tuv[0] < ret.tuv[0])
                        ret = current;
                }
            }
        else{
            bool hitc0 = flatTree[ni+1].bbox.intersect(view, bbhits, bbhits+1);
            bool hitc1 = flatTree[ni+node.rightOffset].bbox.intersect(view, bbhits+2, bbhits+3);

            if(hitc0 && hitc1) {
                closer = ni+1;
                other = ni+node.rightOffset;

                if (bbhits[2] < bbhits[0]) {
                    std::swap(bbhits[0], bbhits[2]);
                    std::swap(bbhits[1], bbhits[3]);
                    std::swap(closer,other);
                }

                todo[++stackptr] = BVHshot<T>(other, bbhits[2]);
                todo[++stackptr] = BVHshot<T>(closer, bbhits[0]);
            }
            else if (hitc0) 
                todo[++stackptr] = BVHshot<T>(ni+1, bbhits[0]);
            else if(hitc1)
                todo[++stackptr] = BVHshot<T>(ni + node.rightOffset, bbhits[2]);
        }
    }
    if(ret.intersected)
    ret.hit = view.from + view.direction * ret.tuv[0];

    return ret;
}

template <class T>
void BVH<T>::build() {
    BVHentry todo[128];
    uint32_t stackptr = 0;
    uint32_t Untouched = 0xffffffff;
    uint32_t TouchedTwice = 0xfffffffd;

    todo[stackptr].start = 0;
    todo[stackptr].end = bricks->size();
    todo[stackptr].parent = 0xfffffffc;
    stackptr++;

    BVHnode node;
    flatTree.reserve(bricks->size()*2);

    while(stackptr > 0) {
        BVHentry &bnode( todo[--stackptr] );
        uint32_t start = bnode.start;
        uint32_t end = bnode.end;
        uint32_t nbricks = end - start;

        nNodes++;
        node.start = start;
        node.nbricks = nbricks;
        node.rightOffset = Untouched;

        Box bb((*bricks)[start]->getBox());
        Box bc((*bricks)[start]->getCentroid());
        for(uint32_t p = start+1; p < end; ++p) {
            bb.include( (*bricks)[p]->getBox());
            bc.include( (*bricks)[p]->getCentroid());
        }
        node.bbox = bb;

        if(nbricks <= leafSize) {
            node.rightOffset = 0;
            nLeafs++;
        }

        flatTree.push_back(node);
        if(bnode.parent != 0xfffffffc) {
            flatTree[bnode.parent].rightOffset --;
            if( flatTree[bnode.parent].rightOffset == TouchedTwice ) {
                flatTree[bnode.parent].rightOffset = nNodes - 1 - bnode.parent;
            }
        }

        if(node.rightOffset == 0)
            continue;
        uint32_t split_dim = bc.maxDimension();
        float split_coord = .5f * (bc.mn[split_dim] + bc.mx[split_dim]);

        uint32_t mid = start;
        for(uint32_t i=start;i<end;++i) {
            if( (*bricks)[i]->getCentroid()[split_dim] < split_coord ) {
                std::swap( (*bricks)[i], (*bricks)[mid] );
                ++mid;
            }
        }

        if(mid == start || mid == end) {
            mid = start + (end-start)/2;
        }

        todo[stackptr].start = mid;
        todo[stackptr].end = end;
        todo[stackptr].parent = nNodes-1;
        stackptr++;

        todo[stackptr].start = start;
        todo[stackptr].end = mid;
        todo[stackptr].parent = nNodes-1;
        stackptr++;
    }
}
template <class T>
void BVH<T>::set_bricks(std::vector<T*>* objects, uint32_t leafSize){
    nNodes = 0;
    this->leafSize = leafSize;
    bricks = objects;
    build();
}

#endif
