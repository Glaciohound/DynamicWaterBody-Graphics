/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: Mon Dec 10 08:18:00 2018
 ************************************************************************/

#include<iostream>
using namespace std;

#include "../matrix.h"
#include "../BVH.h"
#include "scene.cpp"
#include <stdlib.h>

int main(int argc, char* argv[]){
    srand(time(0));
    if (argv[1][0] == 'd'){
        Image image;
        image.from_file(argv[2]);
        image.denoise(atoi(argv[5]), atof(argv[4])).output(argv[3]);
    }
    else{
        string file_name = (argc>=2)? argv[1]:"output";
        int seed = atoi(argv[2]);
        int length = atoi(argv[3]);
        construct_0(seed, length);
        scene->render("outputs/"+file_name+to_string(seed), seed);
        return 0;
    }
}
