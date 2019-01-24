/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: Mon Dec 10 08:18:00 2018
 ************************************************************************/

#include<iostream>
using namespace std;

#include "../matrix.h"
#include "../image.h"
#include <stdlib.h>

int main(int argc, char* argv[]){
    Image image, image2, image3;
    image.from_file(argv[1]);
    image.calculate_value("mix");
    image.remove(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    for (int i=0; i<=0.2*image.height; i++){
        image.calculate_seam("vertical");
    }
    image = image.cut_seam("vertical");
    image.output(argv[2]);
    /*
    image2 = image.display_seam("horizontal");
    image3 = image.display_energy();
    image2.output("test1");
    image3.output("test2");
    */
    return 0;
}
