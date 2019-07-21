/*************************************************************************
	> File Name: myexception.cpp
	> Author: 
	> Mail: 
	> Created Time: Mon Dec 10 11:45:03 2018
 ************************************************************************/

#include<iostream>
#include "utils.h"
#include "stdlib.h"
using namespace std;

ProgressBar::ProgressBar(int goal): goal(goal), progress(0), time_from(time(0)), closed(false), sec_per_job(0), time_last(time(0)){
    show();
}

void ProgressBar::update(int update){
    if (closed){
        printf("\33[2K\rpbar explodes!");
    }
    progress += update;
    unsigned int t = time(0);
    float gamma = 0.9999;
    sec_per_job = gamma * sec_per_job + (1-gamma) * (t - time_last);
    time_last = t;
    if (progress>0 && (goal <= 1000 or rand()%goal%(goal/1000)==0))
        show();
    if (progress > goal-1){
        printf("\n[PBar finished]\n");
        close();
    }
}
void ProgressBar::close(){
    closed = true;
}

void ProgressBar::show(){
    ostringstream s;
    s << "\33[2K\r==>[";
    for (int i=0; i!=total_length; i++)
        if (progress/goal*total_length >= i){
            s << "#";
        }
        else s << " ";
    int time_used = time(0) - time_from,
        time_togo = (goal - progress) / progress * time_used,
        hours = time_used / 3600,
        minutes = time_used % 3600 / 60,
        seconds = time_used % 60,
        hourst = time_togo / 3600,
        minutest = time_togo % 3600 / 60,
        secondst = time_togo % 60;
    s << "] "<<setw(6)<<((float)((int)(progress/goal*10000)))/100<<"%, time used= "<<hours<<":"<<minutes<<":"<<seconds<<", to go= "<<hourst<<":"<<minutest<<":"<<secondst;
    fprintf(stderr, "%s", s.str().c_str());
}
