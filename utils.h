/*************************************************************************
	> File Name: myexception.h
	> Author: 
	> Mail: 
	> Created Time: Mon Dec 10 11:35:48 2018
 ************************************************************************/

#ifndef _MYEXCEPTION_H
#define _MYEXCEPTION_H

#include <exception>
#include <sstream>
#include <iomanip> 

class MyException: public std::exception{
    char const* message;
    public:
    MyException(char const* m){
        message = m;
    }
    virtual const char* what() const throw(){
        return message;
    }
};
inline void Assert(bool assertion, char const* message){
    if (!assertion){
        //fprintf(stderr, message, 0); 
        throw MyException(message);
    }
}

class ProgressBar{
    int goal;
    double progress;
    unsigned int time_from, time_last;
    double sec_per_job;
    bool closed;
    int total_length=85;
    public:
    ProgressBar(int goal=1);
    void update(int update=1);
    void show();
    void close();
};

#endif
