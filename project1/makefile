target = main
depends = matrix image utils BVH
objects = $(target:=.o) $(depends:=.o)

source = $(target:=.cpp)
output = trial

compiler = g++
optim = -O3 -fopenmp
debug = #-g -fsanitize=address -fno-omit-frame-pointer 
FLAGS = $(debug) $(optim) -std=c++11

kernel:
	make clear
	make $(target)
	make run
	#make show

$(target): $(objects) $(source) scene.cpp #../matrix.cpp ../image.cpp ../utils.cpp ../BVH
	$(compiler) $(FLAGS) -o $(target) $(objects)

$(target).o: $(source) ../matrix.h ../image.h
	$(compiler) $(FLAGS) -c $(source)
matrix.o: ../matrix.h ../matrix.cpp ../utils.h
	$(compiler) $(FLAGS) -c ../matrix.cpp
utils.o: ../utils.h ../utils.cpp
	$(compiler) $(FLAGS) -c ../utils.cpp
image.o: ../image.cpp ../image.h  
	$(compiler) $(FLAGS) -c ../image.cpp 
BVH.o: ../BVH.cpp ../BVH.h 
	$(compiler) $(FLAGS) -c ../BVH.cpp 

clear:
	rm -rf $(objects) $(target) main.dSYM ppm.o
edit:
	mvim $(source)
show:
	open $(output).bmp
	#open $(output).obj
run:
	./$(target) $(output) 0 100000
