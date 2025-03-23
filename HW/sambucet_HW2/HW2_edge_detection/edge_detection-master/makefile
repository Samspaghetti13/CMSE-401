all: serial serial_opt omp_opt

serial: main_process.o png_util.o
	gcc -o serial main_process.c png_util.c -lm -lpng16 -O2

serial_opt: opt_main_process.o png_util.o
    gcc -o serial_opt opt_main_process.c png_util.c -lm -lpng16 -O2 

omp_opt: OpenMP_main_process.o png_util.o
    gcc -o omp_opt OpenMP_main_process.c png_util.c -lm -lpng16 -O2 -fopenmp

main_process.o: main_process.c
    time -f "Serial Compile Time: %E" gcc -c main_process.c -O2

opt_main_process.o: opt_main_process.c
    time -f "serial_opt Compile Time: %E" gcc -c opt_main_process.c -O2

OpenMP_main_process.o: OpenMP_main_process.c
    time -f "OpenMP Compile Time: %E" gcc -c OpenMP_main_process.c -O2 -fopenmp

png_util.o: png_util.c
    gcc -l lpng16 -c png_util.c -O2

test: serial serial_opt omp_opt
    time -f "Time: %E" ./serial ./images/cube.png test_serial.png
    time -f "Time: %E" ./serial_opt ./images/cube.png test_serial_opt.png
    time -f "Time: %E" ./omp_opt ./images/cube.png test_omp.png

clean:
    rm -f *.o serial serial_opt omp_opt