#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <omp.h>
#include "png_util.h"
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

void abort_(const char * s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

char ** process_img(char ** img, char ** output, image_size_t sz, int halfwindow, double thresh)
{
    // Average Filter
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int r=0; r<sz.height; r++)
        for(int c=0; c<sz.width; c++)
        {
            int rw_start = max(0, r-halfwindow);
            int cw_start = max(0, c-halfwindow);
            int rw_end = min(sz.height, r+halfwindow+1);
            int cw_end = min(sz.width, c+halfwindow+1);
            double count = 0, tot = 0;
            
            for(int rw = rw_start; rw < rw_end; rw++)
                for(int cw = cw_start; cw < cw_end; cw++)
                {
                    count++;
                    tot += (double) img[rw][cw];
                }
            output[r][c] = (int) (tot/count);
        }

    // Sobel Filters
    double xfilter[3][3];
	double yfilter[3][3];
	xfilter[0][0] = -1;
	xfilter[1][0] = -2;
	xfilter[2][0] = -1;
	xfilter[0][1] = 0;
	xfilter[1][1] = 0;
	xfilter[2][1] = 0;
	xfilter[0][2] = 1;
	xfilter[1][2] = 2;
	xfilter[2][2] = 1;
	
    yfilter[0][0] = -1;
    yfilter[0][1] = -2;
    yfilter[0][2] = -1;
    yfilter[1][0] = 0;
    yfilter[1][1] = 0;
    yfilter[1][2] = 0;
    yfilter[2][0] = 1;
    yfilter[2][1] = 2;
    yfilter[2][2] = 1;

    double *gradient = (double *) malloc(sz.width * sz.height * sizeof(double));
    double **g_img = malloc(sz.height * sizeof(double*));
    for (int r=0; r<sz.height; r++)
        g_img[r] = &gradient[r * sz.width];

    // Gradient filter
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int r=1; r<sz.height-1; r++)
        for(int c=1; c<sz.width-1; c++)
        {
            double Gx = 0, Gy = 0;
            for(int rw=0; rw<3; rw++)
                for(int cw=0; cw<3; cw++)
                {
                    Gx += ((double) output[r+rw-1][c+cw-1]) * xfilter[rw][cw];
                    Gy += ((double) output[r+rw-1][c+cw-1]) * yfilter[rw][cw];
                }
            g_img[r][c] = sqrt(Gx*Gx + Gy*Gy);
        }

    // Thresholding
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int r=0; r<sz.height; r++)
        for(int c=0; c<sz.width; c++)
            output[r][c] = (g_img[r][c] > thresh) ? 255 : 0;
}

int main(int argc, char **argv)
{
    // Code currently supports only grayscale images
    int channels = 1; 
    double thresh = 50;
    int halfwindow = 3;

    // Ensure at least two input arguments
    if (argc < 3 )
        abort_("Usage: process <file_in> <file_out> <halfwindow=3> <threshold=50>");

    // Set optional arguments
    if (argc > 3 )
        halfwindow = atoi(argv[3]);
    if (argc > 4 )
        thresh = (double) atoi(argv[4]);

    // Allocate memory for images
    image_size_t sz = get_image_size(argv[1]);
    char *s_img = (char *) malloc(sz.width * sz.height * channels * sizeof(char));
    char *o_img = (char *) malloc(sz.width * sz.height * channels * sizeof(char));

    // Read input image
    read_png_file(argv[1], s_img, sz);

    // Convert 1D image arrays to 2D pointers
    char **img = malloc(sz.height * sizeof(char*));
    char **output = malloc(sz.height * sizeof(char*));
    for (int r=0; r<sz.height; r++)
    {
        img[r] = &s_img[r * sz.width];
        output[r] = &o_img[r * sz.width];
    }

    // Process image
    process_img(img, output, sz, halfwindow, thresh);

    // Write output image
    write_png_file(argv[2], o_img, sz);

    return 0;
}