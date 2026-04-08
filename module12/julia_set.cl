/* 
julia_set.cl — OpenCL Kernel for Julia Set Computation

 This kernel runs on the GPU. Each work-item (thread) computes the
 color of exactly one pixel by iterating the Julia set formula:
 
      z(n+1) = z(n)^2 + c
 
 where z(0) is derived from the pixel's position in the complex plane,
 and c is a fixed complex constant that defines the fractal's shape.
 */


/* 
 Helper: Smooth Coloring
 
 Raw integer iteration counts produce visible "bands" in the output.
 Smooth coloring uses the final |z| value to interpolate between bands:
 
 smoothed = iter + 1 - log2(log2(|z|))
 
 This gives a continuous floating-point iteration count that eliminates
 banding artifacts.
 
 Parameters:
 iter     — integer iteration at which |z| escaped
 zx, zy   — final z value when escape was detected
 max_iter — cap value; if reached, point is considered "inside" the set
 */
static float smooth_iteration(int iter, float zx, float zy, int max_iter)
{
    if (iter >= max_iter) {
        return (float)max_iter;
    }

    // |z|^2 is already > 4 at this point
    float modulus_sq = zx * zx + zy * zy;
    float log_zn     = log(modulus_sq) / 2.0f;          // log(|z|)
    float nu          = log(log_zn / log(2.0f)) / log(2.0f);

    return (float)iter + 1.0f - nu;
}


/*
 Helper: Sine-Wave Color Palette
 Map a smoothed iteration value to an RGB color using three sine waves
 offset by ~ 120 degrees (2PI/3 radians ~ 2.094).
 */

static void color_pixel(float smoothed, int max_iter,
                        __global unsigned char *output)
{
    float t = smoothed / (float)max_iter * 10.0f;

    output[0] = (unsigned char)(127.5f + 127.5f * sin(t + 0.000f));  // R
    output[1] = (unsigned char)(127.5f + 127.5f * sin(t + 2.094f));  // G
    output[2] = (unsigned char)(127.5f + 127.5f * sin(t + 4.189f));  // B
    output[3] = 255;                                                   // A
}


/* 
 Kernel: julia_set
 
 Launched as a 2D NDRange of size (width × height). Each work-item
 computes one pixel.
 
 Parameters:
 output   — RGBA pixel buffer (width * height * 4 bytes)
 width    — image width in pixels
 height   — image height in pixels
 c_re     — real part of the Julia constant c
 c_im     — imaginary part of the Julia constant c
 max_iter — maximum iteration count before assuming convergence
 zoom     — visible region in the complex plane (smaller = zoom in)
 offset_x — horizontal pan in the complex plane
 offset_y — vertical pan in the complex plane
 */
__kernel void julia_set(
    __global unsigned char *output,
    const int    width,
    const int    height,
    const float  c_re,
    const float  c_im,
    const int    max_iter,
    const float  zoom,
    const float  offset_x,
    const float  offset_y)
{

    // Get thread pixel coordinates

    const int px = get_global_id(0);
    const int py = get_global_id(1);

    if (px >= width || py >= height) return;

   
    // Map pixel coordinates → complex plane
    const float aspect = (float)width / (float)height;

    float zx = ((float)px / (float)width  - 0.5f) * 2.0f * zoom * aspect
               + offset_x;
    float zy = ((float)py / (float)height - 0.5f) * 2.0f * zoom
               + offset_y;


    // Iterate z = z^2 + c
    int iter = 0;

    while (iter < max_iter) {
        const float zx2 = zx * zx;
        const float zy2 = zy * zy;

        if (zx2 + zy2 > 4.0f) break;

        const float new_zx = zx2 - zy2 + c_re;
        zy = 2.0f * zx * zy + c_im;
        zx = new_zx;

        iter++;
    }

    // Color the pixel
    const int idx = (py * width + px) * 4;

    if (iter == max_iter) {
        output[idx + 0] = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        output[idx + 3] = 255;
    } else {
        float smoothed = smooth_iteration(iter, zx, zy, max_iter);
        color_pixel(smoothed, max_iter, &output[idx]);
    }
}
