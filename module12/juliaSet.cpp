/*
 This program generates a fractal image using the Julia Set algorithm.
 The user can use the arguments listed below to list the available
 GPUs on the system or generate the image. The image size can be
 passed by the user.

 CPU-side orchestrator that discovers GPUs, loads julia_set.cl, and
 launches the fractal kernel on a user-selected device.

 Compile (Linux):
    g++ -std=c++17 julia_set_opencl.cpp -o julia_set -lOpenCL

 Compile (macOS):
    clang++ -std=c++17 julia_set_opencl.cpp -o julia_set -framework OpenCL

 Compile (Windows / MSVC Developer Command Prompt):
    cl julia_set_opencl.cpp /std:c++17 /EHsc ^
      /I "C:\opencl-sdk\include" ^
      /link "C:\opencl-sdk\lib\OpenCL.lib" /out:julia_set.exe

 Usage:
    ./julia_set -operation list_gpu
    ./julia_set -operation generate                     # device 0, 1920x1080
    ./julia_set -operation generate -device 1           # use GPU #1
    ./julia_set -operation generate -device 0 1920 1080 # explicit resolution

 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "helper_functions.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


 //
 // Command Line Arguments
 //

 // Create structure to store default arguments
struct CLIArgs {
    std::string operation;          // "list_gpu" or "generate"
    int         device_index = 0;   // GPU index, default 0
    int         width = 1920;
    int         height = 1080;
};

// Print usage instructions if -help is used
void print_usage(const char* prog_name)
{
    std::cout
        << "Usage:\n"
        << "  " << prog_name << " -operation list_gpu\n"
        << "  " << prog_name << " -operation generate [-device N] [width height]\n"
        << "\n"
        << "Operations:\n"
        << "  list_gpu   List all available OpenCL GPU devices\n"
        << "  generate   Generate Julia set fractal image\n"
        << "\n"
        << "Options:\n"
        << "  -device N  GPU device index (default: 0, see list_gpu)\n"
        << "  width      Image width  in pixels (default: 1920)\n"
        << "  height     Image height in pixels (default: 1080)\n";
}

// Parse CLI arguments
CLIArgs parse_args(int argc, char* argv[])
{
    CLIArgs args;
    bool has_operation = false;

    // Collect positional (non-flag) arguments for width/height
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Parse the operation flag
        if (arg == "-operation") {
            if (i + 1 >= argc) {
                throw std::runtime_error("-operation requires a value: list_gpu or generate");
            }
            args.operation = argv[++i];
            has_operation = true;

            if (args.operation != "list_gpu" && args.operation != "generate") {
                throw std::runtime_error(
                    "Unknown operation '" + args.operation
                    + "'. Must be 'list_gpu' or 'generate'.");
            }

        }
        // Parse the device index flag
        else if (arg == "-device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("-device requires a numeric value");
            }
            args.device_index = std::stoi(argv[++i]);
            if (args.device_index < 0) {
                throw std::runtime_error("Device index must be >= 0");
            }

        }
        // Print usage if -help argument is requested
        else if (arg == "-help") {
            print_usage(argv[0]);
            std::exit(0);

        }
        else {
            positional.push_back(arg);
        }
    }

    if (!has_operation) {
        print_usage(argv[0]);
        throw std::runtime_error("Missing required argument: -operation");
    }

    // Remaining positional args → width, height
    if (positional.size() >= 1) args.width = std::stoi(positional[0]);
    if (positional.size() >= 2) args.height = std::stoi(positional[1]);

    return args;
}



// Structure to hold GPU device information collected during enumeration
struct GPUDeviceInfo {
    cl_platform_id platform;
    cl_device_id   device;
    std::string    name;            // e.g. "NVIDIA GeForce RTX 4090"
    std::string    vendor;          // e.g. "NVIDIA Corporation"
    cl_uint        compute_units;
    int            index;           // our assigned device number
};



// Identify all GPUs
std::vector<GPUDeviceInfo> enumerate_gpus()
{
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return {};

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    std::vector<GPUDeviceInfo> gpus;
    int global_index = 0;

    for (auto& plat : platforms) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0,
            nullptr, &num_devices) != CL_SUCCESS)
            continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices,
            devices.data(), nullptr);

        for (auto& dev : devices) {
            char name[256] = {};
            char vendor[256] = {};
            cl_uint cu = 0;

            clGetDeviceInfo(dev, CL_DEVICE_NAME,
                sizeof(name), name, nullptr);
            clGetDeviceInfo(dev, CL_DEVICE_VENDOR,
                sizeof(vendor), vendor, nullptr);

            gpus.push_back({
                plat, dev,
                std::string(name),
                std::string(vendor),
                cu,
                global_index++
                });
        }
    }
    return gpus;
}


// Print the GPU index, vendor, and model. 
void print_gpu_list()
{
    auto gpus = enumerate_gpus();

    if (gpus.empty()) {
        std::cerr << "No OpenCL GPU devices found.\n"
            << "Make sure your GPU drivers and OpenCL runtime are installed.\n";
        std::exit(1);
    }

    // Calculate column widths for printing
    size_t vendor_width = 6;
    size_t model_width = 5;

    for (const auto& g : gpus) {
        vendor_width = std::max(vendor_width, g.vendor.size());
        model_width = std::max(model_width, g.name.size());
    }

    // Add padding
    vendor_width += 2;
    model_width += 2;

    std::cout << "\nAvailable OpenCL GPU Devices:\n\n";

    // Header
    std::cout << "  " << std::left
        << std::setw(7) << "Index"
        << std::setw(vendor_width) << "Vendor"
        << std::setw(model_width) << "Model"
        << "\n";

    // Separator
    std::cout << "  "
        << std::string(7, '-')
        << std::string(vendor_width, '-')
        << std::string(model_width, '-')
        << "\n";

    // Rows
    for (const auto& g : gpus) {
        std::cout << "  " << std::left
            << std::setw(7) << g.index
            << std::setw(vendor_width) << g.vendor
            << std::setw(model_width) << g.name
            << "\n";
    }

    std::cout << "\nUse -device <index> with -operation generate to select a GPU.\n\n";
}


// Get Device by Index
GPUDeviceInfo get_device_by_index(int device_index)
{
    auto gpus = enumerate_gpus();

    // Validate device index and return the corresponding GPUDeviceInfo
    if (gpus.empty()) {
        throw std::runtime_error(
            "No OpenCL GPU devices found.\n"
            "Make sure your GPU drivers and OpenCL runtime are installed."
        );
    }

    // Check if the device index provided is within the valid range
    if (device_index < 0 || device_index >= static_cast<int>(gpus.size())) {
        std::ostringstream msg;
        msg << "Device index " << device_index << " is out of range.\n"
            << "Available devices (0-" << gpus.size() - 1 << "):\n";
        for (const auto& g : gpus) {
            msg << "  [" << g.index << "] " << g.vendor
                << " — " << g.name << "\n";
        }
        msg << "Run with -operation list_gpu to see full details.";
        throw std::runtime_error(msg.str());
    }

    return gpus[device_index];
}


// Structure to store parameters for the Julia set fractal generation
struct JuliaParams {
    int   width = 1920;
    int   height = 1080;
    int   max_iter = 512;
    float c_re = -0.7f;
    float c_im = 0.27015f;
    float zoom = 1.5f;
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};


/*
 Build Program from External .cl File
 Loads julia_set.cl and compiles it for the specified device
 */
std::pair<cl_program, cl_kernel> build_kernel(
    cl_context context,
    cl_device_id device,
    const std::string& cl_filepath,
    const std::string& entry_point)
{
    // Read the julia_set.cl kernel source file
    std::ifstream srcFile(cl_filepath);
    if (!srcFile.is_open()) {
        throw std::runtime_error(
            "Cannot open kernel file: " + cl_filepath + "\n"
            "Make sure julia_set.cl is in the same directory as the executable."
        );
    }

    std::string source(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char* src = source.c_str();
    size_t length = source.length();

    cl_int err;

    cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithSource failed ("
            + std::to_string(err) + ")");
    }

    err = clBuildProgram(program, 1, &device,
        "-cl-fast-relaxed-math -cl-mad-enable",
        nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &log_size);
        std::vector<char> log_buf(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size, log_buf.data(), nullptr);
        std::string log(log_buf.begin(), log_buf.end());
        clReleaseProgram(program);
        throw std::runtime_error(
            "Kernel compilation failed for " + cl_filepath + ":\n" + log
        );
    }

    cl_kernel kernel = clCreateKernel(program, entry_point.c_str(), &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Kernel entry point '" + entry_point
            + "' not found in " + cl_filepath);
    }

    return { program, kernel };
}



// Generate fractal image and save to file
void generate_fractal(int device_index, int width, int height)
{
    JuliaParams params;
    params.width = width;
    params.height = height;

    const size_t image_size = static_cast<size_t>(params.width)
        * params.height * 4;

    // Get the user-selected device
    auto dev = get_device_by_index(device_index);
    std::cout << "Device [" << dev.index << "]: "
        << dev.vendor << " — " << dev.name << "\n";


    // Create Context and Queue
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &dev.device,
        nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, dev.device,
        CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create command queue");
    }


    // Load and compile kernel
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    try {
        auto result = build_kernel(
            context, dev.device,
            "julia_set.cl",
            "julia_set"
        );
        program = result.first;
        kernel = result.second;
    }
    catch (...) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw;
    }


    // Allocate GPU memory
    cl_mem device_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        image_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to allocate device buffer");
    }


    // Set kernel arguments
    int idx = 0;
    auto set_arg = [&](auto&& val) {
        cl_int e = clSetKernelArg(kernel, idx++, sizeof(val), &val);
        if (e != CL_SUCCESS) {
            clReleaseMemObject(device_buf);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            throw std::runtime_error(
                "Failed to set arg " + std::to_string(idx - 1));
        }
        };

    set_arg(device_buf);
    set_arg(params.width);
    set_arg(params.height);
    set_arg(params.c_re);
    set_arg(params.c_im);
    set_arg(params.max_iter);
    set_arg(params.zoom);
    set_arg(params.offset_x);
    set_arg(params.offset_y);


    // Launch kernel
    constexpr size_t LX = 16, LY = 16;
    auto ceil_div = [](size_t v, size_t m) {
        return ((v + m - 1) / m) * m;
        };

    size_t global[2] = { ceil_div(params.width, LX),
                         ceil_div(params.height, LY) };
    size_t local[2] = { LX, LY };

    // Enqueue kernel
    cl_event kernel_event = nullptr;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
        global, local, 0, nullptr, &kernel_event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(device_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Kernel launch failed ("
            + std::to_string(err) + ")");
    }
    clFinish(queue);


    // Copy image from device to host
    std::vector<unsigned char> pixels(image_size);
    err = clEnqueueReadBuffer(queue, device_buf, CL_TRUE,
        0, image_size, pixels.data(),
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseEvent(kernel_event);
        clReleaseMemObject(device_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to read back pixels");
    }


    // Save image to PNG file
    write_png("julia_set.png", pixels, params.width, params.height);

    // Release memory
    clReleaseEvent(kernel_event);
    clReleaseMemObject(device_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}



int main(int argc, char* argv[])
{
    try {
        CLIArgs args = parse_args(argc, argv);

        // List GPUs available on the system
        if (args.operation == "list_gpu") {
            print_gpu_list();

        }
        // Generate the fractal
        else if (args.operation == "generate") {
            std::cout << "\nImage: " << args.width << "x" << args.height
                << " | Device Selected: " << args.device_index << "\n";

            generate_fractal(args.device_index, args.width, args.height);
        }

    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
