/*
Date: 4/5/2026
File: helper_functions.h

This file contains helper functions used by the following programs:
  - add_gaussian_noise.cu
  - calculate_pratt_fom.cu
  - detect_edge.cu
The functions on this file include:
  - BMP file reading and writing (24-bit RGB and 8-bit grayscale)
  - Directory listing and path manipulation
  - A simple JSON configuration parser for the configurations.json file

*/


#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

namespace fs = std::filesystem;

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)


//
// BMP on-disk structures 
//

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

// 
// Image – a padding-free, row-major pixel buffer
//

struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

// 
// loadBMP
//

inline bool loadBMP(const fs::path& path, Image& img) {
    FILE* f = fopen(path.string().c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[loadBMP] Cannot open '%s'\n", path.string().c_str());
        return false;
    }

    BMPFileHeader fh{};
    BMPInfoHeader ih{};

    if (fread(&fh, sizeof(fh), 1, f) != 1 || fread(&ih, sizeof(ih), 1, f) != 1) {
        fprintf(stderr, "[loadBMP] Cannot read headers: '%s'\n", path.string().c_str());
        fclose(f); return false;
    }
    if (fh.bfType != 0x4D42) {
        fprintf(stderr, "[loadBMP] Not a BMP file: '%s'\n", path.string().c_str());
        fclose(f); return false;
    }
    if (ih.biCompression != 0) {
        fprintf(stderr, "[loadBMP] Compressed BMP not supported: '%s'\n", path.string().c_str());
        fclose(f); return false;
    }
    if (ih.biBitCount != 24 && ih.biBitCount != 32) {
        fprintf(stderr, "[loadBMP] Only 24/32-bit BMP supported (got %d-bit): '%s'\n",
            (int)ih.biBitCount, path.string().c_str());
        fclose(f); return false;
    }

    const int bpp = ih.biBitCount / 8;
    const int w = ih.biWidth;
    const int h = abs(ih.biHeight);
    const bool topDown = (ih.biHeight < 0);
    const int rowStride = (w * bpp + 3) & ~3;

    img.width = w;
    img.height = h;
    img.channels = bpp;
    img.data.resize((size_t)w * h * bpp);

    std::vector<uint8_t> rowBuf(rowStride);
    fseek(f, (long)fh.bfOffBits, SEEK_SET);

    for (int row = 0; row < h; ++row) {
        if (fread(rowBuf.data(), 1, (size_t)rowStride, f) != (size_t)rowStride) {
            fprintf(stderr, "[loadBMP] Unexpected EOF in pixel data: '%s'\n",
                path.string().c_str());
            fclose(f); return false;
        }
        const int dstRow = topDown ? row : (h - 1 - row);
        memcpy(img.data.data() + (size_t)dstRow * w * bpp,
            rowBuf.data(), (size_t)w * bpp);
    }

    fclose(f);
    return true;
}

// 
// saveBMP
//

inline bool saveBMP(const fs::path& path, const Image& img) {
    const int bpp = img.channels;
    const int w = img.width;
    const int h = img.height;
    const int rowStride = (w * bpp + 3) & ~3;
    const uint32_t pixelDataSize = (uint32_t)rowStride * h;

    BMPFileHeader fh{};
    fh.bfType = 0x4D42;
    fh.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    fh.bfSize = fh.bfOffBits + pixelDataSize;

    BMPInfoHeader ih{};
    ih.biSize = sizeof(BMPInfoHeader);
    ih.biWidth = w;
    ih.biHeight = h;
    ih.biPlanes = 1;
    ih.biBitCount = (uint16_t)(bpp * 8);
    ih.biCompression = 0;
    ih.biSizeImage = pixelDataSize;

    FILE* f = fopen(path.string().c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[saveBMP] Cannot write '%s'\n", path.string().c_str());
        return false;
    }

    fwrite(&fh, sizeof(fh), 1, f);
    fwrite(&ih, sizeof(ih), 1, f);

    std::vector<uint8_t> rowBuf(rowStride, 0);
    for (int row = 0; row < h; ++row) {
        const int srcRow = h - 1 - row;
        memcpy(rowBuf.data(),
            img.data.data() + (size_t)srcRow * w * bpp,
            (size_t)w * bpp);
        fwrite(rowBuf.data(), 1, (size_t)rowStride, f);
    }

    fclose(f);
    return true;
}



// JSON configuration parser
/*
Supports the subset of JSON used by configurations.json:
   - Nested objects  { "key": { ... } }
   - String values   "key": "value"
   - String arrays   "key": ["a", "b", "c"]
*/

// A section is a flat map of  key -> string   plus   key -> string-array.
struct JsonSection {
    std::map<std::string, std::string>              strings;
    std::map<std::string, std::vector<std::string>> arrays;
};

struct JsonConfig {
    std::map<std::string, JsonSection> sections;

    // Return a string value, or fallback if the section/key doesn't exist.
    std::string getString(const std::string& section,
        const std::string& key,
        const std::string& fallback = "") const
    {
        auto si = sections.find(section);
        if (si == sections.end()) return fallback;
        auto ki = si->second.strings.find(key);
        return (ki != si->second.strings.end()) ? ki->second : fallback;
    }

    // Return a string array, or empty vector if not found.
    std::vector<std::string> getStringArray(const std::string& section,
        const std::string& key) const
    {
        auto si = sections.find(section);
        if (si == sections.end()) return {};
        auto ki = si->second.arrays.find(key);
        return (ki != si->second.arrays.end()) ? ki->second : std::vector<std::string>{};
    }

    // Check whether a section exists.
    bool hasSection(const std::string& section) const {
        return sections.find(section) != sections.end();
    }
};


namespace json_detail {

    inline bool skipWS(const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
        return pos < s.size();
    }

    inline bool parseString(const std::string& s, size_t& pos, std::string& out) {
        if (pos >= s.size() || s[pos] != '"') return false;
        ++pos;
        out.clear();
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return true;
            if (c == '\\' && pos < s.size()) {
                char esc = s[pos++];
                switch (esc) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                default:   out += esc;  break;
                }
            }
            else {
                out += c;
            }
        }
        return false;
    }

    inline bool parseStringArray(const std::string& s, size_t& pos,
        std::vector<std::string>& out)
    {
        if (pos >= s.size() || s[pos] != '[') return false;
        ++pos;
        out.clear();
        skipWS(s, pos);
        if (pos < s.size() && s[pos] == ']') { ++pos; return true; }

        while (true) {
            skipWS(s, pos);
            std::string val;
            if (!parseString(s, pos, val)) return false;
            out.push_back(std::move(val));
            skipWS(s, pos);
            if (pos >= s.size()) return false;
            if (s[pos] == ']') { ++pos; return true; }
            if (s[pos] == ',') { ++pos; continue; }
            return false;
        }
    }

} 

// JSON loader

inline bool loadJsonConfig(const fs::path& path, JsonConfig& cfg) {
    std::ifstream ifs(path.string());
    if (!ifs.is_open()) {
        fprintf(stderr, "[loadJsonConfig] Cannot open '%s'\n", path.string().c_str());
        return false;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string json = oss.str();
    size_t pos = 0;

    using namespace json_detail;

    skipWS(json, pos);
    if (pos >= json.size() || json[pos] != '{') {
        fprintf(stderr, "[loadJsonConfig] Expected '{' at start of file\n");
        return false;
    }
    ++pos;

    while (true) {
        skipWS(json, pos);
        if (pos >= json.size()) {
            fprintf(stderr, "[loadJsonConfig] Unexpected EOF\n");
            return false;
        }
        if (json[pos] == '}') { ++pos; break; }
        if (json[pos] == ',') { ++pos; continue; }

        std::string sectionName;
        if (!parseString(json, pos, sectionName)) {
            fprintf(stderr, "[loadJsonConfig] Expected section name string\n");
            return false;
        }

        skipWS(json, pos);
        if (pos >= json.size() || json[pos] != ':') {
            fprintf(stderr, "[loadJsonConfig] Expected ':' after section '%s'\n",
                sectionName.c_str());
            return false;
        }
        ++pos;

        skipWS(json, pos);
        if (pos >= json.size() || json[pos] != '{') {
            fprintf(stderr, "[loadJsonConfig] Expected '{' for section '%s'\n",
                sectionName.c_str());
            return false;
        }
        ++pos;

        JsonSection& sec = cfg.sections[sectionName];

        while (true) {
            skipWS(json, pos);
            if (pos >= json.size()) {
                fprintf(stderr, "[loadJsonConfig] Unexpected EOF in section '%s'\n",
                    sectionName.c_str());
                return false;
            }
            if (json[pos] == '}') { ++pos; break; }
            if (json[pos] == ',') { ++pos; continue; }

            std::string key;
            if (!parseString(json, pos, key)) {
                fprintf(stderr, "[loadJsonConfig] Expected key in section '%s'\n",
                    sectionName.c_str());
                return false;
            }

            skipWS(json, pos);
            if (pos >= json.size() || json[pos] != ':') {
                fprintf(stderr, "[loadJsonConfig] Expected ':' after key '%s'\n",
                    key.c_str());
                return false;
            }
            ++pos;
            skipWS(json, pos);

            if (pos >= json.size()) {
                fprintf(stderr, "[loadJsonConfig] Unexpected EOF after key '%s'\n",
                    key.c_str());
                return false;
            }

            if (json[pos] == '"') {
                std::string val;
                if (!parseString(json, pos, val)) {
                    fprintf(stderr, "[loadJsonConfig] Bad string value for key '%s'\n",
                        key.c_str());
                    return false;
                }
                sec.strings[key] = std::move(val);
            }
            else if (json[pos] == '[') {
                std::vector<std::string> arr;
                if (!parseStringArray(json, pos, arr)) {
                    fprintf(stderr, "[loadJsonConfig] Bad array for key '%s'\n",
                        key.c_str());
                    return false;
                }
                sec.arrays[key] = std::move(arr);
            }
            else {
                fprintf(stderr,
                    "[loadJsonConfig] Unexpected token for key '%s' in section '%s'\n",
                    key.c_str(), sectionName.c_str());
                return false;
            }
        }
    }

    return true;
}



// loadImagePaths – load input/output folders and file list from JSON config
/*
Reads a JSON config file, extracts the base_path from an input section,
the file list (string array) from that same section, and the base_path
from an output section.  Creates the output directory if it doesn't exist.

Parameters:
   configFile    – path to the JSON file  (e.g. "configurations.json")
   inputSection  – JSON section for the input folder  (e.g. "testing_image_original")
   fileListKey   – key within inputSection for the filename array (e.g. "original_images")
   outputSection – JSON section for the output folder (e.g. "testing_image_noisy")
   inputDir      – [out] resolved input folder path
   imageNames    – [out] list of filenames to process
   outputDir     – [out] resolved output folder path (created if absent)

Returns true on success.  On failure, prints an error to stderr and
returns false — the caller can then exit or handle the error.
*/

inline bool loadImagePaths(const std::string& configFile,
    const std::string& inputSection,
    const std::string& fileListKey,
    const std::string& outputSection,
    std::string& inputDir,
    std::vector<std::string>& imageNames,
    std::string& outputDir)
{
    // Load JSON configuration
    JsonConfig cfg;
    if (!loadJsonConfig(configFile, cfg)) {
        fprintf(stderr, "Error: Failed to load configuration from '%s'\n",
            configFile.c_str());
        return false;
    }

    // Resolve input folder
    inputDir = cfg.getString(inputSection, "base_path");
    if (inputDir.empty()) {
        fprintf(stderr, "Error: '%s' / 'base_path' not found in '%s'\n",
            inputSection.c_str(), configFile.c_str());
        return false;
    }

    // Resolve file list
    imageNames = cfg.getStringArray(inputSection, fileListKey);
    if (imageNames.empty()) {
        fprintf(stderr, "Error: '%s' / '%s' list is empty or missing in '%s'\n",
            inputSection.c_str(), fileListKey.c_str(), configFile.c_str());
        return false;
    }

    // Resolve output folder
    outputDir = cfg.getString(outputSection, "base_path");
    if (outputDir.empty()) {
        fprintf(stderr, "Error: '%s' / 'base_path' not found in '%s'\n",
            outputSection.c_str(), configFile.c_str());
        return false;
    }

    // Create the output directory tree if it doesn't exist (cross-platform)
    std::error_code ec;
    fs::create_directories(fs::path(outputDir), ec);
    if (ec) {
        fprintf(stderr, "Cannot create output directory '%s': %s\n",
            outputDir.c_str(), ec.message().c_str());
        return false;
    }

    printf("\nInput folder : %s\n", inputDir.c_str());
    printf("Output folder: %s\n", outputDir.c_str());
    printf("Images to process: %zu\n", imageNames.size());

    return true;
}
