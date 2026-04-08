/* 
 helper_functions.h — Image I/O Utilities 
 
 Provide functions for writing fractal pixel data to image files.
 All functions expect RGBA pixel buffers (4 bytes per pixel).
 */

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdint>


namespace {

// CRC-32 Lookup Table and Function

class CRC32Table {
    uint32_t table_[256];

public:
    CRC32Table()
    {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                if (c & 1)
                    c = 0xEDB88320u ^ (c >> 1);
                else
                    c = c >> 1;
            }
            table_[n] = c;
        }
    }

    uint32_t compute(const unsigned char *data, size_t length) const
    {
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < length; ++i) {
            crc = table_[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFFu;
    }
};

// Single global instance, constructed once
static const CRC32Table g_crc32;


// Adler-32 Checksum
uint32_t adler32(const unsigned char *data, size_t length)
{
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;

    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    return (b << 16) | a;
}


/* Write a 32-bit value in big-endian byte order (PNG uses network order) */
void push_be32(std::vector<unsigned char> &out, uint32_t value)
{
    out.push_back(static_cast<unsigned char>((value >> 24) & 0xFF));
    out.push_back(static_cast<unsigned char>((value >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((value >>  8) & 0xFF));
    out.push_back(static_cast<unsigned char>((value >>  0) & 0xFF));
}

void write_be32(std::ofstream &out, uint32_t value)
{
    unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24) & 0xFF),
        static_cast<unsigned char>((value >> 16) & 0xFF),
        static_cast<unsigned char>((value >>  8) & 0xFF),
        static_cast<unsigned char>((value >>  0) & 0xFF)
    };
    out.write(reinterpret_cast<const char *>(bytes), 4);
}


// Write a complete PNG chunk to file
void write_png_chunk(std::ofstream &out,
                     const char *type,
                     const unsigned char *data,
                     size_t data_len)
{
    // Length
    write_be32(out, static_cast<uint32_t>(data_len));

    // Type (4 ASCII bytes)
    out.write(type, 4);

    // Data
    if (data_len > 0) {
        out.write(reinterpret_cast<const char *>(data), data_len);
    }

    // CRC-32 over type + data
    std::vector<unsigned char> crc_input;
    crc_input.insert(crc_input.end(),
                     reinterpret_cast<const unsigned char *>(type),
                     reinterpret_cast<const unsigned char *>(type) + 4);
    if (data_len > 0) {
        crc_input.insert(crc_input.end(), data, data + data_len);
    }
    write_be32(out, g_crc32.compute(crc_input.data(), crc_input.size()));
}



std::vector<unsigned char> build_zlib_stored(
    const unsigned char *data, size_t data_len)
{
    std::vector<unsigned char> result;

    // Pre-allocate: 2 (zlib hdr) + data + overhead per block + 4 (adler)
    size_t num_blocks = (data_len + 65534) / 65535;
    result.reserve(2 + data_len + num_blocks * 5 + 4);

    // zlib header: CMF=0x78, FLG=0x01
    result.push_back(0x78);
    result.push_back(0x01);

    // DEFLATE stored blocks (max 65535 bytes each)
    size_t offset = 0;
    while (offset < data_len) {
        size_t remaining = data_len - offset;
        uint16_t block_len = (remaining > 65535) ? 65535
                           : static_cast<uint16_t>(remaining);
        bool is_last = (offset + block_len >= data_len);

        // Block header byte: BFINAL | BTYPE(00)
        result.push_back(is_last ? 0x01 : 0x00);

        // LEN (little-endian)
        result.push_back(static_cast<unsigned char>(block_len & 0xFF));
        result.push_back(static_cast<unsigned char>((block_len >> 8) & 0xFF));

        // NLEN (one's complement of LEN, little-endian)
        uint16_t nlen = ~block_len;
        result.push_back(static_cast<unsigned char>(nlen & 0xFF));
        result.push_back(static_cast<unsigned char>((nlen >> 8) & 0xFF));

        // Raw block data
        result.insert(result.end(),
                      data + offset,
                      data + offset + block_len);

        offset += block_len;
    }

    // Handle empty input (must still have one stored block)
    if (data_len == 0) {
        result.push_back(0x01);  // BFINAL=1, BTYPE=00
        result.push_back(0x00);  // LEN=0
        result.push_back(0x00);
        result.push_back(0xFF);  // NLEN=0xFFFF
        result.push_back(0xFF);
    }

    // Adler-32 checksum of the uncompressed data (big-endian)
    uint32_t adler = adler32(data, data_len);
    push_be32(result, adler);

    return result;
}

} 


// Save image to PNG file
inline void write_png(const std::string &filename,
                      const std::vector<unsigned char> &rgba,
                      int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename + " for writing");
    }

    static const unsigned char png_sig[8] = {
        0x89, 0x50, 0x4E, 0x47,   // .PNG
        0x0D, 0x0A, 0x1A, 0x0A    // \r\n\x1a\n
    };
    file.write(reinterpret_cast<const char *>(png_sig), 8);

    // Create header
    {
        unsigned char ihdr[13];
        ihdr[0]  = (width  >> 24) & 0xFF;
        ihdr[1]  = (width  >> 16) & 0xFF;
        ihdr[2]  = (width  >>  8) & 0xFF;
        ihdr[3]  = (width  >>  0) & 0xFF;
        ihdr[4]  = (height >> 24) & 0xFF;
        ihdr[5]  = (height >> 16) & 0xFF;
        ihdr[6]  = (height >>  8) & 0xFF;
        ihdr[7]  = (height >>  0) & 0xFF;
        ihdr[8]  = 8;   // bit depth
        ihdr[9]  = 6;   // color type: RGBA
        ihdr[10] = 0;   // compression
        ihdr[11] = 0;   // filter
        ihdr[12] = 0;   // interlace

        write_png_chunk(file, "IHDR", ihdr, 13);
    }

    {
        size_t row_bytes = static_cast<size_t>(width) * 4;
        size_t raw_size  = static_cast<size_t>(height) * (1 + row_bytes);

        // Build filtered scanlines
        std::vector<unsigned char> raw_data;
        raw_data.reserve(raw_size);

        for (int y = 0; y < height; ++y) {
            raw_data.push_back(0x00);  // Filter byte: None

            const unsigned char *row = &rgba[y * row_bytes];
            raw_data.insert(raw_data.end(), row, row + row_bytes);
        }

        // Wrap in zlib stored-block format
        std::vector<unsigned char> idat_data =
            build_zlib_stored(raw_data.data(), raw_data.size());

        write_png_chunk(file, "IDAT",
                        idat_data.data(), idat_data.size());
    }

    // Save chunks to file
    write_png_chunk(file, "IEND", nullptr, 0);

    std::cout << "Saved: " << filename
              << " (" << width << "x" << height << " PNG)\n";
}


#endif 
