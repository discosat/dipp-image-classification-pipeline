#include "module.h"
#include "utils/util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

bool try_parse_metadata(const char* image_file, uint32_t* width, uint32_t* height, uint32_t* bits_per_pixel, uint32_t* channels, char* bayer_pattern) {
    char metadata_file[1024];
    snprintf(metadata_file, sizeof(metadata_file), "%s.metadata", image_file);
    
    FILE* meta_fh = fopen(metadata_file, "rb");
    if (!meta_fh) {
        return false;
    }
    
    fseek(meta_fh, 0, SEEK_END);
    long meta_size = ftell(meta_fh);
    fseek(meta_fh, 0, SEEK_SET);
    
    if (meta_size <= 0) {
        fclose(meta_fh);
        return false;
    }
    
    uint8_t* meta_data = (uint8_t*)malloc(meta_size);
    if (!meta_data) {
        fclose(meta_fh);
        return false;
    }
    
    size_t meta_bytes_read = fread(meta_data, 1, meta_size, meta_fh);
    fclose(meta_fh);
    
    if (meta_bytes_read != meta_size) {
        free(meta_data);
        return false;
    }
    Metadata* metadata = metadata__unpack(NULL, meta_size, meta_data);
    free(meta_data);
    
    if (!metadata) {
        return false;
    }
    
    *width = metadata->width;
    *height = metadata->height;
    *channels = metadata->channels;
    *bits_per_pixel = metadata->bits_pixel;
    
    strcpy(bayer_pattern, "GRBG");
    for (size_t i = 0; i < metadata->n_items; i++) {
        if (metadata->items[i]->key && strcmp(metadata->items[i]->key, "bayer_pattern") == 0) {
            if (metadata->items[i]->value_case == METADATA_ITEM__VALUE_STRING_VALUE) {
                strcpy(bayer_pattern, metadata->items[i]->string_value);
                break;
            }
        }
    }
    
    metadata__free_unpacked(metadata, NULL);
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_raw_image_file> [output_dir]\n", argv[0]);
        printf("Example: %s input.raw /tmp/output/\n", argv[0]);
        printf("Note: If <input_raw_image_file>.metadata exists, it will be used for image parameters\n");
        return -1;
    }

    const char* input_file = argv[1];
    const char* output_dir = (argc > 2) ? argv[2] : "./";

    uint32_t image_width, image_height, bits_per_pixel, image_channels;
    char bayer_pattern[16];
    bool has_metadata = try_parse_metadata(input_file, &image_width, &image_height, &bits_per_pixel, &image_channels, bayer_pattern);
    FILE *fh = fopen(input_file, "rb");
    if (!fh) {
        printf("Error: Could not open file %s\n", input_file);
        return -1;
    }

    fseek(fh, 0, SEEK_END);
    long fsize = ftell(fh);
    fseek(fh, 0, SEEK_SET);

    printf("Input file: %s, size: %ld bytes\n", input_file, fsize);

    if (!has_metadata) {
        printf("No metadata file found, using fallback detection\n");
        // Use full resolution 
        image_width = 2464;
        image_height = 2056;
        bits_per_pixel = 12;
        image_channels = 1;
        strcpy(bayer_pattern, "GRBG");
        
        printf("Using full camera resolution: %dx%d\n", image_width, image_height);
    } else {
        printf("Using metadata: %dx%d, %d bpp, %d channels, Bayer: %s\n", 
               image_width, image_height, bits_per_pixel, image_channels, bayer_pattern);
    }

    unsigned char *image_data = (unsigned char*)malloc(fsize);
    if (!image_data) {
        printf("Error: Failed to allocate memory\n");
        fclose(fh);
        return -1;
    }

    size_t bytes_read = fread(image_data, 1, fsize, fh);
    fclose(fh);

    if (bytes_read != fsize) {
        printf("Error: Could not read entire file\n");
        free(image_data);
        return -1;
    }

    size_t bytes_per_pixel = (bits_per_pixel <= 8) ? 1 : 2;
    size_t expected_image_bytes = image_width * image_height * bytes_per_pixel;
    printf("Buffer size: %ld bytes, Expected: %ld bytes for %dx%d image\n", 
           fsize, expected_image_bytes, image_width, image_height);
    cv::Mat rawImage;
    if (bits_per_pixel <= 8) {
        rawImage = cv::Mat(image_height, image_width, CV_8UC1, image_data);
    } else {
        // For 12-bit data, create as 16-bit and mask to 12-bit values
        rawImage = cv::Mat(image_height, image_width, CV_16UC1, image_data);
        if (bits_per_pixel == 12) {
            // Mask to 12-bit values (& 0x0FFF)
            uint16_t* ptr = (uint16_t*)rawImage.data;
            size_t total_pixels = image_width * image_height;
            for (size_t i = 0; i < total_pixels; i++) {
                ptr[i] = ptr[i] & 0x0FFF;
            }
        }
    }
    
    printf("Raw image stats - Min/Max values before demosaic:\n");
    double minVal, maxVal;
    cv::minMaxLoc(rawImage, &minVal, &maxVal);
    printf("Min: %.0f, Max: %.0f\n", minVal, maxVal);

    cv::Mat rawImageContiguous;
    rawImage.copyTo(rawImageContiguous);

    free(image_data);

    // Convert to 8-bit for demosaicing (scale from 12-bit to 8-bit using bit shift)
    cv::Mat rawImage8bit;
    if (bits_per_pixel <= 8) {
        rawImageContiguous.copyTo(rawImage8bit);
    } else if (bits_per_pixel == 12) {
        // Use bit shift (>>4) to convert 12-bit to 8-bit, matching Python implementation
        rawImage8bit = cv::Mat(image_height, image_width, CV_8UC1);
        uint16_t* src = (uint16_t*)rawImageContiguous.data;
        uint8_t* dst = (uint8_t*)rawImage8bit.data;
        size_t total_pixels = image_width * image_height;
        for (size_t i = 0; i < total_pixels; i++) {
            dst[i] = (uint8_t)(src[i] >> 4);
        }
    } else {
        // For other bit depths, use normalization
        cv::normalize(rawImageContiguous, rawImage8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
    
    // Apply vertical flip to match camera orientation (cameras often capture upside down)
    cv::Mat flippedImage;
    cv::flip(rawImage8bit, flippedImage, 0);  // 0 means vertical flip
    
    cv::Mat demosaicedImage;

    int cv_color_code;
    // Note: Using BGR output to match Python implementation
    if (strcmp(bayer_pattern, "RGGB") == 0) {
        cv_color_code = cv::COLOR_BayerRG2BGR;
    } else if (strcmp(bayer_pattern, "GRBG") == 0) {
        cv_color_code = cv::COLOR_BayerGR2BGR;
    } else if (strcmp(bayer_pattern, "GBRG") == 0) {
        cv_color_code = cv::COLOR_BayerGB2BGR;
    } else if (strcmp(bayer_pattern, "BGGR") == 0) {
        cv_color_code = cv::COLOR_BayerBG2BGR;
    } else {
        printf("Warning: Unknown Bayer pattern '%s', defaulting to GRBG\n", bayer_pattern);
        cv_color_code = cv::COLOR_BayerGR2BGR;
    }

    printf("Performing demosaicing (Bayer pattern: %s) with vertical flip...\n", bayer_pattern);
    cv::cvtColor(flippedImage, demosaicedImage, cv_color_code);

    // Demosaicing output is already 8-bit BGR, no normalization needed
    cv::Mat demosaicedImage_8bit = demosaicedImage;

    std::filesystem::path input_path(input_file);
    std::string base_name = input_path.stem().string();
    std::string output_file = std::string(output_dir) + "/" + base_name + "_demosaiced.png";
    
    printf("Saving result to: %s\n", output_file.c_str());
    if (!cv::imwrite(output_file, demosaicedImage_8bit)) {
        printf("Error: Could not save output image\n");
        return -1;
    }

    printf("Demosaic test completed successfully!\n");
    printf("Input: %dx%d (%d-bit %s), Output: %dx%d (8-bit BGR), channels: %d\n", 
           image_width, image_height, bits_per_pixel, bayer_pattern,
           demosaicedImage_8bit.cols, demosaicedImage_8bit.rows, demosaicedImage_8bit.channels());

    return 0;
}
