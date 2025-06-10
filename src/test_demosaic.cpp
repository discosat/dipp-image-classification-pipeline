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
        image_width = 2464;
        bits_per_pixel = 12;
        image_channels = 1;
        strcpy(bayer_pattern, "GRBG");
        
        size_t actual_data_bytes = 4794484;
        size_t bytes_per_pixel = 2;
        
        if (fsize > actual_data_bytes) {
            image_height = actual_data_bytes / (image_width * bytes_per_pixel);
            printf("Detected zero-padding in buffer, using actual image height: %d\n", image_height);
        } else {
            image_height = fsize / (image_width * bytes_per_pixel);
            printf("No padding detected, calculated height: %d\n", image_height);
        }
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
        rawImage = cv::Mat(image_height, image_width, CV_16UC1, image_data);
    }
    
    printf("Raw image stats - Min/Max values before demosaic:\n");
    double minVal, maxVal;
    cv::minMaxLoc(rawImage, &minVal, &maxVal);
    printf("Min: %.0f, Max: %.0f\n", minVal, maxVal);

    cv::Mat rawImageContiguous;
    rawImage.copyTo(rawImageContiguous);

    free(image_data);

    cv::Mat demosaicedImage;

    int cv_color_code;
    if (strcmp(bayer_pattern, "RGGB") == 0) {
        cv_color_code = cv::COLOR_BayerBG2RGB;
    } else if (strcmp(bayer_pattern, "GRBG") == 0) {
        cv_color_code = cv::COLOR_BayerGR2RGB;
    } else if (strcmp(bayer_pattern, "GBRG") == 0) {
        cv_color_code = cv::COLOR_BayerGB2RGB;
    } else if (strcmp(bayer_pattern, "BGGR") == 0) {
        cv_color_code = cv::COLOR_BayerRG2RGB;
    } else {
        printf("Warning: Unknown Bayer pattern '%s', defaulting to GRBG\n", bayer_pattern);
        cv_color_code = cv::COLOR_BayerGR2RGB;
    }

    printf("Performing demosaicing (Bayer pattern: %s)...\n", bayer_pattern);
    cv::cvtColor(rawImageContiguous, demosaicedImage, cv_color_code);

    printf("Normalizing to 8-bit...\n");
    cv::Mat demosaicedImage_8bit;
    if (bits_per_pixel <= 8) {
        demosaicedImage.copyTo(demosaicedImage_8bit);
    } else {
        cv::normalize(demosaicedImage, demosaicedImage_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        
        double minVal, maxVal;
        cv::minMaxLoc(demosaicedImage, &minVal, &maxVal);
        printf("Demosaiced image range: %.0f - %.0f, normalized to 0-255\n", minVal, maxVal);
    }

    std::filesystem::path input_path(input_file);
    std::string base_name = input_path.stem().string();
    std::string output_file = std::string(output_dir) + "/" + base_name + "_demosaiced.png";
    
    printf("Saving result to: %s\n", output_file.c_str());
    if (!cv::imwrite(output_file, demosaicedImage_8bit)) {
        printf("Error: Could not save output image\n");
        return -1;
    }

    printf("Demosaic test completed successfully!\n");
    printf("Input: %dx%d (%d-bit %s), Output: %dx%d (8-bit RGB), channels: %d\n", 
           image_width, image_height, bits_per_pixel, bayer_pattern,
           demosaicedImage_8bit.cols, demosaicedImage_8bit.rows, demosaicedImage_8bit.channels());

    return 0;
}
