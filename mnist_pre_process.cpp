#include "mnist_pre_process.h"

#include "errno.h"
#include "stdio.h"
#include "stdlib.h"

#include <cstring>

void ReadData(std::vector<std::vector<float>>& dataset,
              std::vector<float>& labels, const char* data_path,
              const char* label_path) {
    FILE* data_file = fopen(data_path, "rb");
    if (!data_file) {
        perror("Error opening data file");
        exit(errno);
    }

    FILE* label_file = fopen(label_path, "rb");
    if (!label_file) {
        perror("Error opening label file");
        fclose(data_file);
        exit(errno);
    }

    int mbs = 0, number = 0, col = 0, row = 0;
    if (fread(&mbs, 4, 1, data_file) != 1) {
        perror("Error reading data file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }
    if (fread(&number, 4, 1, data_file) != 1) {
        perror("Error reading data file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }
    if (fread(&row, 4, 1, data_file) != 1) {
        perror("Error reading data file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }
    if (fread(&col, 4, 1, data_file) != 1) {
        perror("Error reading data file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }

    RevertInt(mbs);
    RevertInt(number);
    RevertInt(row);
    RevertInt(col);

    if (fread(&mbs, 4, 1, label_file) != 1) {
        perror("Error reading label file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }
    if (fread(&number, 4, 1, label_file) != 1) {
        perror("Error reading label file");
        fclose(data_file);
        fclose(label_file);
        exit(errno);
    }
    RevertInt(mbs);
    RevertInt(number);

    dataset.resize(number, std::vector<float>(row * col));
    labels.resize(number);

    std::vector<unsigned char> buffer(row * col);
    for (int i = 0; i < number; ++i) {
        if (fread(buffer.data(), 1, row * col, data_file) != row * col) {
            perror("Error reading data file");
            fclose(data_file);
            fclose(label_file);
            exit(errno);
        }
        for(int j = 0; j < row * col; ++j) {
            dataset[i][j] = static_cast<float>(buffer[j]);
        }
        unsigned char temp;
        if (fread(&temp, 1, 1, label_file) != 1) {
            perror("Error reading label file");
            fclose(data_file);
            fclose(label_file);
            exit(errno);
        }
        labels[i] = static_cast<float>(temp);
    }

    fclose(data_file);
    fclose(label_file);
}
