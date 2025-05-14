#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

using namespace std;

uint32_t read_uint32(ifstream& file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), 4);
    return __builtin_bswap32(value); 
}

vector<uint8_t> read_mnist_labels(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Failed to open label file: " << filename << "\n";
        return {};
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = __builtin_bswap32(magic);

    if (magic != 0x00000801) {
        cerr << "Invalid magic number in label file: " << hex << magic << dec << "\n";
        return {};
    }

    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = __builtin_bswap32(num_labels);

    vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

int main() {
    ifstream file("mnist/t10k-images.idx3-ubyte", ios::binary);
    if (!file) {
        cerr << "Failed to open file.\n";
        return 1;
    }

    uint32_t magic = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t num_rows = read_uint32(file);
    uint32_t num_cols = read_uint32(file);

    cout << "Magic: " << hex << magic << dec << "\n";
    cout << "Images: " << num_images << ", Size: " << num_rows << "x" << num_cols << "\n";
    vector<uint8_t> labels = read_mnist_labels("mnist/t10k-labels.idx1-ubyte");

// Show label for first image
    
    for (int img_idx = 0; img_idx < 10; ++img_idx) {
        vector<uint8_t> image(num_rows * num_cols);
        file.read(reinterpret_cast<char*>(image.data()), image.size());

        cv::Mat img(num_rows, num_cols, CV_8UC1, image.data());
        cv::Mat enlarged;
        cv::resize(img, enlarged, cv::Size(), 10, 10, cv::INTER_NEAREST);
        cout << "Label for image " << img_idx << ": " << (int)labels[img_idx] << endl;
        string window_name = "Image " + to_string(img_idx);
        cv::imshow(window_name, enlarged);
        cv::waitKey(0);  // Wait for key press
        cv::destroyWindow(window_name);  // Close before showing the next
    }

    return 0;
}