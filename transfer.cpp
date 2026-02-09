#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "mobilenet_v2qfeatures.h"
#include <iostream>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

// Path of the Kaggle dataset
const fs::path datasetpath = ".cache/kagglehub/datasets/abdalnassir/the-animalist-cat-vs-dog-classification/versions/1/Cat vs Dog/train/";

// Path to the loss log file
const char loss_file[] = "loss.dat";

// Subdirs of the two classes
const std::vector<fs::path> classes = {"Cat", "Dog"};

// The batch size for training
const int batch_size = 32;

// The number of epochs
const int epochs = 50;

// -------------------------
// Dataset implementation
// -------------------------
struct ImageFolderDataset : torch::data::Dataset<ImageFolderDataset>
{
    struct Sample
    {
        fs::path image_path;
        int label;
    };

    std::vector<Sample> samples;

    ImageFolderDataset(const fs::path &root, const std::vector<fs::path> &classes)
    {
        for (size_t label = 0; label < classes.size(); label++)
        {
            const fs::path class_path = root / classes[label];
            for (const auto &p : fs::directory_iterator(class_path))
            {
                if (p.is_regular_file())
                {
                    samples.push_back({p.path(), (int)label});
                }
            }
        }
        std::cout << "Loaded " << samples.size() << " samples from " << datasetpath.string() << "\n";
    }

    torch::data::Example<> get(size_t idx) override
    {
        const auto &sample = samples[idx];
        const cv::Mat img = cv::imread(sample.image_path.string());
        if (img.empty())
        {
            throw std::runtime_error("Failed to load image: " + sample.image_path.string());
        }
        const torch::Tensor data = MobileNetV2qFeatures::preprocess(img);
        const torch::Tensor label = torch::tensor(sample.label, torch::kLong);
        return {data, label};
    }

    torch::optional<size_t> size() const override
    {
        return samples.size();
    }
};

// Simple progress output on the same line. No new line.
void progress(int epoch, int epochs, double loss, float f)
{
    std::cout << "Epoch [" << epoch << "/" << epochs << "], Loss: "
              << loss << "\t" << f << "Hz" << "\r" << std::flush;
}

// Classifier for nClasses
struct MobileNetV2classifier : torch::nn::Module
{
    const char *classifierModuleName = "classifer";
    MobileNetV2classifier(int nFeatures, int nClasses)
    {
        sequ = torch::nn::Sequential(
            torch::nn::Dropout(0.2),
            torch::nn::Linear(nFeatures, nClasses));

        register_module(classifierModuleName, sequ);
        for (auto &module : sequ->modules(/*include_self=*/false))
        {
            if (auto M = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
            {
                torch::nn::init::normal_(M->weight, 0.0, 0.01);
                torch::nn::init::zeros_(M->bias);
            }
        }
    }
    torch::nn::Sequential sequ{nullptr};
};

// -------------------------
// Main training program
// -------------------------
int main()
{
    torch::manual_seed(42);
    torch::Device device(torch::kCPU);

    const fs::path homedir(getpwuid(getuid())->pw_dir);
    ImageFolderDataset ds(homedir / datasetpath, classes);

    // Creates a DataLoader instance for a stateless dataset.
    // The sampler is RandomSampler so shuffling is enabled.
    auto loader = torch::data::make_data_loader(
        ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // Model setup
    MobileNetV2qFeatures features;
    MobileNetV2classifier classifier(features.getNfeatures(), classes.size());

    // Optimizer only for classifier.
    torch::optim::Adam optimizer(classifier.sequ->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;

    // Logging of the loss
    std::fstream floss;
    floss.open(loss_file, std::fstream::out);

    float f = 0;

    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float cumloss = 0;
        classifier.train();
        int n = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (auto &batch : *loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();
            // executorch feature detector (without learning and pre-trained weights)
            auto fout = features.forward(data);
            // libtorch classifier (with learning)
            auto output = classifier.sequ->forward(fout);
            auto loss = criterion(output, target);
            loss.backward();
            optimizer.step();
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
            f = n * 1000 / (float)duration.count();
            progress(epoch, epochs, loss.item<double>(), f);
            cumloss += loss.item<double>();
            n++;
        }
        const double avgLoss = cumloss / (double)n;
        progress(epoch, epochs, avgLoss, f);
        floss << epoch << "\t" << avgLoss << std::endl;
        std::cout << std::endl
                  << std::flush;
    }
    std::cout << "Done.\n";
    return 0;
}
