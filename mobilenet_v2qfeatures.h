#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <regex>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <opencv2/opencv.hpp>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

/***
 * Quantised MobileNetV2 features with pre-trained weights with Executorch
 * (c) 2025-2026 Bernd Porr, BSD3
 ***/

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif

/**
 * @brief Implementation of MobileNetV2 features with pre-trained quanitised weights
 * 
 * It loads the network with the pre-trained weights from "mobilenet_features_quant.pte".
 */
class MobileNetV2qFeatures : public torch::nn::Module
{
public:
/**
 * @brief Construct a new Mobile Net V2q Features object
 * 
 * @param features_pte_filename The file which contains the quantised model of the feature det.
 */
    MobileNetV2qFeatures(std::string features_pte_filename = "mobilenet_features_quant.pte")
    {
        quantFeatures = std::make_shared<executorch::extension::Module>(features_pte_filename);
        const auto result = quantFeatures->load_forward();
        if (executorch::runtime::Error::Ok != result) {
            std::cerr << "Could not load forward(): " << (int)result << std::endl;
        }

        const auto method_meta = quantFeatures->method_meta("forward");

        if (debugOutput && method_meta.ok())
        {
            std::cerr << "Num of inputs: " << (int)(method_meta->num_inputs()) << std::endl;
            const auto input_meta = method_meta->input_tensor_meta(0);
            if (input_meta.ok())
            {
                std::cerr << "Input Scalar type (6): " << (int)(input_meta->scalar_type()) << std::endl;
                std::cerr << "Sizes: ";
                for(auto &s : input_meta->sizes()) std::cerr << s << " ";
                std::cerr << std::endl;
            }

            std::cerr << "Num of outputs: " << (int)(method_meta->num_outputs()) << std::endl;
            const auto output_meta = method_meta->output_tensor_meta(0);
            if (output_meta.ok())
            {
                std::cerr << "Output Scalar type (6): " << (int)(output_meta->scalar_type()) << std::endl;
                std::cerr << "Sizes: ";
                for(auto &s : output_meta->sizes()) std::cerr << s << " ";
                std::cerr << std::endl;
            }
        }

        executorch::runtime::Error error = quantFeatures->load();
        if (!(quantFeatures->is_loaded()))
        {
            std::cerr << "Err:" << (int)error << executorch::runtime::to_string(error) << std::endl;
            throw error;
        }
    }

    /**
     * @brief Performs the forward pass.
     *
     * @param x The batch of input images.
     * @return torch::Tensor The category scores for the different labels.
     */
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.contiguous().cpu();
        std::vector<int> insizes(
            x.sizes().begin(),
            x.sizes().end());
        auto et_tensor = executorch::extension::from_blob(
            x.data_ptr<float>(),
            insizes);
        auto result = quantFeatures->forward(et_tensor);
        if (!result.ok())
        {
            std::cerr << "Fatal. No result from model: ";
            std::cerr << (int)(result.error()) << ", " << executorch::runtime::to_string(result.error()) << std::endl;
            throw result.error();
        }
        const auto et = result->at(0).toTensor();
        std::vector<long int> outsizes(
            et.sizes().begin(),
            et.sizes().end());
        x = torch::from_blob(
            et.mutable_data_ptr<float>(),
            outsizes,
            torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(torch::kCPU));
        const torch::nn::functional::AdaptiveAvgPool2dFuncOptions &ar = torch::nn::functional::AdaptiveAvgPool2dFuncOptions({1, 1});
        x = torch::nn::functional::adaptive_avg_pool2d(x, ar);
        x = torch::flatten(x, 1);
        return x;
    }

    /**
     * @brief Preprocessing of an openCV image for inference or learning.
     * The images are resized to 256x256, followed by a central crop of 224x224.
     * Finally the values are first rescaled to [0.0, 1.0]
     * and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
     *
     * @param img 8bit BGR openCV image with an aspect ratio of 1:1.
     * @param resizeOnly If true the image is only resized to 224x224 but not cropped. Default: false.
     * @return torch::Tensor The image as a tensor ready to be used for inference and learning.
     */
    static torch::Tensor preprocess(cv::Mat img, bool resizeOnly = false)
    {
        constexpr int imageSizeBeforeCrop = 256;
        constexpr int finalImageSize = 224;
        constexpr int numChannels = 3; // colour

        if (img.depth() != CV_8U)
            throw std::invalid_argument("Image is not 8bit.");
        if (img.channels() != numChannels)
            throw std::invalid_argument("Image is not BGR / colour.");

        if (resizeOnly)
        {
            cv::resize(img, img, cv::Size(finalImageSize, finalImageSize));
        }
        else
        {
            cv::resize(img, img, cv::Size(imageSizeBeforeCrop, imageSizeBeforeCrop));
            constexpr int start = (imageSizeBeforeCrop - finalImageSize) / 2;
            const cv::Rect roi(start, start, finalImageSize, finalImageSize);
            img = img(roi).clone();
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
        tensor = tensor.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);
        tensor = torch::data::transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})(tensor);
        return tensor;
    }

    /**
     * @brief Gets the number of features.
     *
     * @return The number of features.
     */
    int getNfeatures() const
    {
        return features_output_channels;
    }

private:
    // Features output channels
    const int features_output_channels = 1280;
    // the module with all the inverted residuals
    std::shared_ptr<executorch::extension::Module> quantFeatures;
};
