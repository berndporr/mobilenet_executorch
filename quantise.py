import kagglehub
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
pte_filename_features = "mobilenet_features_quant.pte"

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

def load_model_features():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    model.to("cpu")
    return model

def load_model_classifier():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).classifier
    model.to("cpu")
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loader(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Standard for CNNs
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],     # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Datasets
    dataset = datasets.ImageFolder(
        root=data_path+"/train/",
        transform=transform
    )

    sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=sampler)

    return data_loader

def createFeaturesFile():
    data_path = kagglehub.dataset_download("abdalnassir/the-animalist-cat-vs-dog-classification")+"/Cat vs Dog"
    print("File are here: ",data_path)

    float_model = load_model_features().to("cpu")
    float_model.eval()

    # create another instance of the model since
    # we need to keep the original model around
    model_to_quantize = load_model_features().to("cpu")
    model_to_quantize.eval()

    sample_inputs = (torch.randn(1, 3, 224, 224), )
    exported_model = torch.export.export(model_to_quantize, sample_inputs).module()
    quantizer = XNNPACKQuantizer()
    prepared_model = prepare_pt2e(exported_model, quantizer)
    data_loader = prepare_data_loader(data_path)
    with torch.no_grad():
        for image, _ in data_loader:
            prepared_model(image)
    quantized_model = convert_pt2e(prepared_model)

    # Baseline model size
    print("Size of baseline model")
    print_size_of_model(float_model)

    # Quantized model size
    print("Size of model after quantization")
    # export again to remove unused weights
    quantized_model = torch.export.export(quantized_model, sample_inputs).module()
    print_size_of_model(quantized_model)

    # capture the model to get an ExportedProgram
    quantized_ep = torch.export.export(quantized_model, sample_inputs)

    # Optimize for target hardware (switch backends with one line)
    program = to_edge_transform_and_lower(
        quantized_ep,
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    # 3. Save for deployment
    with open(pte_filename_features, "wb") as f:
        f.write(program.buffer)

    # Test locally via ExecuTorch runtime's pybind API (optional)
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    method = runtime.load_program(pte_filename_features).load_method("forward")
    outputs = method.execute([torch.randn(1, 3, 224, 224)])
    print(outputs)


createFeaturesFile()
