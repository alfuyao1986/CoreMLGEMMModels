import torch
import numpy as np
import time
import argparse
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os

def create_mlp(D,N,NumLayers) :
    layers = nn.ModuleList()
    for i in range(NumLayers):
        layers.append(nn.Linear(D, N, bias=False))
    return torch.nn.Sequential(*layers)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

class GenerateModelError (Exception):
    pass

def generate_mlmodels(M, N, batch, Layers):

    filename1 = str(M)+"x"+str(N)+"x"+str(Layers)+"x"+str(batch)+"_FP16.mlmodel"
    filename2 = str(M)+"x"+str(N)+"x"+str(Layers)+"x"+str(batch)+"_INT8LUT.mlmodel"
    if os.path.exists(filename1) and os.path.exists(filename2):
        return
    
    model = create_mlp(M,N,Layers)
    model.apply(init_weights)
    model.eval()

    with torch.no_grad():
        x = torch.randn(batch,M)
        output = model(x)

    traced_model = torch.jit.trace(model, x)
    try:
        apple_model = ct.convert(traced_model, inputs=[ct.TensorType(name="input", shape = (batch,M) )])
        #filename = str(M)+"x"+str(N)+"x"+str(Layers)+"x"+str(batch)+".mlmodel"
        #apple_model.save(filename)
    except:
        raise GenerateModelError
        return

    try:
        apple_model_FP16 = quantization_utils.quantize_weights(apple_model, nbits=16)
    except:
        raise GenerateModelError
        return
    filename = str(M)+"x"+str(N)+"x"+str(Layers)+"x"+str(batch)+"_FP16.mlmodel"
    apple_model_FP16.save(filename)

    try:
        apple_model_INT8LUT = quantization_utils.quantize_weights(apple_model, nbits=8, quantization_mode="kmeans")
    except:
        raise GenerateModelError
        return
    filename = str(M)+"x"+str(N)+"x"+str(Layers)+"x"+str(batch)+"_INT8LUT.mlmodel"
    apple_model_INT8LUT.save(filename)

for i in range(7):
    M = 256*2**(i)
    N = M
    for l in range(5):
        if l ==0:
            layers = 1
        else:
            layers = 8*l
        try:
            generate_mlmodels(M,N,1,layers)
        except GenerateModelError:
            print("Model with dimension ", M, "x", N, " with", layers, " layers not created")
            pass