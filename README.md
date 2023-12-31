# CoreMLGEMMModels
Create simple GEMM models in coreML to test achievable peak compute and bandwidth for apple's silicon (M1/M2)

## Environment Setup

```
conda create --name CoreML python=3.10
conda activate CoreML

pip install torch==2.0.0 numpy scikit-learn==1.1.2 coremltools
pip install jupyterlab
```
## Generate CoreML models
Short cut: generated default models can be downloaded from here: [Models.zip](https://1drv.ms/u/s!AmZvROUhZtUdgZFhd7l3VNuAyfMNCA?e=QeoIZ7)


### with Notebook
Open Jupyter Lab
```
jupyter lab
```
and run "GenerateGEMMModels.ipynb"

### with python script
```
python GenerateGEMMMOdels.py
```

By defualt, a list of models with dimension from 256x256 to 16384x16384, with 1, 8, 16, 24, 32 layers will be generated in both FP16 and INT8LUT format.

NOTE: To generate all the default models, it may take a few hours.

NOTE: if the size of model exceed 2GB, the generation will fail.

TIP: you may modify the script to generate models with different dimensions

## Performance profiling with xcode
Install xcode from AppStore

![xcode icon](screenshots/xcode_icon.png)

Once xcode is installed, you should be able to open the generated mlmodel files

![example mlmodel file](screenshots/mlmodel_icon.png)

With the mlmodel file open, performance profiling can be done by click on the performance tab, then, click on the "+" sign to add performance tests.

![create performance report](screenshots/create_performance_report.png)

![example performance report](screenshots/performance_report_example.png)
