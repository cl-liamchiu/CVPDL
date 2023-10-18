# Environments
```Ubuntu Version=18.04, CUDA Version: 11.2, GeForce RTX 2080 Ti```

``` python=3.7.3, pytorch=1.9.0, cuda=11.1```


# How to run my code
## Setup environment

1. Create a virtual environment with conda

```bash 
conda create -n your-env-name python=3.7.3
conda activate your-env-name
```

2. Install Pytroch
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
 ```

3. Install other dependencies
```bash
cd DINO
pip install -r requirements.txt
 ```

4. Compiling CUDA operators
```bash
cd models/dino/ops
python setup.py build install
cd ../../..
 ```

5. Prepare dataset

```
dataset/
  ├── train/
  ├── valid/
  └── annotations/
  	├── train.json
  	└── val.json
```
## Run training, testing and inference
```bash
# Check if your are in DINO folder
pwd # need to return "/path/to/DINO"
```
### Training 
1. Downloand pretrained weight to /path/to/DINO/
    * [checkpoint0029_4scale_swin.pth](https://drive.google.com/file/d/1CrzFP0RycSC24KKmF5k0libLRJgpX9x0/view?usp=share_link) 
    * [swin_large_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)

```
DINO/
  ├── checkpoint0029_4scale_swin.pth
  ├── swin_large_patch4_window12_384_22k.pth
  ├── main.py
  ├── other file
  └── other folder/
```

2. Run training script

```bash
./scripts/DINO_train_swin.sh /path/to/result/directory /path/to/your/dataset/directory
 ```

### Testing
```bash
./scripts/DINO_eval.sh /path/to/your/dataset/directory /path/to/your/checkpoint.pth
# path/to/your/checkpoint.pth in the result directory
```

### Inference and visualization
1. Download my model
    * [checkpoint_best_regular.pth](https://drive.google.com/file/d/1lvKdp5UJQWSTLQZ4cUmxe7xlib4UwpmT/view?usp=sharing) 

2. Check path/to/DINO/inference_and_visualization.ipynb

3. Change the following path to your own path in the notebook
    ```python
    folder_path = "../dataset/hw1_dataset/test"
    output_path = "../output.json"
    model_checkpoint_path = "./logs/DINO/SWL-MS4/checkpoint_best_regular.pth"
    ```

