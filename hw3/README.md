# Environments
```Ubuntu Version: 18.04, CUDA Version: 11.2, GeForce RTX 3090 Ti 24GB```

``` python=3.11.5, pytorch=2.1.1 + cu118 ```


# How to run my code
## Setup environment

1. Create a virtual environment with conda

```bash 
conda create -n your-env-name python=3.11.5
conda activate your-env-name
```

2. Install Pytroch
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
 ```

3. Install other dependencies
```bash
pip install -r requirements.txt
 ```

4. Compiling CUDA operators
```bash
cd DINO/models/dino/ops
python setup.py build install
cd ../../../..
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

## Image captioning
This would select 140 images from the dataset and generate captions for them.
```bash
python Image_Captioning.py --dataset_dir /path/to/your/dataset/directory
```

## Generate new picture with caption and bounding box
1. Download the model from [Github](https://github.com/gligen/GLIGEN)

2. Run the following command
```bash
cd GLIGEN
python gligen_inference.py --save_folder /path/to/save/generated/images --ckpt_file /path/to/your/checkpoint.pth
```

## Add new annotations
This would modify the annotations/train.json to add new annotations.
```bash
cd ..
python generate_new_train.py --dataset_dir /path/to/your/dataset/directory
```

## Run training, testing
```bash
cd DINO
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

2. Copy the generated images to your train folder in dataset

3. Run training script

```bash
./scripts/DINO_train_swin.sh /path/to/save/model/directory /path/to/your/dataset/directory
 ```

### Testing
```bash
./scripts/DINO_eval.sh /path/to/your/dataset/directory /path/to/your/checkpoint.pth
```

