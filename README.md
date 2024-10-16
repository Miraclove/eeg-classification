
# Reading
## Reference article (You should read first before going through)
TorchEEG: A PyTorch Lib for Deep EEG Analysis

    https://medium.com/@tczhangzhi/torcheeg-a-pytorch-lib-for-deep-eeg-analysis-a25ca12175e8

How to work with Pytorch, setting dataset and training

    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

TorchEEG Documentation

    https://torcheeg.readthedocs.io/en/latest/index.html

Sometimes the document is not up-to-date, you need to check pypi repo

    https://pypi.org/project/torcheeg/  
SEEDIV dataset

    https://torcheeg.readthedocs.io/en/v1.1.0/generated/torcheeg.datasets.SEEDIVDataset.html

Source Code of SEEDIV dataset

    https://torcheeg.readthedocs.io/en/v1.0.10/_modules/torcheeg/datasets/module/emotion_recognition/seed_iv.html


# Environment
## Environment Setup, it can be tough and painful especially with cuda
### install conda
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### install cuda==11.8 if you have Nvidia GPU, the most widely support version, this can be time consuming
    https://developer.nvidia.com/cuda-11-8-0-download-archive

### create environment, as pytorch only support python >= 3.8, do not go too advanced
    conda create -n eeg python=3.9
    conda activate eeg


### install packages if you have cuda

    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    pip install torcheeg
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
    pip install torch-geometric


    # torcheeg uses very old repo, fix if you need
    pip install mne==1.0.3 
    pip install scipy==1.7.3 
    pip install numpy==1.23 


### install packages if you do not have cuda

    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
    pip install torcheeg
    pip install torch-scatter
    pip install torch-sparse
    pip install torch-cluster
    pip install torch-spline-conv
    pip install torch-geometric


    # torcheeg uses very old repo, fix if you need
    pip install mne==1.0.3 
    pip install scipy==1.7.3 
    pip install numpy==1.23 


# Experiment
## Try quick-start.ipynb