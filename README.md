# EAPNet
The source code of "Extractor-Attention-Predictor network for quantitative photoacoustic tomography".

Python Dependencies/Requirements

python==3.7.16
torch==1.10.1
torchvision==0.11.2
numpy==1.21.5
scipy==1.7.3
matplotlib==3.5.3
scikit-image==0.19.3

OS Requirements

This net is supported for Windows 10.

The "main.py" scipt is used for training and validation.

You can directly execute "display.py" to obtain results of example samples, along with visualized images.

The "data" fold contains five subfolds: "RS", "Va", "ST", "AR", "EXP".
RS: random shape dataset
Va: vasculature dataset
ST: sparse target dataset
AR: acoustic reconstructure dataset
EXP: phantom data

Each subfold comtains the corresponding example samples and the final checkpoint file of the used model.
