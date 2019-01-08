# Animal art with GANs
PyTorch implementations of Wasserstein Generative Adversarial Network https://arxiv.org/abs/1701.07875
and a more recent Self-Attention GAN https://arxiv.org/abs/1805.08318 as well webscrapers to download images from flickr and unsplash.
The goal of this project is to experiment with creating artsy looking images of animals using GANs by swapping images for different animals for a few epochs during training as well as to compare visual results achieved from training with different types of GANs.
The networks can generate either 64x64 or 128x128 images.

## Examples

## Prerequisites
The models are meant to be run on CUDA enabled GPU.
The main requirements are Python 3 and packages contained in `requirements.txt`.
For the unsplash scraper, geckodriver and FireFox are also required.

## Getting Started
To download images from Flickr/Unsplash run one of the scraper scripts providing a keyword and the number of images to download for Flickr:
`python3 flickr_scraper.py --keyword jellyfish --num_images 3000`
`python3 unsplash_scraper.py --keyword jellyfish`

Place the images in `input_data`
The training scripts train the network either from scratch or resuming from a checkpoint file.
It saves images every `gen_img_freq` epochs (default 5) and saves model and optimiser checkpoints as well as debug info if desired every
`checkpoint_freq` epochs. The full list of arguments is in `train.py` script. Example training a WGAN from scratch:
`python3 train_WGAN.py --bs 128 --im_size 128 --num_epochs 3500 --version_name jellyfish128 --img_folder_name jellyfish --checkpoint_freq 400`
Example training a WGAN from checkpoint saved at 2000 epoch:
`python3 train_WGAN.py --bs 128 --im_size 128 --num_epochs 3500 --version_name jellyfish128 --img_folder_name jellyfish --checkpoint_freq 400 --resume True --resume_epoch_num 2000`

## Acknowledgements
- The code for the WGAN was inspired by fast.ai course implementation https://github.com/fastai/fastai/blob/master/courses/dl2/wgan.ipynb
- The flickr scraper is a modified version of https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6
- The idea of swapping images during training to achieve artistic effects inspired by Robbie Barrat's work https://github.com/robbiebarrat/art-DCGAN
