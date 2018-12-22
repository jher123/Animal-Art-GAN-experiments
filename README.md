# Animal art with GANs
Implementation of Wasserstein Generative Adversarial Network https://arxiv.org/abs/1701.07875
and a more recent Self-Attention GAN https://arxiv.org/abs/1805.08318 as well webscrapers to download images from flickr and unsplash also included.
The goal of this project is to experiment with creating artsy looking images of animals using GANs by swapping images for different animals for a few epochs during training and compare visual results achieved from training with different types of GANs.

## Examples

## Getting Started

## Prerequisites
The models are meant to be run on CUDA enabled GPU.
The main requirements are

## Acknowledgements
- The code for the WGAN was inspired by fast.ai course implementation https://github.com/fastai/fastai/blob/master/courses/dl2/wgan.ipynb
- The flickr scraper is a modified version of https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6
- The idea of swapping images during training to achieve artistic effects inspired by Robbie Barrat's work https://github.com/robbiebarrat/art-DCGAN
