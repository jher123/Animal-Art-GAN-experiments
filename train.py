import os
import time
import random
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable as V
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

from model import *
from utils import *


def train(
    dataloader,
    netG,
    netD,
    optimiserG,
    optimiserD,
    num_epochs,
    nz,
    gen_img_path,
    checkpoint_path,
    gen_img_freq=5,
    checkpoint_freq=500,
    # resume_training=False,
    debug=True,
    resume_path=None
):
    """ A method to train Wasserstein GAN given dataset loader object and options.

    Parameters
    ----------
    dataloader An object loading data.
    netG: nn.Module object, Generator net
    netD: nn.Module object, Discriminator net
    optimiserD: object, Discriminator optimiser
    optimiserG: object, Generator optimiser
    nz: int, the length of the random numbers vector
    num_epochs: int, the total number of epochs
    gen_img_path: str, path to the folder where generated images will be saved
    checkpoint_path: str, path to where model checkpoints will be saved
    gen_img_freq: int, every how many epochs to save image samples, default 5
    checkpoint_freq: 5, every how many epochs to save model checkpoints, default 500
    resume_from_checkpoint_path: str, path to the checkpoint file from which to resume training
    # resume_training: Boolean, whether to resume training from checkpoint. Default False.
    #                 If True, training will be resume from the checkpoint specified by
    #                 resume_from_checkpoint_path. This file will contain the number of
    #                 the epoch when it was saved.
    debug: Boolean, whether to save a dict with debug info:
            lossD, lossG, D(fake batch) and D(real batch). Default True.
    resume_path

    Returns
    -------
    netG: Trained Generator object
    netD: Trained Discriminator object
    debug_info: a dictionary with debug information as described above.
            If debug was set to False, this will be an empty dictionary.
    """

    # lists to store logging information
    debug_info = {}
    debug_info['lossD'] = []
    debug_info['lossG'] = []
    debug_info['real_res'] = []
    debug_info['fake_res'] = []

    if resume_path is not None:
        checkpoint = torch.load(resume_from_checkpoint_path)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimiserG.load_state_dict(checkpoint['optimiserG_state_dict'])
        optimiserD.load_state_dict(checkpoint['optimiserD_state_dict'])
        last_epoch = checkpoint['epoch']
        lossD = checkpoint['lossD']
        lossG = checkpoint['lossG']
        debug_info = checkpoint['debug_info']
        print('Resuming training from epoch {}, with lossD: {} and lossG: {}'.format(last_epoch, lossD, lossG))
    else:
        # Apply the weights_init function to randomly initialise all weights with mean=0 and stdev=0.2.
        last_epoch = 0
        netG.apply(weights_init)
        netD.apply(weights_init)

    gen_iters = 0
    for epoch in range(num_epochs + 1)[(last_epoch + 1):]:
        print('Running epoch {}/{} \n'.format(epoch, num_epochs + 1))
        # SET THE MODELS TO TRAINING MODE
        netD.train()
        netG.train()
        image_batch = next(iter(dataloader))
        i = 0
        n = len(image_batch[0]) # LENGTH OF THE BATCH
        while i < n:
            # for every 1 iteration of G, have 5 or more later iters of D
            d_iters = 100 if (gen_iters % 500 == 0) else 5
            j = 0

            # STEP 1: TRAIN THE DISCRIMINATOR
            set_trainable(netD, True)
            set_trainable(netG, False)
            while (j < d_iters) and (i < n):
                j += 1
                i += 1
                # CLIP WEIGHTS
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
                # REAL IMAGE BATCH
                real = V(image_batch[0])
                real = real.cuda()
                real_result = netD(real) # the avg output across batch of the D for all the real batch
                # FAKE IMAGE BATCH
                noise = V(torch.zeros(real.size(0), nz, 1, 1).normal_(0, 1))
                fake = netG(noise)
                fake_result = netD(V(fake.data)) # the avg D output for all the fake batch
                # ZERO THE GRADIENTS FOR D AND THEN CALCULATE LOSS + BACKPROP
                netD.zero_grad()
                lossD = real_result-fake_result
                lossD.backward()
                # D OPTIMISER UPDATE STEP
                optimiserD.step()

            # STEP 2: TRAIN THE GENERATOR
            set_trainable(netD, False)
            set_trainable(netG, True)
            # ZERO THE GRADIENTS FOR G AND THEN CALCULATE LOSS + BACKPROP + UPDATE STEP
            netG.zero_grad()
            noise1 = V(torch.zeros(real.size(0), nz, 1, 1).normal_(0, 1)) # TODO
            # opt TODO: add Gaussian noise to every layer of generator: rain_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            # USE THE FAKE FROM ABOVE?
            lossG = netD(netG(noise1)).mean(0).view(1)
            lossG.backward()
            # G OPTIMISER UPDATE STEP
            optimiserG.step()
            gen_iters += 1

        lossDnp = lossD.data.cpu().numpy()
        lossGnp = lossG.data.cpu().numpy()
        realnp = real_result.data.cpu().numpy()
        fakenp = fake_result.data.cpu().numpy()

        if debug is True:
            debug_info['lossD'].append(lossDnp)
            debug_info['lossG'].append(lossGnp)
            debug_info['real_res'].append(realnp)
            debug_info['fake_res'].append(fakenp)

        print(f'\n Loss_D (real - fake result) {lossDnp}; Loss_G {lossGnp}; '
              f'D_real {realnp}; Loss_D_fake {fakenp} \n')

        # SAVE CHECKPOINTS EVERY CHECKPOINT_FREQ EPOCHS
        if epoch%checkpoint_freq == 0:
            print('Saving checkpoint at epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimiserG_state_dict': optimiserG.state_dict(),
                'optimiserD_state_dict': optimiserD.state_dict(),
                'lossD': lossD,
                'lossG': lossG,
                'debug_info': debug_info
            },
                f'{checkpoint_path}/epoch_{str(epoch)}.pth.tar')
        # SAVE IMAGES EVERY IMG_FREQ EPOCHS
        if epoch%gen_img_freq == 0:
            netD.eval()
            netG.eval()
            fixed_noise = create_noise(64)
            fake = netG(fixed_noise).data.cpu()
            vutils.save_image(
                fake,'%s/fake_image_epoch_%03d.jpg' % (gen_img_path, epoch), normalize=True
            )

    return netG, netD, debug_info

def main():
    # set seed
    random.seed(6)

    start_time = time.time()

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=64, type=int, help='batch size')
    parser.add_argument('--im_size', default=64, type=int, help='image size')
    parser.add_argument('--num_epochs', default=2000, required=True, type=int, help='the number of training epochs')
    parser.add_argument('--nz', default=100, type=int, help='the size of the random input vector')
    parser.add_argument('--ks', default=4, type=int, help='kernel size')
    parser.add_argument('--ndf', default=64, type=int, help='determines the depth of the feature maps carried through the discriminator/critic')
    parser.add_argument('--ngf', default=64, type=int, help='determines the depth of the feature maps carried through the generator')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--version_name', required=True, help='what to name the subfolder with info from this run as')
    parser.add_argument('--img_folder_name', type=str, required=True, help='path to folder where to save generated images')
    parser.add_argument('--gen_img_freq', default=5, help='every how many epochs to save generated images')
    parser.add_argument('--checkpoint_freq', default=500, help='every how many epochs to save checkpoints')
    parser.add_argument('--resume_from_checkpoint_path', help='checkpoint file from which to resume training')
    parser.add_argument('--debug', default=True, help='whether to save debug info whilst training')
    parser.add_argument('--resume', default=False, help='whether to resume training from checkpoint')
    parser.add_argument('--epoch_num', type=int, help='Number of epoch from which to restart training, must be a multiple of 500.')

    opt = parser.parse_args()
    print('Parsed arguments: \n {}'.format(opt))

    # HYPERPARAMS
    BATCH_SIZE = opt.bs # default: 64
    IM_SIZE = opt.im_size # default: 64x64
    NZ = opt.nz # default: 100
    NUM_EPOCHS = opt.num_epochs # default: 2000
    KS = opt.ks # default: 4x4 Jeremy, 5x5 Siraj
    LR = opt.lr # 1e-4 Jeremy, 2e-4 # Siraj
    NDF = opt.ndf # default: 64
    NGF = opt.ngf # default: 64
    version = opt.version_name
    img_folder_name = opt.img_folder_name

    PATH = os.path.abspath(__file__ + "/../../")
    INPUT_PATH = os.path.join(PATH, 'input_data')
    PATH = os.path.join(PATH, 'data')
    TMP_PATH = os.path.join(os.path.join(PATH, 'checkpoints'), version)
    os.makedirs(TMP_PATH, exist_ok=True)
    GEN_PATH = os.path.join(os.path.join(PATH, 'generated_imgs'), version)
    os.makedirs(GEN_PATH, exist_ok=True)


    # PREPARING THE DATA
    # dataset
    tfms = transforms.Compose(
        [transforms.Resize(IM_SIZE),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(), # this swaps colour axis as numpy image is H x W x C and torch image is C x H x W, and does torch.from_numpy(image)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = dset.ImageFolder(root=INPUT_PATH, transform=tfms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # DEFINE THE MODELS
    netG = Generator(IM_SIZE, KS, NZ, NGF).cuda()
    netD = Discriminator(IM_SIZE, KS, NDF).cuda()

    # DEFINE THE OPTIMISERS
    optimiserD = optim.RMSprop(netD.parameters(), lr = LR)
    optimiserG = optim.RMSprop(netG.parameters(), lr = LR)

    if resume_training:
        if opt.resume_from_checkpoint_path is None:
            checkpoint = f'{TMP_PATH}/epoch_{str(opt.epoch_num)}.pth.tar'
        else:
            checkpoint = opt.resume_from_checkpoint_path
    else:
        checkpoint = None

    # TRAINING
    train(
        dataloader,
        netG,
        netD,
        optimiserG,
        optimiserD,
        NUM_EPOCHS,
        NZ,
        gen_img_path=GEN_PATH,
        checkpoint_path=TMP_PATH,
        gen_img_freq=opt.gen_img_freq,
        checkpoint_freq=opt.checkpoint_freq,
        debug=opt.debug,
        resume_path=checkpoint
    )

    print('Time elapsed in min: {}'.format((time.time() - start_time)/60.))


if __name__ == '__main__':
    main()
