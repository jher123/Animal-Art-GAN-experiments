import torchvision.datasets as dset
from torchvision import transforms, utils

# Create the dataloader
dataset = dset.ImageFolder(
            root=dataroot,
            transforms=transforms.Compose(
                    [
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
            )
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


real_batch = next(iter(dataloader))
