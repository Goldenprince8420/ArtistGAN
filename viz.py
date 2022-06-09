import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_example(data_loaded):
    photo_img, monet_img = next(iter(data_loaded))

    def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(s)

        return img

    f = plt.figure(figsize=(8, 8))

    f.add_subplot(1, 2, 1)
    plt.title('Photo')
    photo_img = unnorm(photo_img)
    plt.imshow(photo_img[0].permute(1, 2, 0))

    f.add_subplot(1, 2, 2)
    plt.title('Monet')
    monet_img = unnorm(monet_img)
    plt.imshow(monet_img[0].permute(1, 2, 0))
