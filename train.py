from models import *
from losses import *
from tqdm import tqdm
import torch
from torch import nn
from viz import show_tensor_images


def params_and_models(device):
    PARAMS = {"adversarial_criterion": nn.MSELoss(),
              "recon_criterion": nn.L1Loss(),
              "EPOCHS": 3,
              "dim_A": 3,
              "dim_B": 3,
              "DISPLAY_STEP": 200,
              "BATCH_SIZE": 1,
              "learning_rate": 0.000197,
              "target_shape": 256,
              "device": device}

    gen_AB = Generator(PARAMS["dim_A"], PARAMS["dim_B"]).to(PARAMS["device"])
    gen_BA = Generator(PARAMS["dim_B"], PARAMS["dim_A"]).to(PARAMS["device"])
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()),
                               lr=PARAMS["learning_rate"], betas=(0.5, 0.999))
    disc_A = Discriminator(PARAMS["dim_A"]).to(PARAMS["device"])
    disc_A_opt = torch.optim.Adam(disc_A.parameters(),
                                  lr=PARAMS["learning_rate"],
                                  betas=(0.5, 0.999))
    disc_B = Discriminator(PARAMS["dim_B"]).to(PARAMS["device"])
    disc_B_opt = torch.optim.Adam(disc_B.parameters(),
                                  lr=PARAMS["learning_rate"],
                                  betas=(0.5, 0.999))
    return PARAMS, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def weighted_models(gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt,
                    pretrained = False, weight_path_if_any = None):
    if pretrained:
        # Give the path of the presaved file
        pre_dict = torch.load(weight_path_if_any)
        gen_AB.load_state_dict(pre_dict["gen_AB"])
        gen_BA.load_state_dict(pre_dict["gen_BA"])
        gen_opt.load_state_dict(pre_dict["gen_opt"])
        disc_A.load_state_dict(pre_dict["disc_A"])
        disc_A_opt.load_state_dict(pre_dict["disc_A"])
        disc_B.load_state_dict(pre_dict["disc_B"])
        disc_B_opt.load_state_dict(pre_dict["disc_B"])
    else:
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)
    return gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt


def train(data_loaded, device = 'cpu', is_pretrained = False, do_save_model=False, weight_path_if_any = None):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = data_loaded
    current_step = 0
    PARAMS, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt = params_and_models(device = device)
    MODELS = weighted_models(gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt,
                             weight_path_if_any = weight_path_if_any, pretrained = is_pretrained)
    for epoch in range(PARAMS["EPOCHS"]):
        for real_A, real_B in tqdm(dataloader):
            curr_batch_size = len(real_A)
            real_A = real_A.to(PARAMS["device"])
            real_B = real_B.to(PARAMS["device"])

            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_AB(real_B)
            disc_A_loss = get_discriminator_loss(real_A, fake_A, disc_A, PARAMS["adversarial_criterion"])
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_discriminator_loss(real_B, fake_B, disc_B, PARAMS["adversarial_criterion"])
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_generator_loss(real_A, real_B, gen_AB, gen_BA,
                                                          disc_A, disc_B,
                                                          PARAMS["adversarial_criterion"],
                                                          PARAMS["recon_criterion"],
                                                          PARAMS["recon_criterion"])
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            mean_discriminator_loss += disc_A_loss.item() / PARAMS["DISPLAY_STEP"]
            mean_generator_loss += gen_loss.item() / PARAMS["DISPLAY_STEP"]

            # Code for Visualization
            if current_step % PARAMS["DISPLAY_STEP"] == 0:
                print(
                    f"Epoch {epoch}: Step {current_step}: Generator (U-Net) loss: {mean_generator_loss}, "
                    f"Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(PARAMS["dim_A"],
                                                                      PARAMS["target_shape"],
                                                                      PARAMS["target_shape"]))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(PARAMS["dim_B"],
                                                                      PARAMS["target_shape"],
                                                                      PARAMS["target_shape"]))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if do_save_model:
                    torch.save({
                        "gen_AB": gen_AB.state_dict(),
                        "gen_BA": gen_BA.state_dict(),
                        "gen_opt": gen_opt.state_dict(),
                        "disc_A": disc_A.state_dict(),
                        "disc_A_opt": disc_A_opt.state_dict(),
                        "disc_B": disc_B.state_dict(),
                        "disc_B_opt": disc_B_opt.state_dict()
                    }, f"cycleGAN_{current_step}.pth")
            current_step += 1
        return gen_AB, gen_BA

