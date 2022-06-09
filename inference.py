import PIL
import torch.cuda
import os
import numpy as np
from tqdm import tqdm
import shutil


def get_submission(photo_data_loaded, generator, device = 'cpu',
                   save_to_directory = "./Results", saved_file_name = "images"):
    i = 1
    for real_A in tqdm(photo_data_loaded):
        real_A = real_A.to(device)
        with torch.no_grad():
            prediction = generator(real_A)[0].cpu().numpy()
        prediction = (prediction * 127.5 + 127.5).T.astype(np.uint8)
        output = PIL.Image.fromarray(prediction)
        output = output.rotate(270)
        output.save(os.path.join(save_to_directory, "painted{}.png".format(i)))
        i += 1
        torch.cuda.empty_cache()
        break

    shutil.make_archive(os.path.join(".", saved_file_name), 'zip', save_to_directory)
    return
