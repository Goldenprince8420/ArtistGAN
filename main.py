from data import *
from viz import *
from inference import *
from train import *

if __name__ == '__main__':
    os.makedirs('Saved_Models', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    seed_index = 1003
    set_seed(seed_index)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = ImageDataset('data/monet_jpg', 'data/photo_jpg')
    photo_data = PhotoDataset('data/photo_jpg')
    monet_data = PaintingDataset('data/monet_jpg')
    data_loaded = load_data(data)
    monet_data_loaded = load_data(monet_data)
    photo_data_loaded = load_data(photo_data)
    show_example(data_loaded)
    generator_real_to_painter, generator_painter_to_real = train(data_loaded, device = device,
                                                                 weight_path_if_any = "weights/cyclegan_17800.pth")
    get_submission(photo_data_loaded, generator = generator_real_to_painter)
