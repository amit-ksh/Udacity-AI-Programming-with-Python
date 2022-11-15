import argparse
import numpy as np
from PIL import Image 

def get_train_input_args():
  """Parse the command line arguments provided by the users 
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('data_dir', type=str, help='path to the data directory')
  parser.add_argument('--save_dir', type=str, default='model_checkpoints/', help='derictory to save model checkpoints')
  parser.add_argument('--checkpoint', type=str, default='checkpoint', help='name of the checkpoint file')
  parser.add_argument('--arch', type=str, default='alexnet', 
                      help='choose the model from these options: 1. alexnet  2. vgg13  3. vgg16  4. densenet121')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate. (default: 0.001)')
  parser.add_argument('--hidden_units', type=int, action='append', default=[], help='hidden layer units')
  parser.add_argument('--epochs', type=int, default=12, help='hidden layer units')
  parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
  parser.add_argument('--gpu', action='store_true', default=False, help='switch to GPU from CPU for training')

  return parser.parse_args()

def get_predict_input_args():
  """Parse the command line arguments provided by the users 
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('image_path', type=str, default='flowers/', help='path to image')
  parser.add_argument('checkpoint', type=str, help='path to saved checkpoint file')
  parser.add_argument('--top_k', type=int, default=5, help='return K classes having highest probability')
  parser.add_argument('--category_names', type=str, default='cat_to_name.json')
  parser.add_argument('--gpu', action='store_true', default=False, help='switch to GPU from CPU for training')

  return parser.parse_args()

def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, 
        returns an Numpy array

    Args:
        image: image path

    Returns:
        image: processed image
    """

    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    with Image.open(image) as im:
        im_resized = im.resize(size)
        im_cropped = im_resized.crop((16, 16, 240, 240))
        im_encoded = np.array(im_cropped) / 255
        im_normalized = (im_encoded - mean) / std
        im_final = im_normalized.transpose((2, 0, 1))
        
        return im_final
