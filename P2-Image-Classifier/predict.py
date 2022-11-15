import json
from utils import get_predict_input_args, process_image
from model import predict, load_checkpoint, select_device

# Usage: python predict.py pink_primrose.jpg model_checkpoints/checkpoint.pth
# Usage: python predict.py globe_thistle.jpg model_checkpoints/checkpoint.pth
# Usage: python predict.py fire_lily.jpg model_checkpoints/checkpoint.pth

def main():
  # Get the input arguments
  in_args = get_predict_input_args()
  # print(in_args)

  with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

  # Load the datasets from the data directory
  model = load_checkpoint(in_args.checkpoint)

  # process image
  image_tensor = process_image(in_args.image_path)
  
  # Select the device
  device = select_device(in_args.gpu)
  
  # Predict the model
  top_probs, top_classes = predict(image_tensor, model, in_args.top_k, device)

  top_labels = [cat_to_name[str(c)] for c in top_classes]

  # Show the prediction
  print('RESULT:-')
  for prob, label in zip(top_probs, top_labels):
    print(f'{label: <20}: {prob*100:.2f}%')



if __name__ == '__main__':
  main()