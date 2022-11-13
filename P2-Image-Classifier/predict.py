import json
from utils import get_predict_input_args, process_image
from model import predict, load_checkpoint, select_device

def main():
  # Get the input arguments
  in_args = get_predict_input_args()
  print(in_args)

  with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

  # Load the datasets from the data directory
  model = load_checkpoint(in_args.checkpoint)

  # process image
  image_tensor = process_image(in_args.image_path)
  
  # Select the device
  device = select_device(in_args.gpu)
  
  # Predict the model
  top_probs, top_classes = predict(image_tensor, model, in_args.top_k)

  top_labels = [cat_to_name[str(c)] for c in top_classes]

  # Show the prediction
  print(top_probs, top_labels)



if __name__ == '__main__':
  main()
  