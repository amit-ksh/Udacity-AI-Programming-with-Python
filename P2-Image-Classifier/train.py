import os
from utils import get_train_input_args
from model import create_model, load_datasets, select_device, train, save_checkpoint

# Usage: python train.py flowers --gpu

def main():
  # Get the input arguments
  in_args = get_train_input_args()
  # print(in_args)

  if in_args.hidden_units == []:
    in_args.hidden_units = [512, 256]

  # Load the datasets from the data directory
  print('START: Loading Dataset!')
  dataloaders, class_to_idx = load_datasets(in_args.data_dir)
  print('END: Loading Dataset!')
                            
  # Create the model
  print('START: Creating Model!')
  model = create_model(in_args.arch, in_args.hidden_units)
  model.class_to_idx = class_to_idx
  print('END: Creating Model')
  
  # Select the device
  device = select_device(in_args.gpu)
  
  # Train the model
  print('START: Training Model!')
  accuracy = train(dataloaders, model, in_args.epochs, device)
  print('END: Training Model!')

  # Create the save directory, if not exists
  if not os.path.exists(in_args.save_dir):
    os.mkdir(in_args.save_dir)
    
  # Save the model
  save_path = f"{in_args.save_dir}{in_args.checkpoint}.pth"
  print(f'START: Saving Model to {save_path}!')
  save_checkpoint(save_path, model)
  print(f'END: Saving Model!')



if __name__ == '__main__':
  main()
  