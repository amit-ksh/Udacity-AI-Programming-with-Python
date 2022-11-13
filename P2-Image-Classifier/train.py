import os
from utils import get_train_input_args
from model import create_model, load_datasets, select_device, train, save_checkpoint

def main():
  # Get the input arguments
  in_args = get_train_input_args()
  # print(in_args)

  if in_args.hidden_units == []:
    in_args.hidden_units = [120, 60]

  # Load the datasets from the data directory
  dataloaders, class_to_idx = load_datasets(in_args.data_dir)

  # Create the model
  model = create_model(in_args.arch, in_args.hidden_units)
  model.class_to_idx = class_to_idx
  
  # Select the device
  device = select_device(in_args.gpu)
  
  # Train the model
  accuracy = train(dataloaders, model, in_args.epochs, device)

  # Create the save directory, if not exists
  if not os.path.exists(in_args.save_dir):
    os.mkdir(in_args.save_dir)
    
  # Save the model
  save_path = f"{in_args.save_dir}/{in_args.checkpoint}.pth"
  save_checkpoint(save_path, model)



if __name__ == '__main__':
  main()
