import wandb
from torch.utils.tensorboard import SummaryWriter
from wandb.integration import torch as wandb_torch
from wandb.integration import tensorboard


def initialization_wandb_and_tensorboard(experiment_name , user_name = "yogjoshi14"):
    '''Inputs: takes in experiment_name and user_name for dandb
       Outputs: Object for tensorflow and wandb logger'''
    # Initialize Weights & Biases
    wandb.init(project=experiment_name, entity=user_name)

    # Set up TensorBoard with wandb integration
    writer = SummaryWriter('logs')
    wandb_logger = wandb_torch.WandbLogger()
    return writer, wandb_logger

'''
# Log data to wandb
    wandb.log({'training_loss': running_loss})
whereever writer is logged

 # Log example images to TensorBoard and wandb
    example_images = make_grid(inputs)
    writer.add_image('example_images', example_images, epoch * len(trainloader) + i)
    wandb.log({'example_images': [wandb.Image(example_images, caption='Example Images')]})

# Close TensorBoard writer
writer.close()

# Finish wandb run
wandb.finish()

'''

def upload_tensorboard_logs_to_wandb(tensorboard_log_dir, project_name, entity):
    # Initialize Weights & Biases
    wandb.init(project=project_name, entity=entity)

    # Load the TensorBoard log data
    tensorboard_data = tensorboard.backend.event_file_loader.EventFileLoader(tensorboard_log_dir)

    # Extract and log data from TensorBoard to wandb runs
    for scalar_event in tensorboard_data.Scalars():
        step = scalar_event.step
        tag = scalar_event.tag
        value = scalar_event.value

        # Log data to wandb
        wandb.log({tag: value}, step=step)

    # Finish wandb run
    wandb.finish()