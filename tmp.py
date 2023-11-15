import wandb

sweep_config = {
    'name': 'my-sweep', 
    'method': 'grid',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64, 128]},
    }
}
print("Hellow World")
sweep_id = wandb.sweep(sweep_config)

def train():
    run = wandb.init()  # This will get the next configuration
    config = run.config
    print("Learning rate: ", config.learning_rate)
    print("Batch size: ", config.batch_size)
    # Insert your training code here
    run.finish()  # This will mark the current run as finished

# This will iterate over all runs in the sweep
# wandb.agent(sweep_id, function=train)


print("Hellow World")


