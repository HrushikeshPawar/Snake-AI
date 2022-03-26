settings = {
    # Regarding Game Environment
    'block_size'    : 20,
    'grid_height'   : 6,
    'grid_width'    : 6,
    'border'        : 3,
    'speed'         : 5000,
    'font'          : 'Lora-Regular.ttf',

    #  Colors for the Game
    'black'         : (0, 0, 0),
    'white'         : (255, 255, 255),
    'grey'          : (150, 150, 150),
    'red'           : (255, 0, 0),
    'green'         : (0, 255, 0),
    'green2'        : (100, 255, 0),
    'blue'          : (0, 0, 255),
    'blue2'         : (0, 100, 255),

    # File Paths
    'GIF_path'          : 'Data/GIFs/',
    'Graph_path'        : 'Data/Graphs/',
    'Model_path'        : 'Data/Models/',
    'Checkpoint_path'   : 'Data/Checkpoints/',
    'Log_path'          : 'Data/Logs/',

    # Training Environment
    'max_memory'        : 100_000,
    'batch_size'        : 3000,
    'learning_rate'     : 0.001,
    'discount_rate'     : 0.95,
    'epsilon_max'       : 1.0,
    'epsilon_min'       : 0.01,
    'epsilon_decay'     : 0.9995,
    'epochs'            : 5000,
    'hidden_layer1_size': 256,
    'hidden_layer2_size': 0,
}
