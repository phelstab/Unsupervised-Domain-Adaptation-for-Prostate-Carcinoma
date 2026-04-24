"""
Default hyperparameters for ISUP Classification CNN
"""

def get_isup_hyperparams(args):
    """Set default hyperparameters for ISUP classification"""
    
    # Model architecture
    if not hasattr(args, 'num_channels'):
        args.num_channels = 3
    if not hasattr(args, 'num_classes'):
        args.num_classes = 6  # ISUP grades 0-5
    if not hasattr(args, 'model_features'):
        args.model_features = [32, 64, 128, 256, 512, 1024]
    if not hasattr(args, 'model_strides'):
        args.model_strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
    if not hasattr(args, 'use_attention'):
        args.use_attention = True
    if not hasattr(args, 'dropout_rate'):
        args.dropout_rate = 0.5
    
    # Training parameters
    if not hasattr(args, 'batch_size'):
        args.batch_size = 4  # Smaller batch size for classification
    if not hasattr(args, 'learning_rate'):
        args.learning_rate = 1e-3
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 1e-5
    if not hasattr(args, 'num_epochs'):
        args.num_epochs = 100
    if not hasattr(args, 'patience'):
        args.patience = 20
    
    # Loss parameters
    if not hasattr(args, 'classification_weight'):
        args.classification_weight = 1.0
    if not hasattr(args, 'segmentation_weight'):
        args.segmentation_weight = 0.3
    if not hasattr(args, 'use_focal_loss'):
        args.use_focal_loss = False  # Changed default to False (using CrossEntropyLoss)
    if not hasattr(args, 'focal_gamma'):
        args.focal_gamma = 2.0
    
    # Data parameters
    if not hasattr(args, 'image_shape'):
        args.image_shape = [20, 256, 256]
    if not hasattr(args, 'num_threads'):
        args.num_threads = 4
    
    # Validation parameters
    if not hasattr(args, 'validate_n_epochs'):
        args.validate_n_epochs = 1
    if not hasattr(args, 'validate_min_epoch'):
        args.validate_min_epoch = 5
    
    return args
