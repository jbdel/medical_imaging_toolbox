from .cnn import CNN, CNNConstrained


def get_model(args):
    if 'cosine' in args.losses and ('resnet' in args.backbone or 'densenet' in args.backbone):
        return CNNConstrained
    elif 'resnet' in args.backbone or 'densenet' in args.backbone:
        return CNN
    else:
        raise NotImplementedError

