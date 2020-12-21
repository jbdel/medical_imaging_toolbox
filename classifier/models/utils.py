from .cnn import CNN, CNNConstrained


def get_model(args):
    if 'cosine' in args.losses and ('resnet' in args.model or 'densenet' in args.model):
        return CNNConstrained(args)
    elif 'resnet' in args.model or 'densenet' in args.model:
        return CNN(args)
    else:
        raise NotImplementedError
