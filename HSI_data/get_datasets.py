def get_class_splits(args):
    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'PaviaU':
        args.image_size = 13
        args.train_classes = range(4)
        args.unlabeled_classes = range(4, 9)

    elif args.dataset_name == 'Trento':
        args.image_size = 7
        args.train_classes = range(3)
        args.unlabeled_classes = range(3, 6)

    elif args.dataset_name == 'SA':
        args.image_size = 9
        args.train_classes = range(8)
        args.unlabeled_classes = range(8, 16)

    return args
