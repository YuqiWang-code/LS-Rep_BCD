"""
Learning Rate Scheduler Utilities
"""

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    """
    Adjust learning rate with poly decay and warmup

    Args:
        args: training arguments
        optimizer: PyTorch optimizer
        epoch: current epoch
        iter: current iteration
        max_batches: total batches per epoch
        lr_factor: additional learning rate multiplier
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = args.max_steps if hasattr(args, 'max_steps') else max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    # Warmup
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr

    lr *= lr_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

    return lr
