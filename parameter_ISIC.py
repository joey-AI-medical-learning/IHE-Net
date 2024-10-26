
import argparse


def get_parameters():
    parser = argparse.ArgumentParser(description = 'Mean Teacher Trainer Pytorch')

    parser.add_argument('--exp', type=str,
                        default='IHE-Net(C_M)', help='experiment_name')

    parser.add_argument('--model', type=str,
                        default='1000', help='model_name')

    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')

    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')

    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')

    parser.add_argument('--patch_size', type=list, default=[224, 224],
                        help='patch size of network input')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    parser.add_argument('--num_classes', type=int, default=1,
                        help='output channel of network')

    # label and unlabel
    parser.add_argument('--labeled_bs', type=int, default=4,
                        help='labeled_batch_size per gpu')

    parser.add_argument('--labeled_num', type=int, default=130,
                        help='labeled data')

    parser.add_argument('--ict_alpha', type=int, default=0.2,
                        help='ict_alpha')
    # costs
    parser.add_argument('--ema_decay', type=float,
                        default=0.99, help='ema_decay')

    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')

    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')

    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')
    parser.add_argument('--temperature', type=float,
                        default=0.1, help='temperature of sharpening')
    parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')

    return parser.parse_args()




