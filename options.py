import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
    parser.add_argument("--multi_gpu", type=bool, default=True, help="whether or not multi gpus")
    # Data
    parser.add_argument("--data_root", type=str, default="data", help="path to data root")
    parser.add_argument("--dataset_name", type=str, default="explor_all", choices=["explor_all"],
                        help="name of the dataset")
    parser.add_argument("--img_size", type=int, default=64, help="image size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--n_threads", type=int, default=32, help="number of threads of dataloader")
    parser.add_argument("--n_style", type=int, default=4, help="number of style input images")
    # Channel
    parser.add_argument("--channel", type=int, default=3, help="image channel")
    parser.add_argument("--attr_channel", type=int, default=37, help="attributes channel")
    parser.add_argument("--attr_embed", type=int, default=64,
                        help="attribute embedding channel, attribute id to attr_embed, must same as image size")
    parser.add_argument("--style_out_channel", type=int, default=128, help="number of style embedding channel")
    parser.add_argument("--n_res_blocks", type=int, default=16, help="number of residual blocks in style encoder")
    # Model
    parser.add_argument("--attention", type=bool, default=True, help="whether use the self attention layer in the generator")
    parser.add_argument("--dis_pred", type=bool, default=True, help="whether the discriminator predict the attributes")
    # Adam
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    # Experiment
    parser.add_argument("--experiment_name", type=str, default="att2font_en", help='experiment name')
    parser.add_argument("--check_freq", type=int, default=10, help='frequency of checkpoint epoch')
    parser.add_argument("--sample_freq", type=int, default=400, help="frequency of sample validation batch")
    parser.add_argument("--log_freq", type=int, default=100, help="frequency of sample training batch")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'test_interp'], help='mode')
    parser.add_argument("--test_epoch", type=int, default=0, help='epoch to test, 0 to test all epoches')
    parser.add_argument("--interp_cnt", type=int, default=11, help='number of interpolations')
    # Lambdas
    parser.add_argument("--lambda_l1", type=float, default=50.0, help='pixel l1 loss lambda')
    parser.add_argument("--lambda_char", type=float, default=3.0, help='char class loss lambda')
    parser.add_argument("--lambda_GAN", type=float, default=5.0, help='GAN loss lambda')
    parser.add_argument("--lambda_cx", type=float, default=6.0, help='Contextual loss lambda')
    parser.add_argument("--lambda_attr", type=float, default=20.0, help='discriminator predict attribute loss lambda')
    # Other Modules
    return parser
