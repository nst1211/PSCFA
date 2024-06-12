import argparse


def get_config():

    parse = argparse.ArgumentParser(description='peptide default main')
    parse.add_argument('-task', type=str, default='model_select')

    # 模型训练参数

    parse.add_argument('-subtest', type=bool, default=False)
    parse.add_argument('-vocab_size', type=int, default=21,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=21,
                       help='Number of peptide functions')
    parse.add_argument('-batch_size', type=int, default=64*4,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=2) #230
    parse.add_argument('-learning_rate', type=float, default=0.001)
    parse.add_argument('-threshold', type=float, default=0.7)

    #contrastive
    parse.add_argument('--contrastive_weight', type=float, default=0.1,
                        help="weight of contrastive learning")
    parse.add_argument('--T', type=float, default=0.02,
                        help="temperature of contrastive learning")

    parse.add_argument('--pretrained', type=bool, default=True,#False True
                        help="use pretrained textcnn model")
    parse.add_argument('-pretrained_path', type=str, default='saved_models/model.pth',
                       help='Path of the new_model')
    parse.add_argument('--FA', type=bool, default=True,#False True
                        help="use pretrained textcnn model")

    parse.add_argument('-check_pt_new_model_path', type=str, default='saved_models/new_model.pth',
                       help='Path of the new_model')
    parse.add_argument('-check_pt_model_path', type=str, default='saved_models/model.pth',
                       help='Path of the model')
    parse.add_argument('-feature_dict_path', type=str, default='saved_models/feature_dict.npy')

    # #VAE&Augmentaion
    parse.add_argument("--vae_learning_rate", default=0.0001, required=False, type=float,
                        help="learning rate of VAE")
    parse.add_argument('--threshold2', type=int, default=500 ,
                        help="head to tail threshold")#20
    parse.add_argument('--da_number', type=int, default=25,
                        help="times of augmentation")
    parse.add_argument('--vae_epochs', type=int, default=200,#250
                        help="eopch of VAE")
    parse.add_argument('--vae_batch_size', type=int, default=64*2,
                        help="batch size of VAE")
    parse.add_argument('-check_pt_vae_model_path', type=str, default='saved_models/vae.pth',
                       help='Path of the vae')
    parse.add_argument('--vae_early_stop_tolerance', type=int, default=10,
                        help="early stop of VAE")
    #adjustment(calibartion)
    parse.add_argument('--calibration_batch_size', type=int, default=64*3,
                        help="batch size of adjustment")
    parse.add_argument("--calibration_learning_rate", default=0.001,required=False, type=float,
                        help="learning rate of adjustment")
    parse.add_argument('--calibration_epochs', type=int, default=162)#100


    # 深度模型参数
    parse.add_argument('-embedding_size', type=int, default=64*4,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=126,
                       help='Number of the filter')
    parse.add_argument('-filter_size', type=list, default=[3, 4, 5, 6],
                       help='Size of the filter')

    parse.add_argument('-model_path', type=str, default=None,
                       help='Path of the training data')
    parse.add_argument('-train_direction', type=str, default='dataset/train.txt',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='dataset/test.txt',
                       help='Path of the test data')


    config = parse.parse_args()
    return config


if __name__ == '__main__':
    args = get_config()
    from train_test import TrainAndTest
    TrainAndTest(args)
