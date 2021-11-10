import os
from resnet_model import ResNet
import input
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-train_phase', required=True, help='[whether to train or test the model]')
parser.add_argument('-unsup_num', required=True, type=int, help='[numbers of unsupervised samples]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-restore_epoch', default=0, type=int)
parser.add_argument('-num_class', default=10, type=int)
parser.add_argument('-ksize', default=3, type=int)
parser.add_argument('-out_channel1', default=16, type=int)
parser.add_argument('-out_channel2', default=32, type=int)
parser.add_argument('-out_channel3', default=64, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-weight_decay', default=1e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

X_train, Y_train, X_unsupervised = input.load_CT('../DataPKL/')
X_test, Y_test, _ = input.load_CT('../DataTest/')
X_train, X_test = input.data_preprocessing(X_train, X_test)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = ResNet(model_name=args.model_name,
                       sess=sess,
                       train_data=[X_train, Y_train],
                       tst_data=[X_test, Y_test],
                       unsupervised_data=X_unsupervised,
                       epoch=args.epoch,
                       restore_epoch=args.restore_epoch,
                       num_class=args.num_class,
                       ksize=args.ksize,
                       out_channel1=args.out_channel1,
                       out_channel2=args.out_channel2,
                       out_channel3=args.out_channel3,
                       learning_rate=args.learning_rate,
                       weight_decay=args.weight_decay,
                       batch_size=args.batch_size,
                       img_height=args.img_height,
                       img_width=args.img_width,
                       train_phase=args.train_phase)

    sess.run(tf.global_variables_initializer())
    if args.train_phase == 'Train':
        res_model.train()

    if args.train_phase == 'Test':
        res_model.test()

    if args.train_phase == 'Semi':
        res_model.unsupervised_train()
