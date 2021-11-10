import os
from resnet_model import ResNet
import input
import tensorflow as tf

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    X_train, Y_train, X_unsupervised = input.load_CT('../DataPKL/')
    X_test, Y_test, _ = input.load_CT('../DataTest/')
    X_train, X_test = input.data_preprocessing(X_train, X_test)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        res_model = ResNet(model_name='lung',
                           sess=sess,
                           train_data=[X_train, Y_train],
                           tst_data=[X_test, Y_test],
                           unsupervised_data=X_unsupervised,
                           epoch=200,
                           restore_epoch=0,
                           num_class=6,
                           ksize=3,
                           out_channel1=64,
                           out_channel2=128,
                           out_channel3=256,
                           learning_rate=1e-4,
                           weight_decay=1e-4,
                           batch_size=128,
                           img_height=32,
                           img_width=32,
                           train_phase='Train')

        sess.run(tf.global_variables_initializer())
        res_model.train()
        #res_model.unsupervised_train()
        #res_model.test()

