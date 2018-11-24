import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载lenet_inference.py和lenet_train.py中定义的常量和函数
from lenet import lenet_inference
from lenet import lenet_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [5000, lenet_inference.IMAGE_SIZE,
                                        lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, lenet_inference.OUTPUT_NODE],
                            name='y-input')


        y = lenet_inference.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(lenet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(lenet_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    vx = np.reshape(mnist.validation.images, (5000, lenet_inference.IMAGE_SIZE,
                                                              lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS))
                    vy = mnist.validation.labels
                    validate_feed = {x: vx, y_: vy}
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g'
                          % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
