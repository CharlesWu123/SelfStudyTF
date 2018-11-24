import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

# 通过tensorflow中的placeholder定义输出
x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

# 直接使用Tensorflow中提供的Keras API定义网络层结构
net = tf.keras.layers.Dense(500, activation='relu')(x)
y = tf.keras.layers.Dense(10, activation='softmax')(net)

# 定义损失函数和优化方法 注意这里可以混用Keras的API和源生态Tensorflow的API
loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 定义预测的正确率作为指标
acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_, y))

# 使用原生态Tensorflow的方式训练模型。这样可以有效的实现分布式
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10000):
        xs, ys = mnist_data.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={x: xs, y_: ys})
        if i % 1000 == 0:
            print('After %d training step(s), loss on training batch is %g.'
                  % (i, loss_value))
    print(acc_value.eval(feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))