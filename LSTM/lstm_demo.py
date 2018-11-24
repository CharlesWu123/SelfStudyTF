import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# state是一个包含两个张量的LSTMStateTuple类，其中state.c和state.h分别对应了
# c状态和h状态
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0
for i in range(num_steps):
    # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
    if i > 0: tf.get_variable_scope().reuse_variables()
    lstm_output, state = lstm(current_input, state)
    final_output = fully_connected(lstm_output)

    loss += calc_loss(final_output, excepted_output)

