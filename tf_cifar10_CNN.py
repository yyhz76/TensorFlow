from random import shuffle
from keras.datasets import cifar10
from keras.utils import np_utils
import pickle
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

classes = 10
Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)

learning_rate = 0.001
batch_size = 96
batch_num = int(X_train.shape[0] / batch_size)
max_epochs = 50
dropout = 0.5


x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
y = tf.placeholder(tf.float32, [None, classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

weights = {
    'w_conv1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'w_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'w_conv3': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'w_fc1': tf.Variable(tf.random_normal([8*8*64, 512])),      # 32 / 2 / 2 = 8 b/c using 2 maxpool with stride 2
    'w_fc2': tf.Variable(tf.random_normal([512, classes]))
}

bias = {
    'b_conv1': tf.Variable(tf.random_normal([32]), dtype=tf.float32),
    'b_conv2': tf.Variable(tf.random_normal([64])),
    'b_conv3': tf.Variable(tf.random_normal([64])),
    'b_fc1': tf.Variable(tf.random_normal([512])),
    'b_fc2': tf.Variable(tf.random_normal([classes]))
}


conv1 = tf.nn.conv2d(x, weights['w_conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, bias['b_conv1'])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    # maxpool

conv2 = tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, bias['b_conv2'])
conv2 = tf.nn.relu(conv2)

conv3 = tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, bias['b_conv3'])
conv3 = tf.nn.relu(conv3)
conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    # maxpool

fc1 = tf.reshape(conv3, [-1, weights['w_fc1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['w_fc1']), bias['b_fc1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)                                                        # dropout

fc2 = tf.add(tf.matmul(fc1, weights['w_fc2']), bias['b_fc2'])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc2))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


train_loss = []
test_loss = []
train_acc = []
test_acc = []

ind_list = [i for i in range(50000)]

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(max_epochs):
        step = 0
        shuffle(ind_list)
        X_train_new = X_train[ind_list, :, :, :]          # shuffle training data
        Y_train_new = Y_train[ind_list, :]

        for num in range(batch_num):

            num = num + 1

            batch_x = X_train_new[num * batch_size:(num + 1) * batch_size, :]
            batch_y = Y_train_new[num * batch_size:(num + 1) * batch_size, :]

            _, train_loss_single, train_acc_single = \
                sess.run([train_op, loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            test_loss_single, test_acc_single = \
                sess.run([loss, accuracy], feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})

            print('Epoch {0:2d}  Step {5:2d}\nTrain Loss {1:12.6f} Train Accuracy {2:5.2f}\n Test Loss {3:12.6f}'
                  '  Test Accuracy {4:5.2f}\n'
                  .format(epoch, train_loss_single, train_acc_single, test_loss_single, test_acc_single, step))

            train_loss.append(train_loss_single)
            train_acc.append(train_acc_single)
            test_loss.append(test_loss_single)
            test_acc.append(test_acc_single)
            step += 1


with open('result.pkl', 'wb') as f:
    pickle.dump((train_loss, test_loss, train_acc, test_acc), f)
