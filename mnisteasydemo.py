import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784]) #一个像素为784列行不定的输入图片
y_actual = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))        #初始化权值W，784行10列的矩阵
b = tf.Variable(tf.zeros([10]))            #初始化偏置项b，10列的矩阵
# 在图片中每一个像素点都表示某张图片里的某个像素的强度值，值介于0和1之间
# x与W矩阵相乘+b得到的就是图片的证据，实际上就是每一次的对于图片像素点的鉴别过程，因为输入的图片行数不定远大于W值
# 权值对于图片鉴定得到的结果，只是一个10列大小的图片结果
# 对于结果进行softmax回归，得到一个10列未知行的图片矩阵
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     #加权变换并进行softmax回归，得到预测概率，得到的是
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵，就是loss
# 对loss进行训练，上面表示了我们预期是1，不满足1的我们需要调整，调整的是W和b
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):               #训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)           #按批次训练，每批100行数据
        #用训练数据实际输入x和y
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   #执行训练
        if(i%100==0):                  #每训练100次，测试一次
            print ("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))