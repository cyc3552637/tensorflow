import tensorflow as tf
input1 = tf.placeholder(tf.float32)
#tf.placeholder(dtype, shape=None, name=None)
#此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
#dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
#shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
#name：名称。
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)#input1*input2

with tf.Session() as sess:
  print (sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
  print (sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

#  答案是，tf.Variable适合一些需要初始化或被训练而变化的权重或参数，
#  而tf.placeholder适合通常不会改变的被训练的数据集。
#  [array([ 14.], dtype=float32)]数组外面在套[]，会自动标识一下array
#  [ 14.]
