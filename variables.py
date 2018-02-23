import tensorflow as tf
# Create four variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# shape: 一维的张量，也是输出的张量
# mean: 正态分布的均值
# stddev: 正态分布的标准差
# dtype: 输出的类型
# seed: 一个整数，当设置之后，每次生成的随机数都一样
# name: 操作的名字

biases = tf.Variable(tf.zeros([200]), name="biases")
# tf.zeros(shape,type=tf.float32,name=None)为0的float32数组

w2 = tf.Variable(weights.initialized_value(), name="w2")

w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

saver = tf.train.Saver() #指定部分变量saver = tf.train.Saver({"my_w2": w2})

# Later, when launching the model
with tf.Session() as sess: #类似sess = tf.Session()
  # Run the init operation.
  sess.run(init_op)

  #存储变量
  save_path = saver.save(sess, "D:\pydemo\data\model.ckpt")
  print ("Model saved in file: ", save_path)
  print ("weights = ", weights.eval())
  print ("biases = ", biases.eval())
  print ("w2 = ", w2.eval())
  print ("w_twice = ", w_twice.eval())

 
 
