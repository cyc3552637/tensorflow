import tensorflow as tf

#建模，必须跟原来模型类型完全一样，数据无所谓，最后会使用恢复的变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")

biases = tf.Variable(tf.zeros([200]), name="biases")

w2 = tf.Variable(weights.initialized_value(), name="w2")

w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()#恢复变量的时候不需要这个

 #恢复变量
with tf.Session() as sess:
  sess.run(init_op)#建模生成的数据
  print ("weightsmodel = ", weights.eval())
  print ("biasesmodel = ", biases.eval())
  print ("w2model = ", w2.eval())
  print ("w_twicemodel = ", w_twice.eval())
    
  saver.restore(sess, "D:\pydemo\data\model.ckpt")#恢复变量生成的数据
  print ("Model restored.")
  print ("weights = ", weights.eval())
  print ("biases = ", biases.eval())
  print ("w2 = ", w2.eval())
  print ("w_twice = ", w_twice.eval())
