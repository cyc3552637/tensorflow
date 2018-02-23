import tensorflow as tf

#建模，必须跟原来模型类型完全一样，数据无所谓，最后会使用恢复的变量
myweights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="myweights")

myw2 = tf.Variable(myweights.initialized_value(), name="myw2")


saver = tf.train.Saver({"weights":myweights,"w2":myw2})#恢复指定变量，前一个是存储的变量名称，后一个是本机调用的变量的名称


 #恢复变量
with tf.Session() as sess: 
  saver.restore(sess, "D:\pydemo\data\model.ckpt")#恢复变量生成的数据
  print ("Model restored.")
  print ("myweights = ", myweights.eval())
  print ("myw2 = ", myw2.eval())

