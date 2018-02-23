import tensorflow as tf
sess = tf.InteractiveSession()#类似sess = tf.Session()

x = tf.Variable([1.0, 2.0]) #定义变量
a = tf.constant([3.0, 3.0]) #定义常量

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run() #类似x = tf.constant([1.0, 2.0])

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.subtract(x, a)


print (sub.eval())#类似print(sess.run(sub)) 


#为了便于使用诸如 IPython 之类的 Python 交互环境, 可以使用 InteractiveSession 代替 Session 类,使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话.
