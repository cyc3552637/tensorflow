import tensorflow as tf
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")#定义变量

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)#定义常量
new_value = tf.add(state, one)#递增，state的值+1
update = tf.assign(state, new_value)#替换，state新值替换旧值


# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print("init",sess.run(state))
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

# 变量经过init op以后初始化，然后得到初始化的值
# update op则是运行update的操作，而update调用new_value,new_value调用one   
