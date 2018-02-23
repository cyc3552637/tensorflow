import math
import tensorflow as tf
sess = tf.Session()
# 假设函数为f(x）=aX+b,a为权值，b为偏值，x为输入的数据，a和b均为变量，初始值为1，初始值可以随便，反正会被优化
# 这里需要注意，初始值有一些值会使得这个函数公式无法达到50，所以基于这个的调整也很难达到50
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
# 输入数据x=5
input = 5
# 将输入变为占位符，这是必须的
input_placeholder = tf.placeholder(dtype=tf.float32)

# f(x)函数表达式，a*x+b
algo = tf.add(tf.multiply(a, input_placeholder), b)

# 我们预期的输出是50
out = 50.
out_placeholder = tf.placeholder(dtype=tf.float32)

# loss=（实际输出-预期输出）^2
loss = tf.square(tf.subtract(algo, out_placeholder))

# 叼炸天东西，tensorflow自己的优化器，可以优化权值和偏值，直到达到预期
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 变量初始化变为常量计算
init = tf.global_variables_initializer()
# 执行初始化
sess.run(init)


for i in range(50):
# 将input值赋给input_placeholder，将out值赋给out_placeholder，根据x值和输出预期，以及loss函数,f(x)函数进行参数优化
    sess.run(train, feed_dict={input_placeholder: input, out_placeholder: out})
# 优化的参数重新赋值给a_val和b_val
    a_val, b_val = (sess.run(a), sess.run(b))
# 把x值赋值进来，配合优化一次以后的a和b，得到函数输出结果
    result = sess.run(algo, feed_dict={input_placeholder: input})
# 多次训练并每次重新计算f(x)函数结果，如果f(x)结果达到到50就直接退出

    print("第" + str(i+1) + "次" + str(a_val) + "*" + str(input) + "+" + str(b_val) + '=' + str(float(result)))
    if math.isclose(result, 50):
        print("break")
        break