import tensorflow as tf
matrix1 = tf.constant([[3,3]])#构造矩阵(3,3)
matrix2 = tf.constant([[2],[3]])#构造矩阵(2
                                #         3)
product = tf.matmul(matrix1, matrix2)#矩阵乘法 3*2+3*3
sess = tf.Session()
result=sess.run(product)
print(result)
