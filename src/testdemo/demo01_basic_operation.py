#coding:UTF-8
'''
Created on 2017年1月27日
@author: Ayumi Phoenix

了解TF的基本数据类型和运算方法, 用计算图谱来实现一些简单的函数

基本流程为：
1. 定义计算图 tf.Graph
    1.1 在计算图中添加变量并定义初始化值 tf.Variable
    1.2 在计算图中添加常量并设定常量值 tf.constant
    1.3 在计算图中添加占位符, 用于从外部端口获得值
    1.4 在任何时候，对节点使用print(), 可以获得类型，形状等基本信息

2. 在计算图中，用上述op定义计算式

3. 设置会话 tf.Session
    3.1 '初始化'系统所有变量 tf.global_variable_initializer
    3.2 使用sess.run执行计算图中的计算式
    3.3 run函数可能要接受feed_dict
    3.4 run函数可以同时执行多个节点的计算，返回对应的ndarray值
    3.5 也可以在对应的计算节点在响应eval(session=mySess) 来获得ndarray值
    3.6 要一次性获得多个值，使用run
        eg. vf1,vf2 = mySess.run(f1,f2, [feed_dict={...}])
        若只计算一个节点的值，可以只使用eval
        eg. vf1 = f1.eval([feed_dict={...}])
''' 
import tensorflow as tf

def basic_operation():
    print("basic opration")
    myGraph = tf.Graph()
    with myGraph.as_default():
        value1 = tf.constant([1,2])
        value2 = tf.Variable([3,4])
        mul = value1 * value2 + 2   # 按元素相乘
        print(value1,value2,mul,sep='\n')

    with tf.Session(graph=myGraph) as mySess:
        tf.global_variables_initializer().run()
        # default: tf.global_variables_initializer().run(session=mySess)
        print('mySess.run(mul)) = ', mySess.run(mul))
        print('or mul.eval()) = ', mul.eval())
        # default mul.eval(session=mySess)
    print("\n")

basic_operation()

def load_from_remote():
    return [x for x in range(10)]

def load_partial(value, step):
    """ 使用迭代器从value中每次获得长度为step的数据 """
    index = 0
    while index < len(value):
        yield value[index:index+step]
        index += step
    return

def use_placeholder():
    print("use placeholder to fetch outside data")
    myGraph = tf.Graph()
    with myGraph.as_default():
        value1 = tf.placeholder(dtype=tf.float64)
        print(value1)
        value2 = tf.constant([1, -1], dtype=tf.float64)
        mul = value1 * value2
    
    with tf.Session(graph=myGraph) as mySess:
        tf.global_variables_initializer().run()
        value = load_from_remote()
        for partialValue in load_partial(value, 2):
            evalResult = mySess.run(mul, feed_dict={value1:partialValue})
            # option: evalResult = mul.eval(feed_dict={value1:partialValue})
            print("(value1 * value2)=",evalResult)
    print("\n")

use_placeholder()


def main():
    pass
    basic_operation()
    use_placeholder()
    
if __name__=="__main__":
    pass
#     main()

"""
Note:
1. 4个重要的类型
    @Variable    计算图谱中的变量
    @Tensor      一个多维矩阵，带有很多方法
    @Graph       一个计算图谱
    @Session     用来运行一个计算图谱

2. 迭代器
    在一个 generator function 中，如果没有 return，则默认执行至函数完毕，
    如果在执行过程中 return，则直接抛出 StopIteration 终止迭代
"""


    