# -*- coding: utf-8 -*-
"""
 Created on 2021/4/4 22:55
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""


# 简单类
class MyClass:
    """ 一个简单的类实例 """
    value = 123

    # self 必备
    def func(self):
        return 'Hello python'


# 类的实例化
my_class = MyClass()
# 访问属性
print(f'MyClass 类的属性 value 为：{my_class.value}')
print(f'MyClass 类的方法 func 输出为：{my_class.func()}')
'''
MyClass 类的属性 value 为：123
MyClass 类的方法 func 输出为：Hello python
'''


# 初始化状态，类的实例化会自动调用 __init__() 方法
class Complex:
    def __init__(self, realpart, imagpart):
        self.re = realpart
        self.im = imagpart


# 类的实例化
my_class = Complex(1, 2)
print(f'my_class 的实部为：{my_class.re}')
print(f'my_class 的虚部为：{my_class.im}')
'''
my_class 的实部为：1
my_class 的虚部为：2
'''


# 私有属性
class People:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性
    __weight = 0

    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print(f'{self.name} 说：我 {self.age} 岁')


# 类的实例化
people = People('Taosy', 30, 65)
people.speak()
'''
Taosy 说：我 30 岁
'''


# 单继承
class Student(People):
    # 继承了 People 的属性和方法
    # 新增的属性
    grade = 0

    def __init__(self, n, a, w, g):
        # 调用父类方法
        People.__init__(self, n, a, w)
        self.grade = g

    # 复写父类的方法
    def speak(self):
        print(f'{self.name} 说：我 {self.age} 岁了，读 {self.grade} 年级')


# 类的实例化
student = Student('Taosy', 10, 65, 3)
student.speak()
'''
Taosy 说：我 10 岁了，读 3 年级
'''


# 多继承
class Speaker:
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print(f'我叫 {self.name}，我演讲的题目是 {self.topic}')


# 继承 Speaker 和 Student
class Sample(Speaker, Student):

    def __init__(self, n, a, w, g, t):
        Student.__init__(self, n, a, w, g)
        Speaker.__init__(self, n, t)


# 类的实例化
sample = Sample('Taosy', 25, 60, 4, 'Python')
# 方法同名，默认调用括号中排前的父类的方法
sample.speak()
'''
我叫 Taosy，我演讲的题目是 Python
'''