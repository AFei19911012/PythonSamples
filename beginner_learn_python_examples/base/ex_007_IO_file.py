# -*- coding: utf-8 -*-
"""
 Created on 2021/4/4 18:08
 Author: ff_wang
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""

# 格式化字符串输出
for x in range(1, 11):
    print('{:2d} {:3d} {:4d}'.format(x, x*x, x*x*x))

for x in range(1, 11):
    print(f'{x:2d} {x*x:3d} {x*x*x:4d}')
'''
 1   1    1
 2   4    8
 3   9   27
 4  16   64
 5  25  125
 6  36  216
 7  49  343
 8  64  512
 9  81  729
10 100 1000
'''
# 键盘输入
# input 读取一行，返回字符串
str1 = input('input: ')
print(type(str1))
'''
input: 25
<class 'str'>
'''

# 打开和关闭文件
# 踩坑 w 模式打开
fo = open('../file.txt', 'r')
print(f'文件名：{fo.name}')
'''
文件名：file.txt
'''
# 关闭文件
fo.close()

# 换一种简便的写法
with open('../file.txt', 'r') as fid:
    print(f'文件名：{fid.name}')
'''
文件名：file.txt
'''

# read() 从打开的文件中读取一个字符串
with open('../file.txt', 'r') as fid:
    print(fid.read())
    # 一定要加这一句，将游标移动到文件开头
    fid.seek(0)
    print(fid.read(10))
'''
Zhihu : https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
Zhihu : ht
'''
# readline() 读取一行，返回一个空字符串，说明已经到了最后一行
with open('../file.txt', 'r') as fid:
    fid.seek(0)
    print(fid.readline())
    fid.seek(0)
    print(fid.readline(10))
'''
Zhihu : https://www.zhihu.com/people/1105936347

Zhihu : ht
'''

# write() 将字符串写入到打开的文件
with open('../file.txt', 'w') as fid:
    fid.write('Zhihu : https://www.zhihu.com/people/1105936347\n'
              'Github: https://github.com/AFei19911012')
# file.txt 内容
'''
Zhihu : https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
'''

# 重命名
# os.rename(current_file_name, new_file_name)
# 删除文件
# os.remove(file_name)
# 新建文件夹
# os.mkdir("newdir")
