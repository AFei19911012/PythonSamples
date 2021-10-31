# -*- coding: utf-8 -*-
"""
Created on 2021/3/27 23:12
author: ff_wang
E-mail: 1105936347@qq.com
Zhihu: https://www.zhihu.com/people/1105936347
Github: https://github.com/AFei19911012
"""

# 转义字符
str1 = 'C:\\test\\file'
str2 = r'C:\test\file'
print(str1)
print(str2)
'''
C:\test\file
'''
str1 = r'C:\test\file' + '\\'
str2 = 'C:\\test\\file\\'
print(str1)
print(str2)
'''
C:\test\file\
'''

# 字符串访问
str1 = 'Hello Python'
print('str1[2] = ', str1[2])
'''
str1[2] =  l
'''

# 多行字符串
text = """
骐骥一跃，不能十步；
驽马十驾，功在不舍；
锲而舍之，朽木不折；
锲而不舍，金石可镂。
"""
text = '''
骐骥一跃，不能十步；
驽马十驾，功在不舍；
锲而舍之，朽木不折；
锲而不舍，金石可镂。
'''
print(text)
'''
骐骥一跃，不能十步；
驽马十驾，功在不舍；
锲而舍之，朽木不折；
锲而不舍，金石可镂。
'''
errHTML = '''
<HTML><HEAD><TITLE>
Friends CGI Demo</TITLE></HEAD>
<BODY><H3>ERROR</H3>
<B>%s</B><P>
<FORM><INPUT TYPE=button VALUE=Back
ONCLICK="window.history.back()"></FORM>
</BODY></HTML>
'''
print(errHTML)
'''
<HTML><HEAD><TITLE>
Friends CGI Demo</TITLE></HEAD>
<BODY><H3>ERROR</H3>
<B>%s</B><P>
<FORM><INPUT TYPE=button VALUE=Back
ONCLICK="window.history.back()"></FORM>
</BODY></HTML>
'''

# 格式化字符串，以 f 打头，字符串中的表达式用 {} 包起来，它会将变量或表达式计算后的值替换进去
name = 'ff_wang'
url = 'www.zhihu.com/people/1105936347'
print(f'{name}: {url}')
'''
ff_wang: www.zhihu.com/people/1105936347
'''
x = 1
print(f'{x + 1}')
'''
2
'''
print(f'{x + 1 = }')
'''
x + 1 = 2
'''
# 5 个占位符，小数点后 1 位
print('%5.1f' % 123.456)
'''
123.5
'''
# 5 个占位符
print('%5d' % 5)
'''
    5
'''
# 左对齐
print('%-5d' % 5)
'''
5    
'''

# 字符串内建函数
# capitalize() 首字母大写，其余字母小写
str1 = 'hello Python'
print(str1.capitalize())
'''
Hello python
'''
# lower() 大写转小写
# upper() 小写转大写
# swapcase() 大小写互换
str1 = 'Love'
print(str1.lower())
'''
love
'''
print(str1.upper())
'''
LOVE
'''
print(str1.swapcase())
'''
lOVE
'''
# 指定宽度居中，空格填充
str1 = 'hello Python'
print(str1.center(20))
'''
    hello Python    
'''
# 指定宽度左对齐
print(str1.ljust(20))
'''
hello Python        
'''
# 指定宽度右对齐
print(str1.rjust(20))
'''
        hello Python
'''
# 指定宽度居中，* 号填充
print(str1.center(20, '*'))
'''
****hello Python****
'''

# count() 指定字符串 sub 出现次数，可以指定范围
str1 = 'Hello python'
print(str1.count('l'))
'''
2
'''

# startswith() endswith() 字符串是否以 sub 开始、结尾
str1 = 'Hello python'
print(str1.startswith('He'))
'''
True
'''
print(str1.endswith('o'))
'''
False
'''

# find() 返回字符串中 sub 出现的下标，可指定范围
# rfind() 从右边查找
# index() 指定 sub 不存在报错
str1 = 'Hello python'
print(str1.find('l'))
'''
2
'''
print(str1.find('ll'))
'''
2
'''
# 指定 sub 不存在，返回 -1
print(str1.find('P'))
'''
-1
'''

# isalpha() 只包含字母
# isdecimal() 只包含十进制数字
# isdigit() 只包含数字
# isnumeric() 只包含数字字符
# islower() 是否全部小写
# isupper() 是否全部大写
# isspace() 只包含空格
# istitle() 是否是标题化（首字母大写其余都是小写）

# 用指定字符串分隔 sub
str1 = 'abc'
print(str1.join('123'))
'''
1abc2abc3
'''

# 删除左边、右边、左右两边的空格
str1 = '  a b  c '
print(str1.lstrip())
'''
a b  c 
'''
print(str1.rstrip())
'''
  a b  c
'''
print(str1.strip())
'''
a b  c
'''

# zfill() 指定长度字符串，右对齐，前面填充 0
str1 = 'love'
print(str1.zfill(10))
'''
000000love
'''
print(str1.zfill(2))
'''
love
'''

# split() 分割字符串，返回结果列表
str1 = 'Hello Python'
print(str1.split())
'''
['Hello', 'Python']
'''
print(str1.split('P'))
'''
['Hello ', 'ython']
'''

# replace(old, new) 替换，可指定替换的次数
str1 = 'aaabcdef'
print(str1.replace('a', 'B'))
'''
BBBbcdef
'''
print(str1.replace('a', 'B', 2))
'''
BBabcdef
'''
