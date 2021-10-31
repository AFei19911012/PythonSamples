# -*- coding: utf-8 -*-
"""
 Created on 2021/5/8 14:41
 Filename   : ex_dict.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""

bob = {'name': 'Bob smith', 'age': 42, 'pay': 30000, 'job': 'dev'}
sue = {'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}
sue['pay'] *= 1.1
print(bob['name'])
print(sue['pay'])
print(bob['name'].split()[-1])
'''
Bob smith
44000.0
smith
'''

bob = dict(name='Bob Smith', age=42, pay=30000, job='dev')
sue = {}
sue['name'] = 'Sue Jones'
sue['age'] = 45
sue['pay'] = 40000
sue['job'] = 'hdw'
print(bob)
print(sue)
'''
{'name': 'Bob Smith', 'age': 42, 'pay': 30000, 'job': 'dev'}
{'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}
'''

names = ['name', 'age', 'pay', 'job']
values = ['Sue Jones', 45, 40000, 'hdw']
print(list(zip(names, values)))
'''
[('name', 'Sue Jones'), ('age', 45), ('pay', 40000), ('job', 'hdw')]
'''
sue = dict(zip(names, values))
print(sue)
'''
{'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}
'''

people = [bob, sue]
names = [person['name'] for person in people]
names = list(map((lambda x: x['name']), people))
print(names)
'''
['Bob Smith', 'Sue Jones']
'''

# 类似 SQL 查询
name = [rec['name'] for rec in people if rec['age'] >= 45]
print(name)
'''
['Sue Jones']
'''

# 嵌套
bob2 = {'name': {'first': 'Bob', 'last': 'Smith'},
        'age': 42,
        'job': ['software', 'writing'],
        'pay': (40000, 50000)}
print(bob2['name']['last'])
'''
Smith
'''
for job in bob2['job']:
    print(job)
'''
software
writing
'''

# pickle 存储和获取，类似 json
import pickle

db = {'bob': bob, 'sue': sue}
with open('../people-pickle', 'wb') as dbfile:
    pickle.dump(db, dbfile)

with open('../people-pickle', 'rb') as dbfile:
    db = pickle.load(dbfile)

for key in db:
    print(key, ' =>\n', db[key])
'''
bob  =>
 {'name': 'Bob Smith', 'age': 42, 'pay': 30000, 'job': 'dev'}
sue  =>
 {'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}
'''

# shelve
import shelve

db = shelve.open('../people-shelve')
db['bob'] = bob
db['sue'] = sue
db.close()

db = shelve.open('../people-shelve')
for key in db:
    print(key, ' =>\n', db[key])
'''
bob  =>
 {'name': 'Bob Smith', 'age': 42, 'pay': 30000, 'job': 'dev'}
sue  =>
 {'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}
'''