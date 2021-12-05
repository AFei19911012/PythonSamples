# -*- coding: utf-8 -*-
"""
 Created on 2021/6/29 13:01
 Filename   : pandas_excel_read_write.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: 利用 Pandas 完成表格读取、表格取数、表格合并
"""

# =======================================================
import pandas as pd


def pandas_excel():
    """ 利用 Pandas 完成表格读取、表格取数、表格合并、表格写出 """
    """ sheet_name=0 表示读取第一个 sheet 表（默认），等于 1 表示读取第二个 sheet 表 """
    """ sheet_name="sheet1" 读取指定名称的 sheet """
    df = pd.read_excel('data/test.xlsx', sheet_name=0)
    print(df)
    print('---')

    """ header=None 针对没有标题行的 excel 文件，系统按照序号做标题 """
    """ header=1 指定第一行为标题行（默认） """
    df = pd.read_excel('data/test.xlsx', sheet_name=0, header=None)
    print(df)
    print('---')

    """ usecols=None 选择所有列（默认） """
    df = pd.read_excel('data/test.xlsx', sheet_name=0, usecols=[1, 2])
    df = pd.read_excel('data/test.xlsx', sheet_name=0, usecols="A:B")
    print(df)
    print('---')

    """ 设置标题行 """
    name_list = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']
    df = pd.read_excel('data/test.xlsx', sheet_name=0, names=name_list)
    print(df)
    print('---')

    """ 访问数据 切片、索引 """
    """ 按列访问 """
    df = pd.read_excel('data/test.xlsx')
    print(df['最高报价'])
    print('---')
    """ 这个写法 """
    print(df.最高报价)
    print('---')
    print(df[['最高报价', '下浮率']])
    print('---')
    """ 按行访问 """
    """ 位置索引 """
    print(df.iloc[0])
    print('---')
    """ 标签索引 """
    print(df.loc[0])
    print('---')

    """ 数据拼接 """
    df1 = pd.read_excel('data/test.xlsx')
    df2 = pd.read_excel('data/test.xlsx')
    df = pd.concat([df1, df2], ignore_index=True)
    print(df)
    print('---')
    """ 去重 """
    df.drop_duplicates()
    print(df)

    """ 数据写出 """
    df = pd.concat([df1, df2], ignore_index=True)
    """ index=None 去掉默认索引列 """
    df.to_excel(excel_writer='data/to_excel.xlsx', sheet_name='to_excel', index=None)
    """ ExcelWriter """
    with pd.ExcelWriter('data/excel_write.xlsx', datetime_format='YYYY_MM_DD') as xlsx:
        df1.to_excel(excel_writer=xlsx, sheet_name='df1', index=None)
        df2.to_excel(excel_writer=xlsx, sheet_name='df2', index=None)


if __name__ == '__main__':
    pandas_excel()
