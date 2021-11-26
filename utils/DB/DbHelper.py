import pymssql


# 获取数据库链接信息：对应数据库服务器 数据库用户 数据库密码 数据库名称
def get_connection_string():
    # return ['SHELI-LAPTOP\SQLEXPRESS', 'sa', 'Shris2020', 'CenterController']
    return ['10.58.37.201', 'sa', 'Shris2020', 'CenterController']
    # return ['118.195.161.38', 'sa', 'Hdxk12345', 'CenterController']


def select_sql(sql):
    connect_string = get_connection_string()
    connect = pymssql.connect(server=connect_string[0],
                              user=connect_string[1],
                              password=connect_string[2],
                              database=connect_string[3])  # 建立连接
    results = None
    if connect:
        cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
        try:
            cursor.execute(sql)  # 执行sql语句
            results = cursor.fetchall()
        except Exception as e:
            print("%s execute error" % sql)
        finally:
            cursor.close()
            connect.close()
    return results


def insert_sql(sql):
    connect_string = get_connection_string()
    connect = pymssql.connect(server=connect_string[0], user=connect_string[1], password=connect_string[2], database=connect_string[3])  # 建立连接
    if connect:
        cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
        try:
            cursor.execute(sql)  # 执行sql语句
            connect.commit()
        except Exception:
            connect.rollback()
            print("%s execute error" % sql)
        finally:
            cursor.close()
            connect.close()


def update_sql(sql):
    connect_string = get_connection_string()
    connect = pymssql.connect(server=connect_string[0], user=connect_string[1], password=connect_string[2], database=connect_string[3])  # 建立连接
    if connect:
        cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
        try:
            cursor.execute(sql)  # 执行sql语句
            connect.commit()
        except Exception:
            connect.rollback()
            print("%s execute error" % sql)
        finally:
            cursor.close()
            connect.close()


def delete_sql(sql):
    connect_string = get_connection_string()
    connect = pymssql.connect(server=connect_string[0], user=connect_string[1], password=connect_string[2], database=connect_string[3])  # 建立连接
    if connect:
        cursor = connect.cursor()  # 创建一个游标对象,python里的sql语句都要通过cursor来执行
        try:
            cursor.execute(sql)  # 执行sql语句
            connect.commit()
        except Exception:
            connect.rollback()
            print("%s execute error" % sql)
        finally:
            cursor.close()
            connect.close()