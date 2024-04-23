import pymysql

host = "localhost"
user = "root"
password = "1021"
charset = "utf8"
port = 3306

# 建立与MySQL服务器的连接
connection = pymysql.connect(
    host=host,
    user=user,
    password=password,
    charset=charset,
    port=port
)

# 创建一个新的数据库
new_database = "Factory"
create_database_query = f"CREATE DATABASE IF NOT EXISTS {new_database};"

# 创建游标对象
cursor = connection.cursor()
db = pymysql.connect(host=host, user=user, password=password, charset=charset, port=port)

# 执行创建数据库的SQL语句
cursor.execute(create_database_query)

database = new_database
db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)

# 创建User表
use_database_query = f"USE {new_database};"
create_database_query = """
    CREATE TABLE IF NOT EXISTS Worker (
        `id` INT AUTO_INCREMENT PRIMARY KEY,
        `workername` VARCHAR(255) DEFAULT '',
        `password` VARCHAR(255) DEFAULT '',
        `starttime` VARCHAR(255) DEFAULT '',
        `work` VARCHAR(255) DEFAULT ''
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
"""
create_records_table_query = """
    CREATE TABLE IF NOT EXISTS Records (
        `id` INT AUTO_INCREMENT PRIMARY KEY,
        `errortime` VARCHAR(255) DEFAULT '',
        `workername` VARCHAR(255) DEFAULT '',
        `errortype` VARCHAR(255) DEFAULT ''
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
"""

# 执行切换数据库和创建表的SQL语句
cursor.execute(use_database_query)
cursor.execute(create_database_query)
cursor.execute(create_records_table_query)
# 提交更改并关闭连接
connection.commit()
connection.close()

print(f"Database '{new_database}' and table 'User' created successfully.")
