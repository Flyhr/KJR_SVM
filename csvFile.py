import pandas as pd
import mysql.connector

# 建立数据库连接
conn = mysql.connector.connect(
    host="localhost",
    port = "3306",
    user="root",
    password="123456",
    database="ry-vue"
)

# 编写SQL查询语句，选择你需要的表和字段
query = "SELECT * FROM integrated_data_table"

# 使用pandas读取数据库数据
df = pd.read_sql(query, conn)

# 将数据保存为CSV文件
df.to_csv('integrated_data.csv', index=False, encoding='utf-8')

# 关闭数据库连接
conn.close()