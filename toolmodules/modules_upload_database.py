import pandas as pd
from sqlalchemy import create_engine
import pymssql
import sqlalchemy


class UploadDatabase():
    def __init__(self):
        self.dbname_2_engine = {
            'TRDB': create_engine("mssql+pymssql://mluo:Cfi888_@dbs.cfi:1433/TRDB"),
            'WDDB': create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/WDDB"),
            'DataSupply': create_engine("mssql+pymssql://ht:huangtao@dbs.cfi/DataSupply"),
            'JYDB': create_engine("mssql+pymssql://ht:huangtao@dbs.cfi:1433/JYDB?charset=GBK")
        }

        self.dbname_2_connect = {
            'TRDB': pymssql.connect(host='dbs.cfi:1433', user='mluo', password='Cfi888_', database='TRDB'),
            'WDDB': pymssql.connect(host='dbs.cfi', user='ht', password='huangtao', database='WDDB'),
            'DataSupply': pymssql.connect(host='dbs.cfi', user='ht', password='huangtao', database='DataSupply'),
            'JYDB': pymssql.connect(host='dbs.cfi:1433', user='ht', password='huangtao', database='JYDB'),
        }

    def upload_summary(self, file_path, sheet_name, db_name, curdate, date_str):
        if 'csv' == file_path.split('.')[-1]:
            summary = pd.read_csv(file_path)
        elif 'xlsx' == file_path.split('.')[-1]:
            summary = pd.read_excel(file_path)
        elif 'xls' == file_path.split('.')[-1]:
            summary = pd.read_excel(file_path)
        else:
            assert False, '文件类型不被支持!!!'

        if 'Unnamed: 0' in list(summary.columns):
            summary = summary.set_index('Unnamed: 0').reset_index(drop=True)

        sql = "DELETE FROM %s WHERE %s = %s" % (sheet_name, date_str, curdate)

        with self.dbname_2_engine[db_name].connect() as connection:
            result = connection.execute(sqlalchemy.text(sql))
            print(f"{sheet_name}-{curdate} 删除了 {result.rowcount} 条记录!")

        summary.to_sql(name=sheet_name, con=self.dbname_2_engine[db_name], if_exists='append', index=False)

    def create_sheet(self, sheet_name, db_name='DataSupply'):
        db = self.dbname_2_connect[db_name]
        cursor = db.cursor()

        # 是否该表已经存在，若存在则删除
        cursor.execute(f"DROP TABLE IF EXISTS {sheet_name}")

        # 创建表的SQL语句（不唯一）
        # sql = "CREATE TABLE STUDENT(NAME CHAR(20) NOT NULL,AGE INT,SEX CHAR(1),ID CHAR(20))"
        sql = f"CREATE TABLE {sheet_name}"

        cursor.execute(sql)

    def insert_data(self, db_name, sql):
        """
         "INSERT INTO STUDENT(NAME,AGE, SEX, ID) VALUES ('ZYS', 20, 男,666666666)"
        """
        db = self.dbname_2_connect[db_name]
        cursor = db.cursor()

        # try语句防止连接数据库时发生错误
        try:
            # 执行sql语句
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
            print("数据插入成功")

        except:
            # 如果发生错误则回滚
            db.rollback()
            print("数据插入失败")

    def query_data(self, db_name, data_sql):
        engine = self.dbname_2_engine[db_name]
        df = pd.read_sql(data_sql, engine)

        return df

    def delete_data(self, db_name: str, sql: str):
        """
        DELETE
        FROM SHEET_NAME
        WHERE NAME='ZYS'
        """
        db = self.dbname_2_connect[db_name]
        cursor = db.cursor()
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 提交修改
            db.commit()
            print("数据删除成功")
        except:
            # 发生错误时回滚
            db.rollback()
            print("数据删除失败")


if __name__ == '__main__':
    ud = UploadDatabase()

    # ud.upload_summary(
    #     file_path='',
    #     sheet_name='',
    #     db_name=''
    # )