import pymysql


class DbMysqlOp:
    def __init__(self,ip,user,psd,db_name,charset='utf8',pt=3306):
        self.create_dbname(ip, user, psd, db_name, pt)
        # try:
        #     self.conn = pymysql.connect(host=ip,
        #                            port=pt,
        #                            user=user,
        #                            password=psd,
        #                            database=db_name,
        #                            charset=charset,
        #                            )
        #     self.cur = self.conn.cursor()
        # except Exception as E:
        #     raise E
        # else:
        #     print(f'数据库{db_name}连接成功!!!!!!')

    # 创建数据库
    def create_dbname(self,ip,user,psd,db_name,pt):
        try:
            self.conn = pymysql.connect(host=ip, user=user, password=psd, port=pt)
            self.cur = self.conn.cursor()
            is_exist = self.cur.execute(f"SHOW DATABASES LIKE '{db_name}'")
            if not is_exist:
                self.cur.execute(f'CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARSET utf8 COLLATE utf8_general_ci')
                self.conn.commit()
                print(f'数据库{db_name}创建成功!!!!!!')
            else:
                print(f'数据库{db_name}已存在，无需创建!!!!!!')
            self.cur.execute(f'use {db_name}')
            sql = 'select * from steelrecord LIMIT 1000'
            # self.cur.execute('select * from steelrecord LIMIT 1000')
            # res = self.cur.fetchall()
            # print(list(res))
            res = self.ss_bs(sql)
            print(res)
        except Exception as E:
            raise E
        else:
            self.close_()




    # 查询
    def ss_latest_one(self,sql):
        try:
            self.cur.execute(str(sql))
            one_result = self.cur.fetchone()
            self.conn.commit()
            return one_result
        except Exception as E:
            self.close_()
            raise E

    def ss_bs(self,sql):
        try:
            self.cur.execute(str(sql))
            bs_result = self.cur.fetchall()
            self.conn.commit()
            return bs_result
        except Exception as E:
            self.close_()
            raise E

    # 插入
    def insert_(self,sql,param):
        try:
            self.cur.execute(str(sql),param)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close_()
            raise E
        else:
            print(f'插入{len(param)}条数据成功!!!!!!')

    # 删除
    def delete_(self,sql):
        try:
            self.cur.execute(str(sql))
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close_()
            raise E
        else:
            print(f'删除记录成功!!!!!!')

    # 更新
    def updata_(self,sql):
        try:
            self.cur.execute(str(sql))
            self.conn.commit()
        except Exception as E:
            self.close_()
            self.conn.rollback()
            raise E
        else:
            print(f'更新记录成功!!!!!!')

    def close_(self):
        self.cur.close()
        self.conn.close()




if __name__ == '__main__':
    # ip,user,psd,db_name
    DbMysqlOp(
              ip='127.0.0.1',
              user='root',
              psd='nercar',
              db_name='ncdcoldstrip',
              )



