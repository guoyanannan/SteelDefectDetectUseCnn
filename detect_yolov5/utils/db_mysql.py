import pymysql


class DbMysqlOp:
    def __init__(self,ip,user,psd,db_name,charset='utf8',pt=3306):
        try:
            self.conn = pymysql.connect(host=ip,
                                   port=pt,
                                   user=user,
                                   password=psd,
                                   database=db_name,
                                   charset=charset,
                                   )
            self.cur = self.conn.cursor()
        except Exception as E:
            raise E
        else:
            print(f'数据库{db_name}连接成功!!!!!!')



    # 创建表
    def create_tabel_(self,sql):
        pass

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
            bs_result = self.cur.execute(str(sql)).fetchall()
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
    while 1:
        print(121323355)



