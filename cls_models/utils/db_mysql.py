import pymysql
from cls_models.utils.common_oper import re_print


class DbMysqlOp:
    def __init__(self,ip,user,psd,db_name,logs_oper,charset='utf8',pt=3306):
        self.ip = ip
        self.user = user
        self.psd = psd
        self.db_name = db_name
        self.loger = logs_oper
        self.pt = pt
        self.charset = charset
        self.create_dbname()
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
    def create_dbname(self):
        try:
            self.conn = pymysql.connect(host=self.ip, user=self.user, password=self.psd, port=self.pt, charset=self.charset,)
            self.cur = self.conn.cursor()
            is_exist = self.cur.execute(f"SHOW DATABASES LIKE '{self.db_name}'")
            if not is_exist:
                self.cur.execute(f'CREATE DATABASE IF NOT EXISTS {self.db_name} DEFAULT CHARSET utf8 COLLATE utf8_general_ci')
                self.loger.info(f'数据库{self.db_name}创建成功')
            else:
                # self.loger.info(f'数据库{self.db_name}已存在，无需创建')
                pass
            self.cur.execute(f'use {self.db_name}')
            self.conn.commit()
        except Exception as E:
            self.loger.info(E)
            raise

    # 创建表
    def create_dbtabel(self, tabel_name):
        try:
            sql = f"SHOW TABLES like '{tabel_name}'"
            is_exist = self.cur.execute(sql)
            if not is_exist:
                sql_cl = f"CREATE TABLE {tabel_name} (\
                        id INT(11) NOT NULL AUTO_INCREMENT,\
                        defectID INT(11) NOT NULL,\
                        camNo INT(11) NULL DEFAULT NULL,\
                        seqNo INT(11) NOT NULL,\
                        imgIndex INT(11) NULL DEFAULT NULL,\
                        defectClass INT(11) NULL DEFAULT NULL,\
                        leftInImg INT(11) NULL DEFAULT NULL,\
                        rightInImg INT(11) NULL DEFAULT NULL,\
                        topInImg INT(11) NULL DEFAULT NULL,\
                        bottomInImg INT(11) NULL DEFAULT NULL,\
                        leftInSrcImg INT(11) NULL DEFAULT NULL,\
                        rightInSrcImg INT(11) NULL DEFAULT NULL,\
                        topInSrcImg INT(11) NULL DEFAULT NULL,\
                        bottomInSrcImg INT(11) NULL DEFAULT NULL,\
                        leftInObj INT(11) NULL DEFAULT NULL,\
                        rightInObj INT(11) NULL DEFAULT NULL,\
                        topInObj INT(11) NULL DEFAULT NULL,\
                        bottomInObj INT(11) NULL DEFAULT NULL,\
                        grade TINYINT(4) NULL DEFAULT NULL,\
                        area INT(11) NULL DEFAULT NULL,\
                        leftToEdge INT(11) NULL DEFAULT NULL,\
                        rightToEdge INT(11) NULL DEFAULT NULL,\
                        cycle INT(11) NULL DEFAULT '0',\
                        PRIMARY KEY (`id`) USING BTREE,\
                        INDEX `seqNo` (`seqNo`) USING BTREE\
                    )\
                    COLLATE='utf8_general_ci'\
                    ENGINE=InnoDB\
                    "
                self.cur.execute(sql_cl)
                self.loger.info(f'{self.db_name}.{tabel_name}创建成功')
            else:
                self.loger.info(f'{self.db_name}.{tabel_name}已存在，无需创建')
            self.conn.commit()
        except Exception as E:
            self.close_()
            self.loger.info(E)
            raise

    # 查询
    def ss_latest_one(self,sql):
        try:
            self.cur.execute(str(sql))
            one_result = self.cur.fetchone()
            self.conn.commit()
            return one_result
        except Exception as E:
            self.close_()
            self.loger.info(E)
            raise

    def ss_bs(self,sql):
        try:
            self.cur.execute(str(sql))
            bs_result = self.cur.fetchall()
            self.conn.commit()
            return bs_result
        except Exception as E:
            self.close_()
            self.loger.info(E)
            raise

    # 插入
    def insert_(self,sql,table_name,param):
        try:
            self.cur.executemany(str(sql),param)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close_()
            self.loger.info(E)
            raise
        else:
            re_print(f'表[{table_name}]插入[{len(param)}]条数据成功')

    # 删除
    def delete_(self,sql):
        try:
            self.cur.execute(str(sql))
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close_()
            self.loger.info(E)
            raise
        else:
            re_print(f'删除记录成功')
    # 更新
    def updata_(self,sql):
        try:
            self.cur.execute(str(sql))
            self.conn.commit()
        except Exception as E:
            self.close_()
            self.conn.rollback()
            self.loger.info(E)
            raise
        else:
            re_print(f'更新记录成功')

    def close_(self):
        self.cur.close()
        self.conn.close()


def get_camdefect_no(ip,user,psd,dbname,tables:tuple,logs_oper):
    db_oper = DbMysqlOp(
        ip=ip,
        user=user,
        psd=psd,
        db_name=dbname,
        logs_oper=logs_oper,
    )
    defects_info = {}
    for i in range(1,len(tables)+1):
        defects_info[str(i)] = 0
    for tabel_name in tables:
        sql_cam = f'SELECT * FROM {tabel_name} ORDER BY ID DESC LIMIT 0,1'
        infos = db_oper.ss_latest_one(sql_cam)
        if infos:
            defect_id,cam_id = infos[1:3]
            defects_info[str(cam_id)]=defect_id
    db_oper.close_()
    return defects_info


def check_temp_db(ip,user,psd,dbname,tables:tuple,logs_oper):
    db_oper = DbMysqlOp(
        ip=ip,
        user=user,
        psd=psd,
        db_name=dbname,
        logs_oper=logs_oper,
    )
    for tabel_name in tables:
        db_oper.create_dbtabel(tabel_name)
    db_oper.close_()


def get_dbop(ip,user,psd,dbname,logs_oper):
    db_oper = DbMysqlOp(
        ip=ip,
        user=user,
        psd=psd,
        db_name=dbname,
        logs_oper=logs_oper,
    )
    return db_oper


def write_defect_to_table(oper,defect_info,tables:tuple):
    for i in range(0,len(defect_info)):
        insert_sql = f'INSERT INTO {tables[i]} (defectID,camNo,seqNo,imgIndex,defectClass,' \
                     f'leftInImg,rightInImg,topInImg,bottomInImg,' \
                     f'leftInSrcImg,rightInSrcImg,topInSrcImg,bottomInSrcImg,' \
                     f'leftInObj,rightInObj,topInObj,bottomInObj,' \
                     f'grade,area,leftToEdge,rightToEdge,cycle) VALUES(%s,%s,%s,%s,%s,' \
                     f'%s,%s,%s,%s,' \
                     f'%s,%s,%s,%s,' \
                     f'%s,%s,%s,%s,' \
                     f'%s,%s,%s,%s,%s)'
        oper.insert_(insert_sql,tables[i],defect_info[i])


if __name__ == '__main__':
    # ip,user,psd,db_name
    dop = DbMysqlOp(
              ip='127.0.0.1',
              user='root',
              psd='nercar',
              # db_name='ncdcoldstrip',
              db_name='temp',
              )

    dop.create_dbtabel('tempcam1')



