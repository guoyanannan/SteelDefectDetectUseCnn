import time
import pymssql


def re_print(info):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {info}....................')


class DatabaseSer:
    def __init__(self, db_name, logger_op,db_ip="192.168.0.100", db_user="ARNTUSER", db_pwd="ARNTUSER"):
        self.db_ip = db_ip
        self.db_user = db_user
        self.db_pwd = db_pwd
        self.db_name = db_name
        self.loger = logger_op
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = pymssql.connect(self.db_ip, self.db_user, self.db_pwd, self.db_name)
            self.cursor = self.conn.cursor()
        except Exception as E:
            self.loger.info(E)
            raise Exception(E)

    def search(self, sql):
        try:
            rows = []
            self.cursor.execute(sql)
            self.conn.commit()
            for row in self.cursor:
                rows.append(row)
            return rows
        except Exception as E:
            self.loger.info(E)
            self.close()
            raise Exception(E)

    def update(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close()
            self.loger.info(E)
            raise Exception(E)

    def add(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close()
            self.loger.info(E)
            raise Exception(E)

    def add_bs(self,sql,data_tt):
        try:
            self.cursor.executemany(sql,data_tt)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close()
            self.loger.info(E)
            raise Exception(E)

    def create_procedure(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception as E:
            self.conn.rollback()
            self.close()
            self.loger.info(E)
            raise Exception(E)

    def close(self):
        self.cursor.close()
        self.conn.close()


class SqlSerOp(object):

    def __init__(self, cam_num, db_ip="192.168.0.100", db_user="ARNTUSER", db_pwd="ARNTUSER"):
        self.__dict__.update(locals())

    def create_cam_procedure(self):
        for cam_no in range(1, self.cam_num + 1):
            database_name = "ClientDefectDB{}".format(cam_no)
            db_oper = DatabaseSer(database_name, self.db_ip, self.db_user, self.db_pwd)
            db_oper.connect()

            sql_info = '''
                        IF EXISTS (SELECT * FROM DBO.SYSOBJECTS WHERE ID = OBJECT_ID(N'[dbo].[AddClassifiedDefectToTempTable]') and OBJECTPROPERTY(ID, N'IsProcedure') = 1)
                        DROP PROCEDURE [dbo].[AddClassifiedDefectToTempTable]
                    '''
            db_oper.create_procedure(sql_info)

            sql_info = '''
                    CREATE PROCEDURE AddClassifiedDefectToTempTable
                        @DefectNo int,
                        @SteelNo int,
                        @CameraNo smallint,
                        @ImageIndex smallint,
                        @Class smallint,
                        @Grade smallint,
                        @LeftInImg smallint,
                        @RightInImg smallint,
                        @TopInImg smallint,
                        @BottomInImg smallint,
                        @LeftInSteel smallint,
                        @RightInSteel smallint,
                        @TopInSteel int,
                        @BottomInSteel int,
                        @Area int,
                        @Cycle smallint
                    AS BEGIN
                        insert into DefectTempClassified (DefectNo,SteelNo,CameraNo,ImageIndex,Class,Grade,LeftInImg,RightInImg,TopInImg,BottomInImg,LeftInSteel,RightInSteel,TopInSteel,BottomInSteel,Area,Cycle,ImgData) values (@DefectNo,@SteelNo,@CameraNo,@ImageIndex,@Class,@Grade,@LeftInImg,@RightInImg,@TopInImg,@BottomInImg,@LeftInSteel,@RightInSteel,@TopInSteel,@BottomInSteel,@Area,@Cycle,NULL)
                    END
                    '''
            db_oper.create_procedure(sql_info)
            db_oper.close()

    def write_cam_defects(self, defects_infos):
        cam_info_num = {}
        for cam_no in range(1, self.cam_num + 1):
            database_name = "ClientDefectDB{}".format(cam_no)
            db_oper = DatabaseSer(database_name, self.db_ip,  self.db_user,  self.db_pwd)
            db_oper.connect()
            defects_list = defects_infos[cam_no-1]
            cam_info_num[database_name] = len(defects_list)

            for i in range(0, len(defects_list)):
                db_oper.cursor.callproc('AddClassifiedDefectToTempTable', (
                    int(defects_list[i][0]),
                    int(defects_list[i][1]),
                    int(defects_list[i][2]),
                    int(defects_list[i][3]),
                    int(defects_list[i][4]),
                    int(defects_list[i][5]),
                    int(defects_list[i][6]),
                    int(defects_list[i][7]),
                    int(defects_list[i][8]),
                    int(defects_list[i][9]),
                    int(defects_list[i][10]),
                    int(defects_list[i][11]),
                    int(defects_list[i][12]),
                    int(defects_list[i][13]),
                    int(defects_list[i][14]),
                    int(defects_list[i][15])))
            db_oper.conn.commit()
            db_oper.close()
        re_print(f'当前批次写入各相机数据库数量 {cam_info_num}')

    def get_curr_steel_no(self):
        database_name = "SteelRecord"
        db_oper = DatabaseSer(database_name, self.db_ip,  self.db_user,  self.db_pwd)
        db_oper.connect()
        sql = "select top 1 * from steel order by SequeceNo desc"
        res = db_oper.search(sql)
        for i in range(0, len(res)):
            seq_no = res[i][1]
            steel_no = res[i][3]
            break
        db_oper.close()
        return seq_no, steel_no

    def get_curr_finish_steel_no(self):
        database_name = "SteelRecord"
        db_oper = DatabaseSer(database_name, self.db_ip,  self.db_user,  self.db_pwd)
        db_oper.connect()
        sql = "select top 5 * from steel order by SequeceNo desc"
        res = db_oper.search(sql)
        for i in range(0, len(res)):
            seq_no = res[i][1]
            steel_no = res[i][3]
            steel_bottom_len = res[i][10]
            if int(steel_bottom_len) > 0:
                break
        db_oper.close()
        return seq_no, steel_no

    def get_curr_steel_type(self):
        database_name = "SteelRecord"
        db_oper = DatabaseSer(database_name, self.db_ip,  self.db_user,  self.db_pwd)
        db_oper.connect()
        sql = "select top 1 * from SteelID1 order by No desc"
        res = db_oper.search(sql)
        for i in range(0, len(res)):
            steel_type = res[i][3]
            break
        db_oper.close()
        return steel_type


if __name__ == '__main__':
    sql_ob = SqlSerOp(cam_num=4)
    print(sql_ob.db_ip)
    print(sql_ob.db_user)
    print(sql_ob.db_pwd)