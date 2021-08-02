import private_key as key
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
import pymysql

pymysql.install_as_MySQLdb()


def db_check(db_name):
    db_conn = db_connection()
    sql = f"SELECT 1 FROM Information_schema.SCHEMATA WHERE SCHEMA_NAME = '{db_name}'"
    if not db_conn.cursor().execute(sql):
        db_conn.cursor().execute(f'CREATE DATABASE {db_name}')
        db_conn.commit()
    else:
        pass

def sql_connect(db_name):
    db_url = URL(
        drivername="mysql+mysqldb",
        username=key.db_id,
        password=key.db_passwd,
        host=key.db_ip,
        port=key.db_port,
        database=f'{db_name}'
    )

    db_engine = create_engine(db_url)
    return db_engine


def table_exist(db_name, table_name, engine=None):
    if engine is None:
        engine = sql_connect(db_name)

    sql = f"select 1 from information_schema.tables where table_schema = '{db_name}' and table_name = '{table_name}'"
    rows = engine.execute(sql).fetchall()
    if len(rows) == 1:
        return True
    else:
        return False




def db_connection():
    db_conn = pymysql.connect(
        host=key.db_ip,
        port=int(key.db_port),
        user=key.db_id,
        password=key.db_passwd,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    return db_conn


import MySQLdb


def mysql_cursor(db_name):
    conn = MySQLdb.connect(host=key.db_ip, user=key.db_id, passwd=key.db_passwd, db=f'{db_name}')
    cursor = conn.cursor()
    return cursor


def mysql_conn(db_name):
    conn = MySQLdb.connect(host=key.db_ip, user=key.db_id, passwd=key.db_passwd, db=f'{db_name}')
    return conn

    # Query 문이 SELECT 같은 경우 fetchall / fetchone 으로 데이터 뽑기.
    # CREATE 같이 영향을 주는 경우, commit 필요
# https://m.blog.naver.com/PostView.nhn?blogId=altmshfkgudtjr&logNo=221578178061&categoryNo=27&proxyReferer=https:%2F%2Fwww.google.com%2F

