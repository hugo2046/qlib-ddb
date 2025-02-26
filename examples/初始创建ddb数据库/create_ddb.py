'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 15:47:02
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-26 17:06:02
Description: 测试ddb操作及相关清除创建
'''
import sys
from pathlib import Path
PROJECT_ROOT =str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_ROOT)

import fire
from qlib.data.backend.ddb_qlib.ddb_operator import create_feature_daily_table, create_calendar_table, create_instrument_table, clean_qlib_db

ddb_uri: str = "dolphindb://admin:123456@hostlocal:8848"

def main(ddb_uri: str=ddb_uri,clean_db: bool = False):
    if clean_db:
        print("清除qlib数据库")
        # NOTE: 清空qlib数据库！！！！
        clean_qlib_db(ddb_uri)

    # 创建feature_daily表
    create_feature_daily_table(ddb_uri)
    # 创建calendar表
    create_calendar_table(ddb_uri)
    # 创建instrument表
    create_instrument_table(ddb_uri)
    

if __name__ == "__main__":


    fire.Fire(main)
    
    
    
    
