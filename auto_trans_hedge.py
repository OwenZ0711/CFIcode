import os.path
import subprocess
import datetime
import pandas as pd
import csv
import time


def exec_hedge_trans(trans_cmd, curdate, exec_type, trans_seque):
    exec_file_name = f'{curdate}_{exec_type}_{trans_seque}.txt'
    if not os.path.exists(exec_file_name):
        return ''
    status, log = subprocess.getstatusoutput(f'/home/product/.local/bin/{trans_cmd} -f {exec_file_name}  >> {log_path}')
    pass


def get_hedge_trans_config(curdate, exec_type, exec_path):
    config_dict = {}
    with open(f'/home/product/future_trans_accounts.cfg', 'r') as f:
        for line in f.readlines():
            if not line.strip():
                continue

            paras_name, paras_list = line.replace('(', '').replace(')', '').split('=')
            config_dict[paras_name] = paras_list.split()
    account_2_product = {
        acc: product for acc, product in zip(
            config_dict['automated'] + config_dict['automated_hs'] + config_dict['nonautomated'],
            config_dict['automated_product'] + config_dict['automated_hs_product'] + config_dict['nonautomated_product'])
    }
    df_trans_all = pd.read_csv(f'/home/product/operation/{curdate}_{exec_type}.txt', sep=',', dtype='str')
    df_trans = df_trans_all[df_trans_all['userid'].isin(config_dict['automated'])]
    df_trans_hs = df_trans_all[df_trans_all['userid'].isin(config_dict['automated_hs'])]
    df_trans_non = df_trans_all[~df_trans_all['userid'].isin(config_dict['automated'] + config_dict['automated_hs'])]

    df_trans.to_csv(exec_path + f'{curdate}_{exec_type}_1.txt', sep=',', index=False, quoting=csv.QUOTE_NONE)
    df_trans_hs.to_csv(exec_path + f'{curdate}_{exec_type}_hs_1.txt', sep=',', index=False, quoting=csv.QUOTE_NONE)
    df_trans_non.to_csv(exec_path + f'{curdate}_{exec_type}_non_1.txt', sep=',', index=False, quoting=csv.QUOTE_NONE)

    return account_2_product


def main_exec(curdate, exec_type='hedge_market_close', sleep=180):
    exec_path = f'/home/product/operation_exec/{curdate}/'
    if not os.path.exists(exec_path):
        os.makedirs(exec_path)

    account_2_product = get_hedge_trans_config(curdate, exec_type, exec_path)
    while True:
        trans_seque = 1
        exec_hedge_trans('run', curdate, f'{exec_type}', trans_seque)
        exec_hedge_trans('run_hs', curdate, f'{exec_type}_hs', trans_seque)
        exec_hedge_trans('run', curdate, f'{exec_type}_non', trans_seque)
        time.sleep(sleep)

    
    
if __name__ == '__main__':
    DELTATIME = 120
    curdate = datetime.datetime.now().strftime('%Y%m%d')
    main_exec(curdate)

    """
    功能需求：
        1. flag: hedge_open_index 映射到调拨需求文件： ~/operation/20250415_hedge_market_close.txt 【userid,action,number】
        2. deltatime： 当我的调拨没有成功时，我等待 deltatime 时间后再次发起

        diff_number_dict = {}
        for i in range(0, 10000):
            if  exits ./operation_exec/20241016/20241016_hedge_market_close_flag{ts_flag}.txt
                continue
            else:
                ts_flag = i

        3. 自动化配置文件：future_trans_accounts.cfg 将配置文件里的命令，拆分成2份      更多分  
        while True:
            automated —— /home/product/.local/bin/run -f ./operation_exec/20241016/20241016_hedge_market_close_0.txt, 
            automated_hs —— /home/product/.local/bin/run_hs -f ./operation_exec/20241016/20241016_hedge_market_close_hs_0.txt

            4 串行执行2份文件，日志定位到 
                ./operation_exec/20241016/20241016_hedge_market_close_{ts_flag}.log 读取最后一行，得到结果文件 20250415-150729_result.txt 文件
                ./operation_exec/20241016/20241016_hedge_market_close_hs_{ts_flag}.log 查找 _result.txt 字段，获取结果文件

            5. 存一个flag 文件： 
                ./operation_exec/20241016/20241016_hedge_market_close_flag{ts_flag}.txt
            
            
            failed_continue_list = []
            6. 解析result.txt文件
                
                if True:
                    存到 ./operation_exec/20241016/20241016_hedge_market_close_success_{ts_flag}.txt
                else:
                    资金不足：
                        出金失败：
                            userid action number[可取向下舍入，多少w] > failed_continue_list
                        入金失败：
                            userid action number[银行资金向下舍入，多少w] > failed_continue_list
                            userid action number[diff 金额] > diff_number_dict[_hs][ts_flag]
                    其他原因
                            userid action number  > failed_continue_list

                    diff_number_dict[_hs][ts_flag - 1] > failed_continue_list

            7. ./operation_exec/20241016/20241016_hedge_market_close_success_*.txt 总结历史调拨结果，进度结果

                
            if failed_continue_list > 20241016_hedge_market_close_{ts_flag}.txt
            else: break

            sleep deltatime
            ts_flag += 1

        scp ./auto_hedge_trans.py product:~/


        1. 当调拨的账户数特别多时，卡住
            解决方案，分批





    """