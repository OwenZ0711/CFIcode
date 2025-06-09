# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:40:51 2025

@author: zhangziyao
"""
import os.path
import subprocess
import datetime
import pandas as pd
import csv
import time
from collections import deque


def find_last_result_line(log_file_path):
    """查找log文件里面后缀等于 `_result.txt`的一行, 测试过了

    Args:
        log_file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    last_result_line = None
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip().endswith('_result.txt'):
                    last_result_line = line.strip()
        return last_result_line
    except FileNotFoundError:
        return f"{log_file_path}文件未找到"
    except Exception as e:
        return f"读取文件时出错: {e}"

def exec_hedge_trans(trans_cmd, curdate, exec_type, trans_seque,process_acc,il):
    """
    首先在所有调拨里面筛选出要执行的操作
    然后把需要的操作存为  ./home/product/operation/ 路径下的{curdate}_{exec_type}_{trans_seque}.txt
    传参新建文件并执行获得回执log并储存
    在log中寻找成交细节文件名
    保存成交细节文件名至./operation_exec/{curdate}/{curdate}_{exec_type}_{trans_seque}.txt 便于记录
    """
    df_trans_all = pd.read_csv(f'./home/product/operation/{curdate}_{exec_type}.txt', sep=',', dtype='str')
    prev_failed = f'{curdate}_{exec_type}_{trans_seque}.txt'
    exec_file_name = f"{curdate}_{exec_type}_{trans_seque}_{il}.txt"
    if os.path.exists(os.path.join("./home/product/operation/",prev_failed)):
        #上次执行有错
        df_trans_change = pd.read_csv(f'./home/product/operation/{curdate}_{exec_type}_{trans_seque}.txt', sep=',', dtype='str')
        df_trans_curr = df_trans_all[df_trans_all["userid"] in process_acc.keys()]
        for idx in df_trans_curr.index:
            if df_trans_curr.iloc[idx, "userid"] in list(df_trans_change["userid"]):
                df_trans_curr.iloc[idx, "number"] = df_trans_change[df_trans_change["userid"] == df_trans_curr.iloc[idx, "userid"]]["number"].iloc[0]
    else:
        df_trans_curr = df_trans_all[df_trans_all["userid"].isin(process_acc.keys())]
    
    df_trans_curr.to_csv(f"./home/product/operation/{curdate}_{exec_type}_{trans_seque}_{il}.txt", sep=',', index=False) #这个路径对吗
    
    # 执行并生成log files存放在logfile里面
    status, log = subprocess.getstatusoutput(f'/home/product/.local/bin/{trans_cmd} -f {exec_file_name}  >> {f"~/log_path/{curdate}_{exec_type}_{trans_seque}.log"}')
    result_file_path = find_last_result_line(f"~/log_path/{curdate}_{exec_type}_{trans_seque}.log")
    with open(f"~/operation_exec/{curdate}/{curdate}_{exec_type}_{trans_seque}.txt","w") as file:
        file.write(result_file_path)
    
    return result_file_path

def get_hedge_trans_config(curdate, exec_type, exec_path,ts):
    """获取一个账户：策略的字典

    Args:
        curdate (_type_): str
        exec_type (_type_): str
        exec_path (_type_): str

    Returns:
        _type_: 字典{账户：策略} (auto,hs,nonauto)
    """
    config_dict = {} #存放每种account里面的accountid
    with open(f'~/future_trans_accounts.cfg', 'r') as f:
        for line in f.readlines():
            if not line.strip(): # 如果是空行跳过
                continue

            paras_name, paras_list = line.replace('(', '').replace(')', '').split('=')
            config_dict[paras_name] = paras_list.split()
    account_2_product = {
        acc: product for acc, product in zip(
            config_dict['automated'] + config_dict['automated_hs'] + config_dict['nonautomated'],
            config_dict['automated_product'] + config_dict['automated_hs_product'] + config_dict['nonautomated_product'])
    } 
    #每个账户对应着一个产品（一个产品不止对应一个账户）
    df_trans_all = pd.read_csv(f'~/operation/{curdate}_{exec_type}.txt', sep=',', dtype='str') 
    df_trans = df_trans_all[df_trans_all['userid'].isin(config_dict['automated'])]
    df_trans_hs = df_trans_all[df_trans_all['userid'].isin(config_dict['automated_hs'])]
    df_trans_non = df_trans_all[~df_trans_all['userid'].isin(config_dict['automated'] + config_dict['automated_hs'])]
    df_trans.to_csv(exec_path + f'{curdate}_{exec_type}.txt', sep=',', index=False, quoting=csv.QUOTE_NONE) #手动
    df_trans_hs.to_csv(exec_path + f'{curdate}_{exec_type}_hs.txt', sep=',', index=False, quoting=csv.QUOTE_NONE) #恒生
    df_trans_non.to_csv(exec_path + f'{curdate}_{exec_type}_non.txt', sep=',', index=False, quoting=csv.QUOTE_NONE) #非手动
    
    account_2_product_auto = {acc:account_2_product[acc] for acc in df_trans["userid"].to_list()}
    account_2_product_hs = {acc:account_2_product[acc] for acc in df_trans_hs["userid"].to_list()}
    account_2_product_unauto = {acc:account_2_product[acc] for acc in df_trans_non["userid"].to_list()}
    
    return account_2_product_auto, account_2_product_hs, account_2_product_unauto

# def append_df_to_txt(df, file_path):
#     # 没有或者空文件
#     if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#         # 写入columns
#         df.to_csv(file_path, sep = ",", mode='a', header=True, index=False)
#     else:
#         # 文件已存在而且有数据就不加入columns
#         df.to_csv(file_path, sep = ",", mode='a', header=False, index=False)

def process_errors(df):
    failed_continue_list = {"userid":[], "action":[], "number":[]}

    for i in df.index:
        if df.loc[i,"执行操作"] == "o" and df.loc[i,"可取资金"] < df.loc[i,"操作资金"]:
            # case1: 取出的钱比券商账户余额现金更多，把操作金额改成可取资金的10000约掉
            df.loc[i,"操作资金"] = df.loc[i,"可取资金"] // 10000 * 10000
        elif df.loc[i,"执行操作"] == "i" and df.loc[i,"银行余额"] < df.loc[i,"操作资金"]:
            # case2: 转入券商账户钱要比银行余额更多，把操作金额改成银行余额万分为约分
            df.loc[i,"操作资金"] = df.loc[i,"银行余额"] // 10000 * 10000
        else:
            # 其他错误
            pass
        failed_continue_list["userid"].append(df.loc[i,"账户号"])
        failed_continue_list["action"].append(df.loc[i,"执行操作"])
        failed_continue_list["number"].append(df.loc[i,"操作资金"])
    return pd.DataFrame(failed_continue_list)

def process_res_df(res_df,curdate, exec_type,ts,accounts,accounts_temp):
    new_accounts_temp = {k:accounts_temp[k] for k in accounts_temp.keys() if k not in res_df["账户号"]}
    df_fail = res_df[res_df["转账结果"]!="succeed" | res_df["查询结果"]!="succeed"].reset_index() #reset index 因为后面用了loc
    df_true = res_df[res_df["转账结果"]=="succeed" & res_df["查询结果"]=="succeed"]
    if os.path.exists(f"~/operation_exec/{curdate}/{curdate}_{exec_type}_success_{ts}.txt"):
        df_true.to_csv(f"~/operation_exec/{curdate}/{curdate}_{exec_type}_success_{ts}.txt", sep = ",", mode='a', header=False, index=False)
    else:
        df_true.to_csv(f"~/operation_exec/{curdate}/{curdate}_{exec_type}_success_{ts}.txt", sep = ",", mode='a', header=True, index=False)
    if len(df_fail!=0):
        #重新处理
        failed_res = process_errors(df_fail,accounts)
        #下次要跑的
        if os.path.exists(f"~/operation/{curdate}_{exec_type}_{ts+1}.txt"):
            failed_res.to_csv(f"~/operation/{curdate}_{exec_type}_{ts+1}.txt", sep = ",", mode='a', header=False, index=False)
        else:
            failed_res.to_csv(f"~/operation/{curdate}_{exec_type}_{ts+1}.txt", sep = ",", mode='a', header=False, index=False)
    new_accounts = {k:accounts[k] for k in accounts.keys() if k not in df_true["账户号"]}
    return new_accounts, new_accounts_temp
       

def deepcopy(orig_dict):
    pass

def main_exec(curdate, exec_type='hedge_market_close', sleep=180):
    trans_seque = 1
    exec_path = f'~/operation_exec/{curdate}/'
    if not os.path.exists(exec_path):
        os.makedirs(exec_path)
    auto_acc, hs_acc, nonauto_acc = get_hedge_trans_config(curdate, exec_type, exec_path,trans_seque) #所有要处理的account：strat 字典
    # 并行处理三种accounts
    while True:
        limit_req_num = 20
        il = 1 #in loop 记录，这样报错的会被删除然后放到下一个目录里面sleep再跑而不是直接跟着没跑完的再跑一遍
        auto_acc_temp = deepcopy(auto_acc)
        hs_acc_temp = hs_acc
        nonauto_acc_temp = nonauto_acc
        while len(auto_acc_temp) > 0 or len(hs_acc_temp) > 0 or len(nonauto_acc_temp)>0:
            # 没有把所有跑完就一直运行
            #简单封装一下
            dict_slice = []
            acc2pro = [auto_acc_temp, hs_acc_temp, nonauto_acc_temp]
            for i in range(3):
                if len(acc2pro[i]) > limit_req_num:
                    dict_slice.append({k:acc2pro[i][k] for k in list(acc2pro[i].keys)[:limit_req_num]})
                else:
                    dict_slice.append(acc2pro[i])
            #######
            
            # log file里面的result文件路径
            base_res_path = exec_hedge_trans('run', curdate, f'{exec_type}', trans_seque,dict_slice[0], il)
            hs_res_path = exec_hedge_trans('run_hs', curdate, f'{exec_type}_hs', trans_seque, dict_slice[1], il)
            unauto_res_path = exec_hedge_trans('run', curdate, f'{exec_type}_non', trans_seque, dict_slice[2], il)
            # 提取调拨回执回执
            df_base = pd.read_csv(f'~/product/log/{base_res_path}.txt', sep=',', dtype='str')
            df_hs = pd.read_csv(f'~/product/log/{hs_res_path}.txt', sep=',', dtype='str')
            df_unauto = pd.read_csv(f'~/product/log/{unauto_res_path}.txt', sep=',', dtype='str')
            # 更新剩余操作
            auto_acc,auto_acc_temp = process_res_df(df_base, curdate, exec_type, trans_seque, auto_acc,auto_acc_temp)
            hs_acc, hs_acc_temp = process_res_df(df_hs, curdate, exec_type, trans_seque, hs_acc,hs_acc_temp)
            nonauto_acc, nonauto_acc_temp = process_res_df(df_unauto, curdate, exec_type, trans_seque, nonauto_acc, nonauto_acc_temp)
            il += 1
        trans_seque += 1
        time.sleep(sleep)
        if len(auto_acc) == 0 and len(hs_acc) == 0 and len(nonauto_acc) == 0:
            break


if __name__ == "__main__":
    print(get_hedge_trans_config(20250414,"hedge_market_close","./"))
    