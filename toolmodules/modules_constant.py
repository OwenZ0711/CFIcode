
ProductionDict_LongShortClss = {
    'MSLS1': 'ms',
    'SHORTMSLS1': 'ms',
    'MSLS2': 'msls2',
    'SHORTMSLS2': 'msls2',
    'ZJJT1': 'zj',
    'SHORTZJJT1': 'zj',
}

PLOT_STRUCT_DICT = {
    1: [1, 1],
    2: [2, 1],
    3: [2, 2], 4: [2, 2],
    5: [3, 2], 6: [3, 2],
    7: [3, 3], 8: [3, 3], 9: [3, 3],
    10: [4, 3], 11: [4, 3], 12: [4, 3],
    13: [5, 3], 14: [5, 3], 15: [5, 3], 16: [4, 4],
    17: [5, 4], 18: [5, 4], 19: [5, 4], 20: [5, 4],
    21: [6, 4], 22: [6, 4], 23: [6, 4], 24: [6, 4], 25: [5, 5],
    26: [6, 5], 27: [6, 5], 28: [6, 5], 29: [6, 5], 30: [6, 5],
    31: [7, 5], 32: [7, 5], 33: [7, 5], 34: [7, 5], 35: [7, 5], 36: [6, 6],
    37: [7, 6], 38: [7, 6], 39: [7, 6], 40: [7, 6], 41: [7, 6], 42: [7, 6],
    43: [8, 6], 44: [8, 6], 45: [8, 6], 46: [8, 6], 47: [8, 6], 48: [8, 6], 49: [7, 7]
}

EMAIL_RECEIVER_LIST = [
    'ritchguo@centuryfrontier.com',
    'hyanak@centuryfrontier.com',
    'luom@centuryfrontier.com',
    'yanli@centuryfrontier.com',
    'wangzy@centuryfrontier.com',
    'spchen@centuryfrontier.com',
    'chenxi@centuryfrontier.com',
    'amber.zhu@centuryfrontier.com',
    'zhangziyao@centuryfrontier.com'
]

EMAIL_RECEIVER_TRADING_LIST = [
    'ritchguo@centuryfrontier.com',
    '453326526@qq.com',
    'hyanak@centuryfrontier.com',
    'zhzhou@centuryfrontier.com',
    'luom@centuryfrontier.com',
    '1026688756@qq.com',
    'zhangziyao@centuryfrontier.com'
]

FutureName_2_IndexName = {
    'IM': 'ZZ1000',
    'IC': 'ZZ500',
    'IF': 'HS300',
    'IH': 'SZ50',
}

IndexName_2_IndexinteriorCode = {
    'ZZ500': '4978',
    'ZZA500': '636661',
    'HS300': '3145',
    'SZ50': '46',
    'ZZ1000': '39144',
    'ZZ2000': '561230',
    'ZZ800': '4982',
    'ZZHL': '6973',
    'ZZQZ': '14110',
}

IndexSecuCode_2_IndexName_FutureName = {
    '000016': ['SZ50', 'IH'],
    '000300': ['HS300', 'IF'],
    '000905': ['ZZ500', 'IC'],
    '000510': ['ZZA500', 'I-'],
    '000906': ['ZZ800', 'I-'],
    '000852': ['ZZ1000', 'IM'],
    '932000': ['ZZ2000', 'I-'],

    '000922': ['ZZHL', 'I-'],
    'H00922': ['HZZHL', 'I-'],
    '000985': ['ZZQZ', 'I-'],
}

IndexName_2_IndexSecuCode = {IndexSecuCode_2_IndexName_FutureName[index][0]: index for index in IndexSecuCode_2_IndexName_FutureName}
Future_Value_Multiplier = {'IC': 200, 'IF': 300, 'IH': 300, 'IM': 200, 'ZZ500': 200, 'HS300': 300, 'SZ50': 300, 'ZZ1000': 200}

Production_MulAccDC_Main_Dict = {
    'CMSZX1': 'CMSZX', 'ZX1': 'CMSZX', 'ZX1C': 'CMSZX',
    'ZTZX2': 'ZTZX', 'ZTZX2B': 'ZTZX', 'ZTZX2C': 'ZTZX',
    'JQ11': 'MJQ11', 'JQ11B': 'MJQ11',
    'ZX3': 'MZX', 'ZX3B': 'MZX', 'ZX3C': 'MZX',
    'ZX5': 'MZX5', 'ZX5B': 'MZX5', 'ZX5C': 'MZX5',
    # 'ZX6': 'MZX6', 'ZX6B': 'MZX6',
    'ZX8': 'MZX8', 'ZX8B': 'MZX8', 'ZX8C': 'MZX8',
    'ZX9': 'MZX9', 'ZX9B': 'MZX9',
    'ZX10': 'MZX10', 'ZX10B': 'MZX10', 'ZX10C': 'MZX10',
    'ZX11': 'MZX11', 'ZX11B': 'MZX11', 'ZX11C': 'MZX11',
    'ZX12': 'MZX12', 'ZX12B': 'MZX12',
    'ZX13': 'MZX13', 'ZX13B': 'MZX13',
    'ZX15': 'MZX15', 'ZX15B': 'MZX15',
    'ZX18': 'MZX18', 'ZX18B': 'MZX18',
    'HSDC1': 'MHSDC1', 'HSDC1B': 'MHSDC1',
    'DC9': 'MDC9', 'DC9B': 'MDC9', 'DC9C': 'MDC9',
    'DC25': 'MDC25', 'DC25B': 'MDC25',
    # 'DC24': 'MDC24', 'DC24B': 'MDC24',
    'ZADC1': 'MZADC1', 'ZADC1B': 'MZADC1', 'ZADC1D': 'MZADC1',
    'DC65A': 'MDC65A', 'DC65AB': 'MDC65A', 'DC65AC': 'MDC65A',
    'DC32A': 'MDC32A', 'DC32AB': 'MDC32A',
}

Production_MulAccDC_Main_Future = {
    'CMSZX': 'CMSZX1',
    'ZTZX': 'ZTZX2',
    'MJQ11': 'JQ11',
    'MZX': 'ZX3',
    'MZX5': 'ZX5',
    # 'MZX6': 'ZX6',
    'MZX8': 'ZX8',
    'MZX9': 'ZX9',
    'MZX10': 'ZX10',
    'MZX11': 'ZX11',
    'MZX12': 'ZX12',
    'MZX13': 'ZX13',
    'MZX15': 'ZX15',
    'MZX18': 'ZX18',
    'MDC9': 'DC9',
    'MHSDC1': 'HSDC1',
    'MDC25': 'DC25',
    # 'MDC24': 'DC24',
    'MZADC1': 'ZADC1B',
    'MDC65A': 'DC65A',
    'MDC32A': 'DC32A',
}

Production_MulAccZQ_Main_Dict = {
    'XHb1': 'XHb', 'XHb2': 'XHb', 'XHb3': 'XHb',
    'XHc1': 'XHc', 'XHc2': 'XHc', 'XHc3': 'XHc',
    'XHd1': 'XHd', 'XHd2': 'XHd', 'XHd3': 'XHd',
    'XHe1': 'XHe', 'XHe2': 'XHe',
    'ZQZS1': 'ZQZS', 'ZQZS1B': 'ZQZS', 'ZQZS1C': 'ZQZS',
    'ZQ3': 'MZQ3', 'ZQ3B': 'MZQ3',
    'ZQ24': 'MZQ24', 'ZQ24B': 'MZQ24',
    'ZQZX2': 'MZQZX2', 'ZQZX2B': 'MZQZX2', 'ZQZX2C': 'MZQZX2',
    'ZQZX3': 'MZQZX3', 'ZQZX3B': 'MZQZX3', 'ZQZX3C': 'MZQZX3',
    'ZQZX5': 'MZQZX5', 'ZQZX5B': 'MZQZX5',
    'ZQZX6': 'MZQZX6', 'ZQZX6C': 'MZQZX6',
    'ZQZX7': 'MZQZX7', 'ZQZX7B': 'MZQZX7',
    'ZQ16': 'MZQ16', 'ZQ16B': 'MZQ16', 'ZQ16C': 'MZQ16',
    'ZQ50': 'MZQ50', 'ZQ50B': 'MZQ50',
    'HSZS5': 'MHSZS5', 'HSZS5B': 'MHSZS5',
    'HSZQ10': 'MHSZQ10', 'HSZQ10B': 'MHSZQ10',
    'XHA500ZQ1': 'MXHA500ZQ', 'XHA500ZQ2': 'MXHA500ZQ',
    'A500ZQ3': 'MA500ZQ3', 'A500ZQ3B': 'MA500ZQ3', 'A500ZQ3C': 'MA500ZQ3',
    'ZQZX10': 'MZQZX10', 'ZQZX10B': 'MZQZX10',
    'HLYX3': 'MHLYX3', 'HLYX3B': 'MHLYX3',
}

Production_MulAccZQ_Main_Future = {
    'XHb': 'XHb1',
    'XHc': 'XHc1',
    'XHd': 'XHd2',
    'XHe': 'XHe1',
    'ZQZS': 'ZQZS1',
    'MZQ3': 'ZQ3',
    'MZQ24': 'ZQ24',
    'MZQZX2': 'ZQZX2',
    'MZQZX3': 'ZQZX3',
    'MZQZX5': 'ZQZX5',
    'MZQZX6': 'ZQZX6',
    'MZQZX7': 'ZQZX7',
    'MZQ16': 'ZQ16',
    'MZQ50': 'ZQ50',
    'MHSZS5': 'HSZS5',
    'MHSZQ10': 'HSZQ10',
    'MXHA500ZQ': 'XHA500ZQ1',
    'MA500ZQ3': 'A500ZQ3',
    'MZQZX10': 'ZQZX10',
    'MHLYX3': 'HLYX3',
}

Production_MulAccYX_Main_Dict = {
    # 'YX8': 'MYX8', 'YX8B': 'MYX8',
    'XHYX1A': 'XHYX', 'XHYX1B': 'XHYX', 'XHYX1C': 'XHYX',
    # 'YX9': 'MYX9', 'YX9B': 'MYX9', 'YX9C': 'MYX9',
    'XHYX1B1': 'XHYXB', 'XHYX1B2': 'XHYXB', 'XHYX1B3': 'XHYXB',
    'SSYX1': 'SSYX', 'SSYX1B': 'SSYX', 'SSYX1C': 'SSYX',
    'YX2': 'MYX2', 'YX2B': 'MYX2',
    'YX1': 'MYX1', 'YX1B': 'MYX1', 'YX1C': 'MYX1',
    'YX3A': 'MYX3A', 'YX3AB': 'MYX3A', 'YX3AC': 'MYX3A',
    'YX12': 'MYX12', 'YX12B': 'MYX12', 'YX12C': 'MYX12',
    'YX10': 'MYX10', 'YX10B': 'MYX10',
    'YX3B': 'MYX3B', 'YX3B2': 'MYX3B', 'YX3B3': 'MYX3B',
}

Production_MulAccYX_Main_Future = {
    # 'MYX8': 'YX8',
    'XHYX': 'XHYX1A',
    # 'MYX9': 'YX9',
    'XHYXB': 'XHYX1B1',
    'SSYX': 'SSYX1',
    'MYX2': 'YX2',
    'MYX1': 'YX1',
    'MYX3A': 'YX3A',
    'MYX12': 'YX12',
    'MYX10': 'YX10',
    'MYX3B': 'YX3B',
}

DICT_TIME_2_BAR_30MIN = {
    'Ti1_Weight': 100000, 'Ti1_Weight_Diff': 100000, 1: 100000,
    'Ti2_Weight': 103000, 'Ti2_Weight_Diff': 103000, 2: 103000,
    'Ti3_Weight': 110000, 'Ti3_Weight_Diff': 110000, 3: 110000,
    'Ti4_Weight': 113000, 'Ti4_Weight_Diff': 113000, 4: 113000,
    'Ti5_Weight': 133000, 'Ti5_Weight_Diff': 133000, 5: 133000,
    'Ti6_Weight': 140000, 'Ti6_Weight_Diff': 140000, 6: 140000,
    'Ti7_Weight': 143000, 'Ti7_Weight_Diff': 143000, 7: 143000,
    'Ti8_Weight': 150000, 'Ti8_Weight_Diff': 150000, 8: 150000,
}

TIME_LIST_5MIN_REFRESH_TIME = [
    93000, 93500,  94000,  94500,  95000,  95500, 100000, 100500, 101000,
    101500, 102000, 102500, 103000, 103500, 104000, 104500, 105000,
    105500, 110000, 110500, 111000, 111500, 112000, 112500, 130000,
    130500, 131000, 131500, 132000, 132500, 133000, 133500, 134000,
    134500, 135000, 135500, 140000, 140500, 141000, 141500, 142000,
    142500, 143000, 143500, 144000, 144500, 145000, 145500
]

TIME_LIST_5MIN_BAR = [
    93500,  94000,  94500,  95000,  95500, 100000, 100500, 101000,
    101500, 102000, 102500, 103000, 103500, 104000, 104500, 105000,
    105500, 110000, 110500, 111000, 111500, 112000, 112500, 113000,
    130500, 131000, 131500, 132000, 132500, 133000, 133500, 134000,
    134500, 135000, 135500, 140000, 140500, 141000, 141500, 142000,
    142500, 143000, 143500, 144000, 144500, 145000, 145500, 150000
]

DICT_TIME_2_BAR_5MIN = {f'Ti{_bar + 1}_Weight': _time for _bar, _time in enumerate(TIME_LIST_5MIN_BAR)}
DICT_TIME_2_BAR_5MIN.update({f'Ti{_bar + 1}_Weight_Diff': _time for _bar, _time in enumerate(TIME_LIST_5MIN_BAR)})
DICT_TIME_2_BAR_5MIN.update({_bar + 1: _time for _bar, _time in enumerate(TIME_LIST_5MIN_BAR)})

Ti8_Time_List = {
    'str': ['93500', '100000', '103000', '110000', '113000', '133000', '140000', '143000'],
    'int': [93500, 100000, 103000, 110000, 113000, 133000, 140000, 143000],
    'vwapsimu': [100000, 103000, 110000, 113000, 133000, 140000, 143000, 150000],
    'strStart': ['093500', '100000', '103000', '110000', '130000', '133000', '140000', '143000'],
    'strEnd': ['100000', '103000', '110000', '113000', '133000', '140000', '143000', '150000'],
}

Trades_Columns_List = ['Date', 'time', 'SecuCode', 'LongShort', 'tmp', 'Price', 'Volume', 'status', 'signal', 'CurHold', 'TargetQuota', 'SmoothQuota', 'PreQuota', 'PanKou', 'Send_Price', 'Ask_Price1', 'Bid_Price1', 'TradeSignalValue']
Trades_Columns_List_New = ['Date', 'Bar', 'SecuCode', 'LongShort', 'Volume', 'Price', 'status', 'signal', 'CurHold', 'TargetQuota', 'SmoothQuota', 'PreQuota', 'PanKou', 'Send_Price', 'Ask_Price1', 'Bid_Price1', 'TradeSignalValue']

WECHAT_BOT_KEY_DICT = {
        'check': '7a3ca5b1-c492-4aa7-993c-232ebe238f87',
        'report': 'd58b57f0-045e-4ec6-8535-851313b37e7f',
        'monitor': '5b401775-ed41-470d-8419-209ad7894fbc',
        't0': 'd32375e1-fb3c-4ec5-a565-760146ca6a1c',
        'process': 'b2eded59-4b7f-4492-b725-e2beabd0bdcd',
        'capital-config': '1edef1f7-55f6-4655-8299-1295c96196b0',
        'analysis-report': '0dee33cc-e31f-479a-ab55-ec7ea07e9cd3',
        'ls': 'f0462bcc-b46d-4dab-9d71-abd86b436ed3',
        'except': 'db5fce46-b459-408d-8447-1db7568ef697',
        'ls-new': 'e29d255c-6c71-4b2f-9822-ae1c8976ac68'
}

Winterfell_Monitor_JsonData_List = [
    'DoubleWinterfell.json',
    'DoubleWinterfell-SZ.json',
    'DoubleWinterfell5M.json',
    'DoubleWinterfell5M-SZ.json',
    'Winterfell.json',
]

Winterfell_Simu_Monitor_JsonData_List = [
    'TestWinterfell.json'
]


STAMP_RATE = 0.0005


Temp_Colo_Mechine_Dict = {
    'curdate': [],
    'simulist': ['simu', 'gf-sz-9'],
    'droplist': ['zt-sz-5'],
    '20220927': ['citic-sz-1', 'citic-sh-1'],
}

PreMarket_Start_Delay_Colo = [
    'gf',
    'hto',
    'tf',
    'sw',
    'gs',
    'cfipa',
    'cicc',
    'ebsc',
    'fz',
    'yhgj',
    'ha',
    'axsjqy-dg-app4',
    'axsjqy-sh-app4',
    'citic-sz-6',
    'citic-sh-6',
    'citic-sz-7',
    'citic-sh-7',
    'citic-sz-8',
    'citic-sh-8',
    'em-sz-3',
    'em-sh-3',
    'gx-sz-5',
]
