import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroid.protocols import instance_class_infer_binary_op

from toolmodules.modules import *


class PlotFigure():
    def __init__(self):
        pass

    def plot_compare_paras(self, **kwargs):
        date_list = kwargs['date_list']
        benchmark = kwargs.get('benchmark', None)
        paras_name_list = kwargs['paras_name_list']
        data_dict = kwargs['data_dict']
        twinx_mode = kwargs.get('twinx_mode', True)
        data_mode = kwargs.get('data_mode', 'df')
        multipier = kwargs.get('multipier', 1)

        figsize = kwargs.get('figsize', (20, 12))
        sub_num = kwargs.get('sub_num', None)
        plot_type = kwargs.get('plot_type', 'cumsum')
        origin_data_paras = kwargs.get('origin_data_paras', [])
        diff_data_paras = kwargs.get('diff_data_paras', [])

        title_str = kwargs.get('title_str', f'{date_list[0]}-{date_list[-1]}')
        save_path = kwargs.get('save_path', LOG_TEMP_PATH)
        save_file_name = kwargs.get(
            'save_file_name', f'{date_list[0]}-{date_list[-1]}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')

        if isinstance(plot_type, str): plot_type = {'origin': plot_type, 'diff': plot_type}

        if data_mode == 'df': data_dict = {flag: data[paras_name_list].to_dict(orient='list') for flag, data in data_dict.items()}

        if sub_num is None: subx, suby = PLOT_STRUCT_DICT[len(paras_name_list)]
        else: subx, suby = sub_num

        if not isinstance(multipier, dict): multipier = {paras: multipier for paras in paras_name_list}

        plt.figure(figsize=figsize)
        for iplot, paras in enumerate(paras_name_list):
            ax = plt.subplot(subx, suby, iplot + 1)
            print(paras)
            compare_data, compare_diff_data = {}, {}
            for flag, data in data_dict.items():
                compare_data[flag] = data_dict[flag][paras]
                if benchmark is not None:
                    if flag != benchmark:
                        compare_diff_data[f'{flag}-{benchmark}'] = np.array(data_dict[flag][paras]) - np.array(data_dict[benchmark][paras])

            if paras not in origin_data_paras:
                data_df = rename_df_columns_name(pd.DataFrame(compare_data, index=date_list), mode='4', precision=2, return_mode='str %%')
                if plot_type['origin'] == 'cumsum':
                    data_df = pd.DataFrame(np.nancumsum(data_df, axis=0), index=data_df.index, columns=data_df.columns) * multipier.get(paras, 1)
                elif plot_type['origin'].startswith('rolling'):
                    rolling_n = int(plot_type['origin'].split('_')[-1])
                    data_df = data_df.rolling(window=rolling_n, min_periods=1).mean() * 244
                else:
                    raise 'ValueError'
            else:
                data_df = rename_df_columns_name(pd.DataFrame(compare_data, index=date_list), mode='last/mean', precision=2, return_mode='str')

            bar1 = data_df.plot(ax=ax)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))

            if (benchmark is not None) and (paras in diff_data_paras):
                if paras not in origin_data_paras:
                    data_diff_df = rename_df_columns_name(pd.DataFrame(compare_diff_data, index=date_list), mode='4', precision=2, return_mode='str %%')
                    if plot_type['diff'] == 'cumsum':
                        data_diff_df = pd.DataFrame(np.nancumsum(data_diff_df, axis=0), index=data_diff_df.index, columns=data_diff_df.columns) * multipier.get(paras, 1)
                    elif plot_type['origin'].startswith('rolling'):
                        rolling_n = int(plot_type['origin'].split('_')[-1])
                        data_diff_df = data_diff_df.rolling(window=rolling_n, min_periods=1).mean() * 244
                    else:
                        raise 'ValueError'
                else:
                    data_diff_df = rename_df_columns_name(pd.DataFrame(compare_diff_data, index=date_list), mode='last/mean', precision=2, return_mode='str')
                    
                if not twinx_mode:
                    data_diff_df.plot(ax=ax)
                else:
                    plt.legend().remove()
                    ax_twin = ax.twinx()
                    bar2 = data_diff_df.plot(linestyle='-.', ax=ax_twin)
                    ax_twin.spines['top'].set_color('none')
                    ax_twin.spines['right'].set_color('none')
                    ax_twin.yaxis.set_ticks_position('left')
                    ax_twin.spines['left'].set_position(('data', len(date_list) - 1))
    
                    handles_left, labels_left = bar1.get_legend_handles_labels()  # 获取第一个条形图的图例
                    handles_right, labels_right = bar2.get_legend_handles_labels()  # 获取第二个条形图的图例
                    handles_left += handles_right
                    labels_left += labels_right
                    by_label = dict(zip(labels_left, handles_left))
                    plt.legend(list(by_label.values()), list(by_label.keys()))  # , loc='upper left', bbox_to_anchor=(0.05, 1.2), ncol=1, fontsize=9)

            plt.grid(True, linestyle='-.')
            plt.xticks(rotation=15)
            plt.title(paras)
        plt.suptitle(title_str)
        plt.tight_layout()
        plt.savefig(save_path + save_file_name)

        return [save_path + save_file_name]

    def plot_scatter_flag(self, **kwargs):
        date_list = kwargs['date_list']
        benchmark = kwargs.get('benchmark', None)
        paras_name_list = kwargs['paras_name_list']
        data_dict = kwargs['data_dict']
        twinx_mode = kwargs.get('twinx_mode', True)
        data_mode = kwargs.get('data_mode', 'df')
        multipier = kwargs.get('multipier', 1)

        figsize = kwargs.get('figsize', (20, 12))
        sub_num = kwargs.get('sub_num', None)
        plot_type = kwargs.get('plot_type', 'cumsum')

        title_str = kwargs.get('title_str', f'{date_list[0]}-{date_list[-1]}')
        save_path = kwargs.get('save_path', LOG_TEMP_PATH)
        save_file_name = kwargs.get('save_file_name',
                                    f'{date_list[0]}-{date_list[-1]}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')

        if isinstance(plot_type, str):
            plot_type = {'origin': plot_type, 'diff': plot_type}

        if data_mode == 'df':
            data_dict = {flag: data[paras_name_list].to_dict(orient='list') for flag, data in data_dict.items()}

        if sub_num is None:
            subx, suby = PLOT_STRUCT_DICT[len(paras_name_list)]
        else:
            subx, suby = sub_num

        if not isinstance(multipier, dict):
            multipier = {paras: multipier for paras in paras_name_list}

        plt.figure(figsize=figsize)
        for iplot, paras in enumerate(paras_name_list):
            ax = plt.subplot(subx, suby, iplot + 1)

            compare_data, compare_diff_data = {}, {}
            for flag, data in data_dict.items():
                compare_data[flag] = data_dict[flag][paras]
                if benchmark is not None:
                    if flag != benchmark:
                        compare_diff_data[f'{flag}-{benchmark}'] = np.array(data_dict[flag][paras]) - np.array(data_dict[benchmark][paras])

            data_df = rename_df_columns_name(pd.DataFrame(compare_data, index=date_list), mode='4', precision=2, return_mode='str %%')
            if plot_type['origin'] == 'cumsum':
                data_df = pd.DataFrame(np.nancumsum(data_df, axis=0), index=data_df.index, columns=data_df.columns) * multipier.get(paras, 1)
            elif plot_type['origin'].startswith('rolling'):
                rolling_n = int(plot_type['origin'].split('_')[-1])
                data_df = data_df.rolling(window=rolling_n, min_periods=1).mean() * 244
            else:
                raise 'ValueError'

            bar1 = data_df.plot(ax=ax)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))
            
            if benchmark is not None:
                data_diff_df = rename_df_columns_name(pd.DataFrame(compare_diff_data, index=date_list), mode='4', precision=2, return_mode='str %%')
                if plot_type['diff'] == 'cumsum':
                    data_diff_df = pd.DataFrame(np.nancumsum(data_diff_df, axis=0), index=data_diff_df.index, columns=data_diff_df.columns) * multipier.get(paras, 1)
                elif plot_type['origin'].startswith('rolling'):
                    rolling_n = int(plot_type['origin'].split('_')[-1])
                    data_diff_df = data_diff_df.rolling(window=rolling_n, min_periods=1).mean() * 244
                else:
                    raise 'ValueError'
                    
                if not twinx_mode:
                    data_diff_df.plot(ax=ax)
                else:
                    plt.legend().remove()
                    ax_twin = ax.twinx()
                    bar2 = data_diff_df.plot(linestyle='-.', ax=ax_twin)
                    ax_twin.spines['top'].set_color('none')
                    ax_twin.spines['right'].set_color('none')
                    ax_twin.yaxis.set_ticks_position('left')
                    ax_twin.spines['left'].set_position(('data', len(date_list) - 1))
    
                    handles_left, labels_left = bar1.get_legend_handles_labels()  # 获取第一个条形图的图例
                    handles_right, labels_right = bar2.get_legend_handles_labels()  # 获取第二个条形图的图例
                    handles_left += handles_right
                    labels_left += labels_right
                    by_label = dict(zip(labels_left, handles_left))
                    legend = plt.legend(list(by_label.values()), list(by_label.keys()))  # , loc='upper left', bbox_to_anchor=(0.05, 1.2), ncol=1, fontsize=9)

            plt.grid(True, linestyle='-.')
            plt.xticks(rotation=15)
            plt.title(paras)
        plt.suptitle(title_str)
        plt.tight_layout()
        plt.savefig(save_path + save_file_name)

        return [save_path + save_file_name]


def format_ax(ax=None, left_position=0., legend=True, rotation=None):
    if ax is None:
        ax = eval('plt.gca()')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')

    if rotation is not None:
        eval(f'plt.xticks(rotation={rotation})')

    if left_position is not None:
        ax.spines['left'].set_position(('data', left_position))
        if legend:
            if left_position == 0:
                ax.legend(loc='upper left')
            else:
                ax.legend(loc='upper right', bbox_to_anchor=(1.1, 0.9))
    plt.grid(True, linestyle='--')


def plot_one_figure_quickly(plt_str, title='', prcss_dt=True):
    plt.figure(figsize=(20, 12))
    plt.rcParams['font.size'] = 15

    ax = plt.subplot(111)
    if prcss_dt:
        plt_str = rename_df_columns_name(plt_str, mode='4', precision=2, return_mode='str %%')
        plt_str = pd.DataFrame(np.nancumsum(plt_str, axis=0), index=plt_str.index, columns=plt_str.columns)

    plt_str.plot(ax=ax)

    format_ax()
    plt.title(title)
    return ax


def get_plot_infor_list_from_df_compare(
        df_compare, groupby='Class', columns='Product', values=None,
        col_suf=None, product_list=None, benchmark=None, split_date_list=None, diff_thres=None):
    plot_infor_list = []
    for variable, df_comp in df_compare.groupby(groupby):
        if col_suf is not None:
            col_suf_var = '/'.join([str(np.round(df_comp[col_suf_].mean(), 2)) for col_suf_ in col_suf])
        else:
            col_suf_var = ''

        if isinstance(values, str):
            df_comp = pd.pivot_table(df_comp, index='Date', columns=columns, values=values)
        else:
            conlist = []
            for value in values:
                df_comp_value = pd.pivot_table(df_comp, index='Date', columns=columns, values=value)
                conlist.append(df_comp_value)
            df_comp = pd.concat(conlist, axis=1)

        df_comp = df_comp.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        print(df_comp)
        if product_list is not None and benchmark is not None:
            for product in product_list:
                if product == benchmark:
                    continue

                if product not in df_comp.columns.to_list():
                    df_comp[product] = 0

                if benchmark not in df_comp.columns.to_list():
                    df_comp[benchmark] = 0

                df_comp[f'{product}_Diff_{benchmark}'] = df_comp[product] - df_comp[benchmark]
                if diff_thres is not None:
                    if isinstance(diff_thres, list) or isinstance(diff_thres, tuple):
                        df_comp = df_comp[(df_comp[f'{product}_Diff_{benchmark}'] > diff_thres[0]) &
                                          (df_comp[f'{product}_Diff_{benchmark}'] < diff_thres[1])]
                    else:
                        df_comp = df_comp[np.abs(df_comp[f'{product}_Diff_{benchmark}']) < diff_thres]

        df_comp = df_comp.rename(
            {col: f"{col}|{col_suf_var}: {calculate_sharpe_annual_maxdd(df_comp[col].to_list(), precision=2, return_mode='str %%')}bps"
             if 'Ret' in variable else f"{col}|{col_suf_var}: {np.round(df_comp[col].sum() * 100, 1)}%"
             for col in df_comp.columns}, axis='columns')

        if 'Ret' in variable:
            df_comp = pd.DataFrame(np.nancumsum(df_comp, axis=0), index=df_comp.index, columns=df_comp.columns)

        print(df_comp)
        plot_infor = {
            'data': df_comp,
            'yl': '',
            'title': variable,
        }

        if split_date_list is not None:
            min_df, max_df = np.min(df_comp), np.max(df_comp)
            split_infor = []
            for date in split_date_list:
                if date not in df_comp.index.to_list():
                    continue

                date_index = df_comp.index.to_list().index(date)
                split_infor.append([
                    [date_index, date_index], [min_df, max_df]
                ])

            plot_infor['plot_line'] = split_infor

        plot_infor_list.append(plot_infor)
    return plot_infor_list


def Plot_MutilBarplot(df, title, ax, type='bar'):
    if type == 'bar':
        df = df.rename(
            {col: f'{col}: {round(np.mean(df[col].to_list()), 4)}/{round(np.std(df[col].to_list()), 4)}'
             for col in df.columns.to_list()}, axis='columns')
        df.plot.bar(ax=ax)
    elif type == 'hist':
        ax.hist(df, bins=50)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.yaxis.set_ticks_position('left')

    plt.grid(True, linestyle='-.')
    if type == 'bar':
        plt.ylabel('trading ratio')
        ax.legend(loc='upper left', markerscale=0.3)

    plt.title(title)
    plt.grid(color='r', linestyle='--', linewidth=1, alpha=0.3)


def Plot_MutilBarplot_multitime(dict_df, type, curdate, output_dir, figsize_auto):
    plot_struct_dict = {1: [1, 1], 2: [2, 1], 3: [2, 2], 4: [2, 2], 5: [3, 2], 6: [3, 2], 7: [3, 3],
                        8: [3, 3], 9: [3, 3], 10: [4, 3], 11: [4, 3], 12: [4, 3], 13: [4, 4], 14: [4, 4],
                        15: [4, 4], 16: [4, 4], 17: [5, 4], 18: [5, 4], 19: [5, 4], 20: [5, 4], 21: [5, 5],
                        22: [5, 5], 23: [5, 5], 24: [5, 5], 25: [5, 5]}

    plt.figure(figsize=figsize_auto)
    subx, suby = plot_struct_dict[len(dict_df)][0], plot_struct_dict[len(dict_df)][1]
    for coli, time_key in enumerate(dict_df.keys()):
        ax = plt.subplot(subx, suby, coli + 1)
        df_col = dict_df[time_key].copy(deep=True).sort_values(time_key, ascending=False)

        ratio_list = [ratio for ratio in dict_df[time_key][time_key].to_list() if ratio != 0]
        if ratio_list:
            mean = round(np.mean(ratio_list) * 100, 2)
            median = round(np.median(ratio_list) * 100, 2)
            percent25 = round(np.percentile(ratio_list, 75) * 100, 2)
            maxrat = round(np.max(ratio_list) * 100, 2)
            title = f'{time_key}: {len(ratio_list)}/{mean}%/{median}%/{percent25}%/{maxrat}%_{df_col.index[0]}'
        else:
            title = f'{time_key}: '
        df_col_head = df_col[df_col[time_key] > 0.4].head(10).reset_index()
        Plot_MutilBarplot(ratio_list, title, ax=ax, type='hist')
        if not df_col_head.empty:
            for iba, (code, ratio, amount, price) in enumerate(df_col_head.values):
                ratio = round(ratio * 100, 1)
                content = f'{code}: {ratio}%, {str(round(price, 2)):7}, {round(amount / 10000, 2)}W'
                ax.text(0.5, - 0.8 / 9 * iba + 0.9, content, alpha=0.5, transform=ax.transAxes)
    plt.suptitle(f'[{curdate}]-TradeRatio_1450_after_{type}_/num/mean/median/percent25/max_code')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + f'{curdate}_tradingratio_1450_after_{type}.png')
    # plt.show()


def plot_subplots_use_list_dict_data(
        plot_infor_list, suptitle=None, fig_path=None, kind='line', xrotat=15, linestyle='--', figure_size=(20, 16), secondary_y_list=None):
    sp_x, sp_y = PLOT_STRUCT_DICT[len(plot_infor_list)]
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure(figsize=figure_size)
    for plot_seq, infor_dict in enumerate(plot_infor_list):
        ax = plt.subplot(int(sp_x), int(sp_y), int(plot_seq + 1))
        df_subplot = infor_dict['data']
        df_subplot.columns = df_subplot.columns.astype('str')
        df_subplot.columns.name = None
        df_subplot.index.name = None
        if secondary_y_list is None:
            df_subplot.plot(ax=ax, kind=kind)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))

            if infor_dict.get('plot_line', None) is not None:
                for x_, y_ in infor_dict['plot_line']:
                    ax.plot(x_, y_, 'k--')
        else:
            ax = df_subplot.plot(
                ax=ax, kind=kind, secondary_y=[col for col in df_subplot.columns if secondary_y_list in col])

            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))

            if infor_dict.get('plot_line', None) is not None:
                for x_, y_ in infor_dict['plot_line']:
                    ax.plot(x_, y_, 'k--')

            ax.right_ax.spines['top'].set_color('none')
            ax.right_ax.spines['left'].set_color('none')
            ax.right_ax.yaxis.set_ticks_position('right')
            ax.right_ax.spines['right'].set_position(('data', len(df_subplot) - 1))

        plt.grid(True, linestyle=linestyle)
        plt.xticks(rotation=xrotat)

        plt.xlabel(None)
        if infor_dict.get('yl', None) is not None:
            plt.ylabel(infor_dict['yl'])
        if infor_dict.get('title', None) is not None:
            plt.title(infor_dict['title'])

    plt.suptitle(suptitle)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)


def plot_subplots_hist_use_list_dict_data(
        plot_infor_list, suptitle=None, fig_path=None, xrotat=15, linestyle='-.', figure_size=(20, 16), sp_xy=None):
    if sp_xy is None: sp_x, sp_y = PLOT_STRUCT_DICT[len(plot_infor_list)]
    else: sp_x, sp_y = sp_xy

    matplotlib.rcParams.update({'font.size': 8})
    plt.figure(figsize=figure_size)
    for plot_seq, infor_dict in enumerate(plot_infor_list):
        ax = plt.subplot(int(sp_x), int(sp_y), int(plot_seq + 1))
        data = infor_dict['data']
        if isinstance(data, dict):
            for data_key in data.keys():
                ax.hist(data[data_key], bins=max(min(len(data[data_key]), 50), 10), label=data_key)
            plt.legend(data.keys())
        else:
            ax.hist(data, bins=min(len(data), 50))

        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))

        plt.grid(True, linestyle=linestyle)
        plt.xticks(rotation=xrotat)

        plt.title(infor_dict['title'])

    plt.suptitle(suptitle)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)

    return sp_x, sp_y


def plot_subplots_compare_dict_data(curdate, time_list, dict_com_left, dict_com_right=None, target_paras_list=None, fpath=None, multi_prods=False, one_figure=True, title_str=''):
    dict_com_left = deepcopy(dict_com_left)
    if dict_com_right is not None:
        dict_com_right = deepcopy(dict_com_right)

    time_list = [str(x) for x in time_list]
    if not multi_prods:
        if target_paras_list is None:
            target_paras_list = sorted(list(set(dict_com_left.keys()) - {'product'}))
            print(target_paras_list)

        subx, suby = PLOT_STRUCT_DICT[len(target_paras_list)]
        plt.figure(figsize=(20, 16))
        for iplot, key in enumerate(target_paras_list):
            if key not in target_paras_list:
                continue

            ax = plt.subplot(subx, suby, iplot + 1)
            diff_name = dict_com_left['product']
            if dict_com_right is not None:
                diff_name += '-' + dict_com_right['product']

            plt.plot(time_list, dict_com_left[key], 'b')
            if (dict_com_right is not None) and (dict_com_right.get(key) is not None):
                plt.plot(time_list, dict_com_right[key], 'r')
                compare_diff = np.array(dict_com_left[key]) - np.array(dict_com_right[key])
                plt.plot(time_list, compare_diff, 'k--')
            if 'ret' in key:
                legend_list = [f"{dict_com_left['product']}: {round(dict_com_left[key][-1], 3)} bps"]
                if (dict_com_right is not None) and (dict_com_right.get(key) is not None):
                    legend_list += [
                        f"{dict_com_right['product']}: {round(dict_com_right[key][-1], 3)} bps",
                        f"{diff_name}: {round(compare_diff[-1], 3)} bps"
                    ]
            else:
                legend_list = [
                    f"{dict_com_left['product']}: "
                    f"{round(dict_com_left[key][-1], 4)}/"
                    f"{round((dict_com_left[key][-1] - dict_com_left[key][-2]) * 10000, 3)} bps"
                ]
                if (dict_com_right is not None) and (dict_com_right.get(key) is not None):
                    legend_list += [
                        f"{dict_com_right['product']}: "
                        f"{round(dict_com_right[key][-1], 4)}/"
                        f"{round((dict_com_right[key][-1] - dict_com_right[key][-2]) * 10000, 3)} bps",
                        f"{diff_name}: {round(compare_diff[-1], 4)}/"
                        f"{round((compare_diff[-1] - compare_diff[-2]) * 10000, 2)} bps"
                    ]

            plt.legend(legend_list, framealpha=0.5)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))
            ax.xaxis.set_major_locator(MultipleLocator(30))
            plt.grid(True, linestyle='-.')
            plt.xticks(rotation=15)
            plt.title(key)

        if dict_com_right is not None:
            plt.suptitle(
                f'[{curdate}]-{title_str}: {dict_com_left["product"]} & {dict_com_right["product"]}')
        else:
            plt.suptitle(f'[{curdate}]-{title_str}: {dict_com_left["product"]}')
    else:
        if one_figure:
            paras_df_dict = dict_com_left
            subx, suby = PLOT_STRUCT_DICT[len(target_paras_list)]
            plt.figure(figsize=(20, 16))
            for iplot, key in enumerate(target_paras_list):
                if key not in target_paras_list:
                    continue

                ax = plt.subplot(subx, suby, iplot + 1)
                df_paras = paras_df_dict[key]
                if 'ret' in key:
                    df_paras = df_paras.rename(
                        {col: f'{col}: {round(df_paras[col].to_list()[-1], 2)}bps' for col in df_paras.columns}, axis='columns')
                else:
                    df_paras = df_paras.rename(
                        {col: f'{col}: {round(df_paras[col].to_list()[-1], 4)}' for col in df_paras.columns}, axis='columns')

                df_paras.plot(ax=ax)

                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.yaxis.set_ticks_position('left')
                ax.spines['left'].set_position(('data', 0))
                ax.xaxis.set_major_locator(MultipleLocator(30))
                plt.grid(True, linestyle='-.')
                plt.xticks(rotation=15)
                plt.title(key)
            plt.suptitle(f'[{curdate}]-{title_str}')
        else:
            paras_df_dict = dict_com_left
            paras_name = target_paras_list[0]
            subx, suby = PLOT_STRUCT_DICT[len(paras_df_dict[paras_name].columns.to_list())]
            plt.figure(figsize=(20, 16))
            for iplot, key in enumerate(paras_df_dict[paras_name].columns.to_list()):
                if key not in paras_df_dict[paras_name].columns.to_list():
                    continue

                ax = plt.subplot(subx, suby, iplot + 1)
                df_paras = paras_df_dict[paras_name][[key]]
                if 'ret' in paras_name:
                    df_paras = df_paras.rename(
                        {col: f'{col}: {round(df_paras[col].to_list()[-1], 2)}bps' for col in df_paras.columns},
                        axis='columns')
                else:
                    df_paras = df_paras.rename(
                        {col: f'{col}: {round(df_paras[col].to_list()[-1], 4)}' for col in df_paras.columns},
                        axis='columns')

                df_paras.plot(ax=ax)

                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.yaxis.set_ticks_position('left')
                ax.spines['left'].set_position(('data', 0))
                ax.xaxis.set_major_locator(MultipleLocator(30))
                plt.grid(True, linestyle='-.')
                plt.xticks(rotation=15)
                plt.title(key)
            plt.suptitle(f'[{curdate}]-{title_str} | {paras_name}')
    plt.tight_layout()
    if fpath is not None:
        plt.savefig(fpath)


def plot_compare_execute_return(*args, bm_para_reverse_list=None, df_ret=None, suptitle=None, fig_path=None):
    if bm_para_reverse_list is None: bm_para_reverse_list = []
    df_ret = df_ret.set_index('Date')
    columns_list_origin = df_ret.columns.to_list()

    plt.figure(figsize=(20, 12))
    ax = plt.subplot(2, 2, 1)
    df_ret_origin = rename_df_columns_name(df_ret[columns_list_origin].copy(deep=True), mode='4', precision=2, return_mode='str %%')
    df_ret_origin = pd.DataFrame(np.nancumsum(df_ret_origin, axis=0), index=df_ret_origin.index, columns=df_ret_origin.columns) * 10000
    df_ret_origin.plot(ax=ax)
    format_ax(ax)
    plt.title('origin')

    for bm_n, bm_para in enumerate(args):
        new_paras_list = []
        for col in columns_list_origin:
            if bm_para == col: continue
            if bm_para in bm_para_reverse_list:
                df_ret[f'{bm_para}-{col}'] = df_ret[bm_para] - df_ret[col]
                new_paras_list.append(f'{bm_para}-{col}')
            else:
                df_ret[f'{col}-{bm_para}'] = df_ret[col] - df_ret[bm_para]
                new_paras_list.append(f'{col}-{bm_para}')

        ax = plt.subplot(2, 2, bm_n + 2)
        df_ret_bm = rename_df_columns_name(df_ret[new_paras_list].copy(deep=True), mode='4', precision=2, return_mode='str %%')
        df_ret_bm = pd.DataFrame(np.nancumsum(df_ret_bm, axis=0), index=df_ret_bm.index, columns=df_ret_bm.columns) * 10000
        df_ret_bm.plot(ax=ax)
        format_ax(ax)
        plt.title(f'bm={bm_para}')

    if suptitle is not None: plt.suptitle(suptitle)
    plt.tight_layout()
    if fig_path is not None: plt.savefig(fig_path)

def get_intrati8_weight_by_quota(df_quota_vol, df_price=None, df_30min_close=None, mode='lastprice', bar_mode=8):
    if mode == 'closeprice':
        union_index = df_quota_vol.index.union(df_price.index)
        df_quota_vol = df_quota_vol.reindex(index=union_index, fill_value=np.nan)
        close_mat = pd.DataFrame(np.tile(np.array(df_price[['ClosePrice']]), bar_mode),
                                 index=union_index, columns=[f'Ti{i}_Weight' for i in range(1, bar_mode + 1)])
        df_weight = df_quota_vol * close_mat
        df_weight = df_weight / df_weight.sum(axis=0)
        return df_weight
    else:
        union_index = df_quota_vol.index.union(df_30min_close.index)
        df_quota_vol = df_quota_vol.reindex(index=union_index, fill_value=np.nan)
        df_30min_close = df_30min_close.reindex(index=union_index, fill_value=np.nan).rename(
            {i: f'Ti{i}_Weight' for i in range(1, bar_mode + 1)}, axis='columns')
        df_weight = df_quota_vol * df_30min_close
        df_weight = df_weight / df_weight.sum(axis=0)
        return df_weight


def get_intrati8_weight_row_datel(date, alpha_name, file_path=None):
    DPAlpha = sio.loadmat(f'{file_path}{date}/{alpha_name}.mat')['AF']
    date_list = [datei[0][0] for datei in DPAlpha['DateL'][0][0]]
    date_index = date_list.index(date)

    dl_af = DPAlpha[0, 0]
    dl_afm_df = pd.DataFrame(dl_af['Weight'][date_index] / 100,
                             index=[sc[0][0] for sc in dl_af['SecuCodeL']],
                             columns=[f'Weight']
                             ).fillna(0).sort_index().reset_index().rename({'index': 'SecuCode'}, axis='columns')
    dl_afm_df = dl_afm_df.set_index('SecuCode')
    return dl_afm_df


def get_intrati8_weight_fast(date, alpha_name, file_path=None, bar_mode=8):
    if file_path is None: file_path = f'{PLATFORM_PATH_DICT["z_path"]}Share/ht/d0Prod/'
    if not os.path.exists(f'{file_path}{date}/{alpha_name}.mat'):
        return pd.DataFrame(columns=[f'Ti{i}_Weight' for i in range(1, bar_mode + 1)])

    DPAlpha = sio.loadmat(f'{file_path}{date}/{alpha_name}.mat')['AF']

    dl_af = DPAlpha[0, 0]
    dl_afm_df = pd.DataFrame(dl_af['AFM'][0][-1].T / 100,
                             index=[expand_stockcode(sc) for sc in dl_af['SecuCodeL']],
                             columns=[f'Ti{i}_Weight' for i in range(1, bar_mode + 1)]
                             ).fillna(0).sort_index().reset_index().rename({'index': 'SecuCode'}, axis='columns')
    dl_afm_df = dl_afm_df.set_index('SecuCode')
    return dl_afm_df


def get_intrati8_weight_longtime(date, usedate=None, alpha_name=None, DPalpha=None, long_mode=False, bar_mode=8):
    if DPalpha is None: DPAlpha = sio.loadmat(f'{PLATFORM_PATH_DICT["z_path"]}Share/intramat/{usedate}/{alpha_name}.mat')['AF']
    else: DPAlpha = deepcopy(DPalpha)

    date_list = list(DPAlpha['DateL'][0][0][0])
    if not long_mode:
        date_index = date_list.index(int(date))

        dl_af = DPAlpha[0, 0]
        dl_afm_df = pd.DataFrame(dl_af['AFM'][date_index][-1].T / 100,
                                 index=[expand_stockcode(sc) for sc in dl_af['SecuCodeL']],
                                 columns=[f'Ti{i}_Weight' for i in range(1, bar_mode + 1)]
                                 ).fillna(0).sort_index().reset_index().rename({'index': 'SecuCode'}, axis='columns')
        dl_afm_df = dl_afm_df.set_index('SecuCode')
        return dl_afm_df
    else:
        dl_afm_df_dict = {}
        for date_index in range(date_list.index(int(date)), len(date_list)):
            dl_af = DPAlpha[0, 0]
            dl_afm_df = pd.DataFrame(dl_af['AFM'][date_index][-1].T / 100,
                                     index=[expand_stockcode(sc) for sc in dl_af['SecuCodeL']],
                                     columns=[f'Ti{i}_Weight' for i in range(1, bar_mode + 1)]
                                     ).fillna(0).sort_index().reset_index().rename({'index': 'SecuCode'}, axis='columns')
            dl_afm_df = dl_afm_df.set_index('SecuCode')
            dl_afm_df_dict[date_list[date_index]] = dl_afm_df

        return dl_afm_df_dict


def get_intrati8_weight_longtime_dict(date_list, usedate=None, alpha_name=None, DPalpha=None, Ret_Mode='mat-dict', bar_mode=8):
    weight_mat_dict = get_intrati8_weight_longtime(date_list[0], usedate, alpha_name, DPalpha, long_mode=True, bar_mode=bar_mode)
    if Ret_Mode == 'mat-dict': return weight_mat_dict

    weight_diff_dict = {}
    for pre_date, date in zip(date_list, date_list[1:]):
        weight_diff_dict[date] = pd.concat([
            weight_mat_dict[pre_date][['Ti8_Weight']].rename({'Ti8_Weight': 'Ti0_Weight'}, axis='columns'),
            weight_mat_dict[date]], axis=1).fillna(0).diff(axis=1).rename(
            {f'Ti{i}_Weight': f'Ti{i}_Weight_Diff' for i in range(1, 9)}, axis='columns')[
            [f'Ti{i}_Weight_Diff' for i in range(1, 9)]]

    if Ret_Mode == 'diff-dict': return weight_diff_dict

    return weight_mat_dict, weight_diff_dict


def get_diff_intrati8_weight(date, predate, alpha_name, pre_alpha_name, mode='diff8', bar_mode=8, bar_mode_pre=8):
    pre_ti8_weight = get_intrati8_weight_fast(predate, pre_alpha_name, bar_mode=bar_mode_pre)
    ti8_weight = get_intrati8_weight_fast(date, alpha_name, bar_mode=bar_mode)

    return get_diff_intrati8_weight_simple(pre_ti8_weight, ti8_weight, mode, bar_mode, bar_mode_pre)


def get_diff_intrati8_weight_simple(pre_ti8_weight, ti8_weight, mode='diff8', bar_mode=8, bar_mode_pre=None):
    if bar_mode_pre is None: bar_mode_pre = bar_mode
    pre_ti8_weight = pre_ti8_weight[[f'Ti{bar_mode_pre}_Weight']].rename({f'Ti{bar_mode_pre}_Weight': 'Ti0_Weight'}, axis='columns')

    mat_all_df = pd.concat([pre_ti8_weight, ti8_weight], axis=1).fillna(0)
    if mode == 'total-diff':
        mat_all_df['Total_Weight_Diff'] = 0.
        for i in range(1, bar_mode + 1):
            mat_all_df['Total_Weight_Diff'] += np.abs(mat_all_df[f'Ti{i}_Weight'] - mat_all_df[f'Ti{i - 1}_Weight'])
        return mat_all_df[mat_all_df['Total_Weight_Diff'] > 0]
    elif mode == 'diff7':
        for i in range(bar_mode):
            mat_all_df[f'Ti{i}_Weight_Diff'] = mat_all_df[f'Ti{i + 1}_Weight'] - mat_all_df[f'Ti{i}_Weight']
        return mat_all_df[[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode)]]
    elif mode == 'diff8':
        for i in range(1, bar_mode + 1):
            mat_all_df[f'Ti{i}_Weight_Diff'] = mat_all_df[f'Ti{i}_Weight'] - mat_all_df[f'Ti{i - 1}_Weight']
        return mat_all_df[[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)]]
    elif mode == 'weight9':
        return mat_all_df
    else: raise ValueError


def get_diff_intrati8_weight_longtime(alpha_dict, date_list, mode='daily', bar_mode=8):
    if mode == 'daily':
        weight_list = []
        for date in date_list:
            if isinstance(alpha_dict, str): alpha_name = alpha_dict
            else: alpha_name = alpha_dict[date]
            weight = get_intrati8_weight_fast(date, alpha_name)
            weight_list.append(weight)

        weight_diff_dict = {}
        for i, (pre_weight, cur_weight) in enumerate(zip(weight_list, weight_list[1:])):
            df_weight_diff = pd.concat([pre_weight[[f'Ti{bar_mode}_Weight']].rename(
                {f'Ti{bar_mode}_Weight': 'Ti0_Weight'}, axis='columns'), cur_weight], axis=1).fillna(0)
            weight_diff_dict[date_list[i + 1]] = df_weight_diff.diff(axis=1)[
                [f'Ti{i}_Weight' for i in range(1, bar_mode + 1)]]
    else:
        weight_diff_dict = get_intrati8_weight_longtime_dict(date_list, usedate=date_list[-1], alpha_name=alpha_dict, Ret_Mode='diff-dict', bar_mode=bar_mode)

    return weight_diff_dict


def get_simulate_diff_volume_mat_by_weight(
        date=None, predate=None, alpha_name=None, pre_alpha_name=None, init_mv=1e8, df_price=None, df_lastp30=None, mode='lastprice', matweight_diff=None, bar_mode=8, bar_mode_pre=8):
    if matweight_diff is None:
        matweight_diff = get_diff_intrati8_weight(date, predate, alpha_name, pre_alpha_name, bar_mode=bar_mode, bar_mode_pre=bar_mode_pre)
    else:
        matweight_diff = matweight_diff.copy(deep=True)
    if matweight_diff.empty: return pd.DataFrame()

    matweight_diff = matweight_diff * init_mv
    matweight_diff.columns = [f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)]

    # 交易额除以收盘价得到 持仓量变化矩阵，交易量矩阵
    if mode == 'pre-close':
        if df_price is None: df_price = get_price(date, date).reset_index()[['SecuCode', 'PreClosePrice', 'ClosePrice']].fillna(0).set_index('SecuCode')
        else: df_price = df_price.copy(deep=True)

        union_index = matweight_diff.index.union(df_price.index)
        matweight_diff = matweight_diff.reindex(index=union_index, fill_value=np.nan)
        df_price = df_price.reindex(index=union_index, fill_value=np.nan)
        df_price = pd.DataFrame(np.tile(np.array(df_price[['PreClosePrice']]), bar_mode),
                                index=union_index, columns=[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)])
        matweight_diff /= df_price

    else:  # mode == 'i-1_bar':
        if df_lastp30 is None: df_lastp30 = get_lastprice(date)
        else: df_lastp30 = df_lastp30.copy(deep=True)

        union_index = matweight_diff.index.union(df_lastp30.index)
        matweight_diff = matweight_diff.reindex(union_index, fill_value=np.nan)[[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)]]
        df_lastp30 = df_lastp30.reindex(union_index, fill_value=np.nan).rename(
            {i - 1: f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)}, axis='columns')[[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)]]
        matweight_diff /= df_lastp30

    matweight_diff = matweight_diff.replace([np.inf, -np.inf], np.nan).fillna(0).sort_index()
    return matweight_diff


def read_code_second_summary_npy(date, summ_var=None):
    if summ_var is None:
        summ_var = ['volatility'] + [f'av1_vol_{vol_ratio}_r' for vol_ratio in [5, 10]] + \
        ['av5sum', 'bv5sum', 'av5sum_diff', 'bv5sum_diff', 'a_bv5diff', 'spread', 'spread_ret']

    if isinstance(summ_var, list):
        summary_var_dict = {}
        for summary_var in summ_var:
            print(summary_var)
            summary_var_dict[summary_var] = read_code_second_summary_npy(date, summary_var)
        return summary_var_dict
    else:
        outputdir = f'{PLATFORM_PATH_DICT["v_path"]}/StockData/d0_file/CodeSecondData/{date}_summary/'

        index_list = pd.read_csv(outputdir + f'{date}_1s_{summ_var}_mat_index.csv')['localtime'].to_list()
        columns_list = pd.read_csv(outputdir + f'{date}_1s_{summ_var}_mat_columns.csv')['SecuCode'].to_list()
        columns_list = [expand_stockcode(x) for x in columns_list]

        memmap_ret = np.memmap(outputdir + f'{date}_1s_{summ_var}_mat.npy',
                               mode='r', shape=(len(index_list), len(columns_list)), dtype=np.float64)
        df_mem = pd.DataFrame(memmap_ret, index=index_list, columns=columns_list)
        return df_mem


def get_lastprice(date, df_price=None, df_close_nmin=None, bar_mode=8):
    if df_price is None:
        df_price = get_price(date, date).reset_index()[
            ['SecuCode', 'PreClosePrice', 'ClosePrice']].set_index('SecuCode')
    else:
        df_price = df_price.copy(deep=True)

    if df_close_nmin is None:
        if bar_mode == 8:
            df_close_nmin = get_n_min_stock_daily_data(date, period='30min')
            df_close_nmin = df_close_nmin[(df_close_nmin['bar'] != 8)]
            df_close_nmin['bar'] = np.abs(df_close_nmin['bar'])
        else:
            df_close_nmin = get_n_min_stock_daily_data(date, period='5min')
            df_close_nmin = df_close_nmin[(df_close_nmin['bar'] != -1)]
            df_close_nmin['bar'] += 1

        df_price30 = pd.pivot_table(df_close_nmin, index='code', columns='bar', values='close')[[i for i in range(1, bar_mode + 1)]]
    else:
        df_price30 = df_close_nmin.copy(deep=True)

    last_price = pd.concat([df_price.rename({'PreClosePrice': 0}, axis='columns'), df_price30], axis=1)[[i for i in range(bar_mode)]].fillna(method='ffill', axis=1)
    return last_price


def get_30min_price_multi_mode_longtime(date_list, mode_list='_0935', mode='mode-first', close_output=False):
    close_price_dict = {}
    df_date_closeprice = get_price(date_list[0], date_list[-1]).reset_index()[
        ['QueryDate', 'SecuCode', 'PreClosePrice', 'ClosePrice']].rename(
        {'SecuCode': 'code'}, axis='columns')

    for date, df_close in df_date_closeprice.groupby('QueryDate'):
        df_close = df_close[['code', 'PreClosePrice', 'ClosePrice']].set_index('code')
        close_price_dict[str(date)] = df_close

    if isinstance(mode_list, str):
        price_dict = {}
        for date in date_list:
            print('获取数据', date, mode_list)
            if mode_list == '_open':
                df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min='_0935')[
                    ['code', 'bar', 'open']].rename({'open': 'vwap'}, axis='columns')
            else:
                df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min=mode_list)[
                    ['code', 'bar', 'vwap']]

            df_30_min_price = df_30_min_price[(df_30_min_price['bar'] != 0) & (df_30_min_price['bar'] != 8)]
            df_30_min_price['bar'] = np.abs(df_30_min_price['bar'])
            df_30_min_price = pd.pivot_table(df_30_min_price, values='vwap', index='code', columns='bar')

            df_30_min_price = pd.concat([df_30_min_price, close_price_dict[date]], axis=1)
            price_dict[date] = df_30_min_price

        if close_output:
            return price_dict, close_price_dict
        else:
            return price_dict

    price_dict = {}
    if mode == 'mode-first':
        for mode_l in mode_list:
            price_mode_dict = {}
            for date in date_list:
                print('获取数据', date, mode_l)
                if mode_l == '_open':
                    df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min='_0935')[
                        ['code', 'bar', 'open']].rename({'open': 'vwap'}, axis='columns')
                else:
                    df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min=mode_l)[
                        ['code', 'bar', 'vwap']]

                df_30_min_price = df_30_min_price[(df_30_min_price['bar'] != 0) & (df_30_min_price['bar'] != 8)]
                df_30_min_price['bar'] = np.abs(df_30_min_price['bar'])
                df_30_min_price = pd.pivot_table(df_30_min_price, values='vwap', index='code', columns='bar')

                df_30_min_price = pd.concat([df_30_min_price, close_price_dict[date]], axis=1)
                price_mode_dict[date] = df_30_min_price
            price_dict[mode_l] = price_mode_dict
    else:
        for date in date_list:
            price_date_dict = {}
            for mode_l in mode_list:
                print('获取数据', date, mode_l)
                if mode_l == '_open':
                    df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min='_0935')[
                        ['code', 'bar', 'open']].rename({'open': 'vwap'}, axis='columns')
                else:
                    df_30_min_price = get_n_min_stock_daily_data(date, period='30min', mode30min=mode_l)[
                        ['code', 'bar', 'vwap']]

                df_30_min_price = df_30_min_price[(df_30_min_price['bar'] != 0) & (df_30_min_price['bar'] != 8)]
                df_30_min_price['bar'] = np.abs(df_30_min_price['bar'])
                df_30_min_price = pd.pivot_table(df_30_min_price, values='vwap', index='code', columns='bar')

                df_30_min_price = pd.concat([df_30_min_price, close_price_dict[date]], axis=1)
                price_date_dict[mode_l] = df_30_min_price
            price_dict[date] = price_date_dict
    if close_output:
        return price_dict, close_price_dict
    else:
        return price_dict


def get_n_min_vwap_twap_price(date, df_price=None, df_n_min=None, price_type='vwap', mode30min='_0935', mode5min='', bar_mode=8, with_daily=True):
    if bar_mode == 8:
        if df_n_min is None:
            if isinstance(mode30min, str): df_n_min = get_n_min_stock_daily_data(date, period='30min', mode30min=mode30min)
            else:
                conlist = []
                for delay_min in mode30min:
                    if int(delay_min) > 0: df_data = get_n_min_stock_daily_data(date, period='30min', mode30min=f'_0935_delay{int(delay_min)}min')
                    else: df_data = get_n_min_stock_daily_data(date, period='30min', mode30min=f'_0935')

                    df_data = df_data[df_data['bar'].astype('int').isin(mode30min[delay_min])]
                    conlist.append(df_data)
                df_n_min = pd.concat(conlist, axis=0)
        else: df_n_min = df_n_min.copy(deep=True)

        df_n_min = df_n_min[~df_n_min['bar'].isin([0, 8])]
        df_n_min['bar'] = np.abs(df_n_min['bar'])
        if not isinstance(price_type, str):
            data_res = [
                pd.pivot_table(df_n_min, index='code', columns='bar', values=pt)[[i for i in range(1, bar_mode + 1)]] for pt in price_type
            ]
        else:
            data_res = pd.pivot_table(df_n_min, index='code', columns='bar', values=price_type)[[i for i in range(1, bar_mode + 1)]]
    else:
        if df_n_min is None:
            if isinstance(mode5min, str):
                df_n_min = get_n_min_stock_daily_data(date, '5min', mode5min=mode5min)
                df_n_min = df_n_min[df_n_min['bar_5'] >= 0]
                df_n_min['bar_5'] += 1
            else:
                conlist = []
                for delay_10s in mode5min:
                    if int(delay_10s) > 0: df_data = get_n_min_stock_daily_data(date, period='5min', mode5min=f'_delay{int(delay_10s)}tens')
                    else: df_data = get_n_min_stock_daily_data(date, period='5min', mode5min='')
                    df_data = df_data[df_data['bar_5'] >= 0]
                    df_data['bar_5'] += 1
                    df_data = df_data[df_data['bar_5'].astype('int').isin(mode5min[delay_10s])]

                    conlist.append(df_data)
                df_n_min = pd.concat(conlist, axis=0)
        else:
            df_n_min = df_n_min.copy(deep=True)
            df_n_min = df_n_min[df_n_min['bar_5'] >= 0]
            df_n_min['bar_5'] += 1
        if not isinstance(price_type, str):
            data_res = [
                pd.pivot_table(df_n_min, index='code', columns='bar_5', values=pt)[[i for i in range(1, bar_mode + 1)]] for pt in price_type
            ]
        else:
            data_res = pd.pivot_table(df_n_min, index='code', columns='bar_5', values=price_type)[[i for i in range(1, bar_mode + 1)]]

    if not with_daily:
        if isinstance(data_res, list): return [dr.reset_index() for dr in data_res]
        else: return data_res.reset_index()

    if df_price is None: df_price = get_price(date, date).reset_index()[['SecuCode', 'ClosePrice', 'PreClosePrice']].rename({'SecuCode': 'code'}, axis='columns').set_index('code')
    else: df_price = df_price[['SecuCode', 'ClosePrice', 'PreClosePrice']].copy(deep=True).rename({'SecuCode': 'code'}, axis='columns').set_index('code')
    if isinstance(data_res, list):
        data_res = [
            pd.concat([df_price, dr], axis=1).replace([-np.inf, np.inf], np.NAN).fillna(method='ffill', axis=1).fillna(0).reset_index() for dr in data_res
        ]
    else:
        data_res = pd.concat([df_price, data_res], axis=1).replace([-np.inf, np.inf], np.NAN).fillna(method='ffill', axis=1).fillna(0).reset_index()
    return data_res


def get_intradaily_analysis_all_price_data(curdate, bar_mode=8, mode_nmin=None):
    df_1_min_closeprice_data = get_n_min_stock_daily_data(curdate, '1min')[['code', 'time', 'close']]
    df_1_min_closeprice_data['code'] = df_1_min_closeprice_data['code'].apply(lambda x: expand_stockcode(x))
    time_list = sorted(df_1_min_closeprice_data['time'].unique())
    df_1min_mat_close = pd.pivot_table(df_1_min_closeprice_data, index='code', columns='time', values='close')

    if bar_mode == 8:
        mode_nmin = '_0935' if mode_nmin is None else mode_nmin
        df_mat_vwap, df_mat_close = get_n_min_vwap_twap_price(curdate, price_type=['vwap', 'close'], mode30min=mode_nmin, bar_mode=bar_mode, with_daily=False)
    else:
        mode_nmin = '' if mode_nmin is None else mode_nmin
        df_mat_vwap, df_mat_close = get_n_min_vwap_twap_price(curdate, price_type=['vwap', 'close'], mode5min=mode_nmin, bar_mode=bar_mode, with_daily=False)

    df_mat_vwap = df_mat_vwap.set_index('code')
    df_mat_close = df_mat_close.set_index('code')
    df_price = get_price(curdate, curdate).reset_index()[['SecuCode', 'PreClosePrice', 'ClosePrice']].set_index('SecuCode')
    df_lastprice = get_lastprice(curdate, df_price, df_mat_close, bar_mode=bar_mode)

    return df_price, df_1min_mat_close, df_mat_vwap, df_mat_close, df_lastprice, time_list


def get_intradaily_delay(curdate, flag, period='5min', index_mode='delay', max_delay_ti1=20, max_delay=25, max_delay_5m=24, target_date=None, start_date=None, end_date=None):
    file_path = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/D0_FrenshTime/{curdate}_longtime_delay_min{flag}.csv'
    df_delay = pd.read_csv(file_path, dtype={"Date": 'str'})

    if start_date is not None: df_delay = df_delay[df_delay['Date'].astype('int') >= int(start_date)]
    if end_date is not None: df_delay = df_delay[df_delay['Date'].astype('int') <= int(end_date)]

    if period == '5min':
        df_delay_5m = df_delay.rename({f'{bar}': bar for bar in range(1, 49)}, axis='columns').set_index('Date')
        
        if index_mode == 'over_max_flag':
            df_delay_5m = pd.DataFrame(np.where(df_delay_5m > max_delay_5m, df_delay_5m, np.nan), index=df_delay_5m.index, columns=df_delay_5m.columns)
            df_delay_5m = df_delay_5m.reset_index().melt(id_vars='Date')
            df_delay_5m = df_delay_5m[~df_delay_5m['value'].isna()]
            mode5min = {date: np.abs(df_over['variable']).to_list() for date, df_over in df_delay_5m.groupby('Date')}
            if target_date is None: return mode5min
            else: return mode5min.get(target_date, [])
        else:
            df_delay_5m = pd.DataFrame(np.where(df_delay_5m <= max_delay_5m, df_delay_5m, max_delay_5m), index=df_delay_5m.index, columns=df_delay_5m.columns)
            df_delay_5m = df_delay_5m.astype('int').reset_index()
            if index_mode == 'delay':
                mode5min = {date: {delay: df_dly['index'].to_list() for delay, df_dly in
                                df_date.set_index('Date').T[date].reset_index().groupby(date)} if int(date) >= 20241210 else '' for date, df_date in df_delay_5m.groupby('Date')}
            elif index_mode == 'bar':
                mode5min = df_delay_5m.set_index('Date').to_dict(orient='index')
            else: raise ValueError
            if target_date is None: return mode5min
            else: return mode5min.get(target_date, '')
    else:
        df_delay = df_delay.rename({f'{bar}': bar for bar in range(1, 8)}, axis='columns').rename({'8': -8}, axis='columns').set_index('Date')
        if index_mode == 'over_max_flag':
            df_delay = pd.DataFrame(np.where(df_delay > max_delay, df_delay, np.nan), index=df_delay.index, columns=df_delay.columns)
            df_delay[1] = np.where(df_delay[1].fillna(0) > max_delay_ti1, df_delay[1], np.nan)
            df_delay = df_delay.reset_index().melt(id_vars='Date')
            df_delay = df_delay[~df_delay['value'].isna()]
            mode30min = {date: np.abs(df_over['variable']).to_list() for date, df_over in df_delay.groupby('Date')}
            mode30min['20241008'] = [1, 2, 3]

            if target_date is None: return mode30min
            else: return mode30min.get(target_date, [])

        df_delay = pd.DataFrame(np.where(df_delay <= max_delay, df_delay, max_delay), index=df_delay.index, columns=df_delay.columns)
        df_delay[1] = np.where(df_delay[1] <= max_delay_ti1, df_delay[1], max_delay_ti1)
        df_delay = df_delay.astype('int').reset_index()
        if index_mode == 'delay':
            mode30min = {date: {
                delay: df_dly['index'].to_list()
                for delay, df_dly in df_date.set_index('Date').T[date].reset_index().groupby(date)}
            if date not in ['20240207', '20240208'] else '_0935' for date, df_date in df_delay.groupby('Date')}
        elif index_mode == 'bar':
            mode30min = df_delay.set_index('Date').to_dict(orient='index')
        else: raise ValueError

        if target_date is None: return mode30min
        else: return mode30min.get(target_date, '_0935')


def get_intradaily_delay_bar_mode(curdate, target_date=None, return_period=None, index_mode='delay', max_delay_ti1=20, max_delay=25, max_delay_5m=24):
    if return_period is None:
        mode5min = get_intradaily_delay_bar_mode(curdate, target_date=target_date, return_period='5min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay, max_delay_5m=max_delay_5m)
        mode30min = get_intradaily_delay_bar_mode(curdate, target_date=target_date, return_period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay, max_delay_5m=max_delay_5m)
        return mode30min, mode5min
    elif return_period == '5min':
        return get_intradaily_delay(curdate, '_5m', period='5min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=target_date, start_date=None, end_date=None)
    elif return_period == '30min':
        return get_intradaily_delay(curdate, '', period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=target_date, start_date=None, end_date=None)
    elif return_period == 'ms':
        return get_intradaily_delay(curdate, '_ls_ms', period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=target_date, start_date='20241204', end_date=None)
    elif return_period == 'zj':
        res_30 = get_intradaily_delay(curdate, '', period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=None, start_date='20240708', end_date='20240930')
        res_zj = get_intradaily_delay(curdate, '_ls_zj', period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=None, start_date='20241008', end_date=None)
        res_30.update(res_zj)
        
        if target_date is None: return res_30
        if index_mode == 'over_max_flag': 
            return res_30.get(target_date, [])
        else: return res_30.get(target_date, '_0935')

    elif return_period == 'msls2':
        res_30 = get_intradaily_delay(curdate, '_ls_msls2', period='30min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=None, start_date='20250320', end_date='20250423')
        res_5 = get_intradaily_delay(curdate, '_ls_5m_msls2', period='5min', index_mode=index_mode, max_delay_ti1=max_delay_ti1, max_delay=max_delay_ti1, max_delay_5m=max_delay_ti1, target_date=None, start_date='20250424', end_date=None)
        res_30.update(res_5)

        if target_date is None: return res_30
        if index_mode == 'over_max_flag':
            return res_30.get(target_date, [])
        else:
            if int(curdate) > 20250423: return res_30.get(target_date, '')
            else: return res_30.get(target_date, '_0935')
    else:
        raise ValueError
    

def get_intradaily_delay_bar_mode_details(curdate, target_date=None):
    mode30min, mode5min = get_intradaily_delay_bar_mode(curdate)
    mode30min_zj = get_intradaily_delay_bar_mode(curdate, return_period='zj')
    mode30min_ms = get_intradaily_delay_bar_mode(curdate, return_period='ms')
    mode30min_msls2 = get_intradaily_delay_bar_mode(curdate, return_period='msls2')

    over_max_flag_dict = {
        '30min': get_intradaily_delay_bar_mode(curdate, return_period='30min', index_mode='over_max_flag', target_date=target_date),
        '5min': get_intradaily_delay_bar_mode(curdate, return_period='5min', index_mode='over_max_flag', target_date=target_date),
        'zj': get_intradaily_delay_bar_mode(curdate, return_period='zj', index_mode='over_max_flag', target_date=target_date),
        'ms': get_intradaily_delay_bar_mode(curdate, return_period='ms', index_mode='over_max_flag', target_date=target_date),
        'msls2': get_intradaily_delay_bar_mode(curdate, return_period='msls2', index_mode='over_max_flag', target_date=target_date),
    }
    return mode30min, mode5min, mode30min_zj, mode30min_ms, mode30min_msls2, over_max_flag_dict


def get_intradaily_delay_drop_bar_list(curdate, product, bar_mode):
    if bar_mode == 8:
        if product not in ProductionList_AlphaShort:
            quota_drop_list = get_intradaily_delay_bar_mode(curdate, target_date=curdate, return_period='30min', index_mode='over_max_flag')
        else:
            quota_drop_list = get_intradaily_delay_bar_mode(curdate, target_date=curdate, return_period=ProductionDict_LongShortClss[product], index_mode='over_max_flag')
    else:
        if product not in ProductionList_AlphaShort:
            quota_drop_list = get_intradaily_delay_bar_mode(curdate, target_date=curdate, return_period='5min', index_mode='over_max_flag')
        else:
            quota_drop_list = get_intradaily_delay_bar_mode(curdate, target_date=curdate, return_period=ProductionDict_LongShortClss[product], index_mode='over_max_flag')
    return quota_drop_list


def get_intradaily_analysis_all_mat_data(
        curdate, predate, quota_prod, df_price, df_lastprice, df_30min_mat_close, vol_2_mat_mode='lastprice',
        quota_path=None, pre_alpha_name=None, alpha_name=None, mode='dest-pos', bar_mode=8, bar_mode_pre=8,
        drop_bar_delay=True, scale_pqt_bar_mode=True):
    df_price = df_price.copy(deep=True)
    df_lastprice = df_lastprice.copy(deep=True)
    df_30min_mat_close = df_30min_mat_close.copy(deep=True)

    if pre_alpha_name is None: pre_alpha_name = get_production_list_trading(predate, quota_prod, 'alpha', drop_simu=False)
    if alpha_name is None: alpha_name = get_production_list_trading(curdate, quota_prod, 'alpha', drop_simu=False)

    if pre_alpha_name is None: intra_weight_pre = pd.DataFrame()
    else: intra_weight_pre = get_intrati8_weight_fast(predate, pre_alpha_name, bar_mode=bar_mode_pre)

    if alpha_name is None: intra_weight = pd.DataFrame()
    else: intra_weight = get_intrati8_weight_fast(curdate, alpha_name, bar_mode=bar_mode)

    if (pre_alpha_name is not None) and (alpha_name is not None):
        intra_weight_diff = get_diff_intrati8_weight_simple(intra_weight_pre, intra_weight, bar_mode=bar_mode, bar_mode_pre=bar_mode_pre)
    else:
        intra_weight_diff = pd.DataFrame()

    quota_drop_list = get_intradaily_delay_drop_bar_list(curdate, quota_prod, bar_mode)
    if quota_prod in ProductionList_AlphaShort_Short:
        intra_quota = get_simulate_diff_volume_mat_by_quota(
            curdate, predate, quota_prod, filepath=quota_path, ret_df_mode='Bar8', bar_mode=bar_mode,
            short_mode=True, drop_bar_delay=drop_bar_delay, drop_quota_list=quota_drop_list).rename(
            {f'volume_bar_{i}': f'Ti{i}_Weight' for i in range(1, bar_mode + 1)}, axis='columns')
    else:
        intra_quota = get_simulate_diff_volume_mat_by_quota(
            curdate, predate, quota_prod, filepath=quota_path, ret_df_mode='Bar8', bar_mode=bar_mode,
            drop_bar_delay=drop_bar_delay, drop_quota_list=quota_drop_list).rename(
            {f'volume_bar_{i}': f'Ti{i}_Weight' for i in range(1, bar_mode + 1)}, axis='columns')

    if not intra_quota.empty: intra_weight_quota = get_intrati8_weight_by_quota(intra_quota, df_price, df_30min_mat_close, bar_mode=bar_mode)
    else: intra_weight_quota = pd.DataFrame()

    if scale_pqt_bar_mode:
        df_price_pre = get_price(predate, predate).reset_index()
        quota_value = get_position_values(predate, predate, quota_prod, df_price=df_price_pre, mode='quota', quota_bar=bar_mode, bar_mode=bar_mode)
        holdmv = get_position_values(predate, predate, quota_prod, df_price=df_price_pre)
        if quota_value != 0:
            scale_pqt_bar = holdmv / quota_value
            print(curdate, quota_prod, f'scale昨收quota：{holdmv}/{quota_value}={scale_pqt_bar}')
        else:
            scale_pqt_bar = 1
    else: scale_pqt_bar = 1
    if quota_prod in ProductionList_AlphaShort_Short: intra_quota_diff_volume = get_simulate_diff_volume_mat_by_quota(
        curdate, predate, quota_prod, filepath=quota_path, mode=mode, short_mode=True, bar_mode=bar_mode,
        drop_bar_delay=drop_bar_delay, scale_pqt_bar=scale_pqt_bar, drop_quota_list=quota_drop_list)
    else: intra_quota_diff_volume = get_simulate_diff_volume_mat_by_quota(
        curdate, predate, quota_prod, filepath=quota_path, mode=mode, bar_mode=bar_mode,
        drop_bar_delay=drop_bar_delay, scale_pqt_bar=scale_pqt_bar, drop_quota_list=quota_drop_list)
    intra_weight_diff_volume = get_simulate_diff_volume_mat_by_weight(curdate, predate, alpha_name, pre_alpha_name, df_price=df_price, df_lastp30=df_lastprice, mode=vol_2_mat_mode, matweight_diff=intra_weight_diff, bar_mode=bar_mode, bar_mode_pre=bar_mode_pre)

    if intra_weight.empty: intra_weight = intra_weight_quota
    if intra_weight_diff.empty: intra_weight_diff = intra_quota_diff_volume
    if intra_weight_diff_volume.empty: intra_weight_diff_volume = intra_quota_diff_volume

    return intra_weight, intra_weight_pre, intra_weight_diff, intra_weight_diff_volume, intra_quota_diff_volume, intra_quota, intra_weight_quota


def get_index_return_longtime(date_list, index_name, mode='', index_price_df=None):
    if mode == 'long-dict':
        df_porp = pd.DataFrame(index_name).T
        conlist = []
        for index_n in df_porp.columns.to_list():
            if index_price_df is None:
                df_index_price = get_indexprice(
                    date_list[0], date_list[-1], IndexName=index_n).reset_index().set_index('QueryDate')
            else:
                df_index_price = index_price_df.copy(deep=True)[
                    (index_price_df['IndexName'] == index_n) & (index_price_df['QueryDate'].isin(date_list))]
                df_index_price = df_index_price.set_index('QueryDate')
            df_index_price['return'] = \
                (df_index_price['ClosePrice'] - df_index_price['PrevClosePrice']) / df_index_price['PrevClosePrice']
            df_index_price = pd.concat([df_index_price, df_porp], axis=1)
            df_index_price['return'] *= df_index_price[index_n]
            conlist.append(df_index_price[['return']].copy(deep=True))
        df_index_price = pd.concat(conlist, axis=1).sum(axis=1).reset_index()
        index_ret_dict = {date: ret for date, ret in df_index_price[['index', 0]].values}
        return index_ret_dict

    if isinstance(index_name, str):
        if index_price_df is None:
            df_index_price = get_indexprice(date_list[0], date_list[-1], IndexName=index_name).reset_index()
        else:
            df_index_price = index_price_df.copy(deep=True)[
                (index_price_df['IndexName'] == index_name) & (index_price_df['QueryDate'].isin(date_list))]
            df_index_price = df_index_price.set_index('QueryDate')
        df_index_price['return'] = (df_index_price['ClosePrice'] - df_index_price['PrevClosePrice']) / \
                                   df_index_price['PrevClosePrice']
        index_ret_dict = {date: ret for date, ret in df_index_price[['QueryDate', 'return']].values}
        return index_ret_dict
    else:
        conlist = []
        for index_n in index_name:
            if index_name[index_n] == 0:
                continue
            df_index_price = get_indexprice(date_list[0], date_list[-1], IndexName=index_n).reset_index()
            df_index_price['return'] = \
                (df_index_price['ClosePrice'] - df_index_price['PrevClosePrice']) / df_index_price['PrevClosePrice']
            df_index_price['return'] *= index_name[index_n]
            conlist.append(df_index_price.set_index('QueryDate')[['return']])
        df_index_price = pd.concat(conlist, axis=1).sum(axis=1).reset_index()
        index_ret_dict = {date: ret for date, ret in df_index_price[['QueryDate', 0]].values}
        return index_ret_dict


def calculate_intradaily_trading_process_analysis_result(
        time_list, index_1minret, df_position, df_quota_pre, df_trades,
        df_price, df_1min_mat_close,
        intra_weight, intra_weight_diff, intra_weight_quota=None,
        fee_ratio=0.00015, tax_ratio=STAMP_RATE, add_exec=False, return_mode='array', bar_mode=8, rename_bar_dict=None, qt_ps_ret_mv_real=False):
    if rename_bar_dict is None:
        if bar_mode == 8: rename_bar_dict = DICT_TIME_2_BAR_30MIN
        else: rename_bar_dict = DICT_TIME_2_BAR_5MIN

    df_position = df_position[['SecuCode', 'PreCloseVolume']].set_index('SecuCode')
    df_quota_pre = df_quota_pre.rename({'Volume': 'PreCloseVolume'}, axis='columns').set_index('SecuCode')[['PreCloseVolume']]
    df_trades = df_trades.copy(deep=True)

    df_trades['time'] = df_trades['time'].apply(lambda x: format_trades_time_2_minute(x))

    df_premv = pd.concat([df_position, df_price], axis=1).fillna(0)
    df_premv['PreMV'] = df_premv['PreClosePrice'] * df_premv['PreCloseVolume']
    pre_mv_sum = abs(df_premv['PreMV'].sum())
    df_premv['PreWeight'] = df_premv['PreMV'] / pre_mv_sum

    df_quota_pre = pd.concat([df_quota_pre, df_price], axis=1).fillna(0)
    df_quota_pre['PreQuotaWeight'] = df_quota_pre['PreClosePrice'] * df_quota_pre['PreCloseVolume']
    if qt_ps_ret_mv_real:
        quota_pre_mv_sum = abs(df_quota_pre['PreQuotaWeight'].sum())
        df_quota_pre['PreQuotaWeight'] = df_quota_pre['PreQuotaWeight'] / quota_pre_mv_sum
    else:
        df_quota_pre['PreQuotaWeight'] = df_quota_pre['PreQuotaWeight'] / pre_mv_sum

    # 计算 格式化交易数据
    df_trades['Volume'] = df_trades['Volume'] * (1 - 2 * df_trades['LongShort'])
    df_trades['TradesCos'] = df_trades['Volume'] * df_trades['Price']
    df_trades['LongShort'] = df_trades['LongShort'].astype('int')

    if add_exec:
        if 'status' in df_trades.columns.to_list():
            df_trades_exec = df_trades[df_trades['status'] != 1].copy(deep=True)
        else:
            df_trades_exec = df_trades.copy(deep=True)
        df_trades_exec = df_trades_exec.groupby(['time', 'SecuCode', 'LongShort'])[['Volume']].sum().rename(
            {'Volume': 'VolumeExec'}, axis='columns')
        df_trades = df_trades.groupby(['time', 'SecuCode', 'LongShort'])[['TradesCos', 'Volume']].sum()
        df_trades = pd.concat([df_trades, df_trades_exec], axis=1).fillna(0).reset_index()
        df_trades_long = df_trades[df_trades['LongShort'] == 0][['SecuCode', 'time', 'TradesCos', 'Volume', 'VolumeExec']]
        df_trades_short = df_trades[df_trades['LongShort'] == 1][['SecuCode', 'time', 'TradesCos', 'Volume', 'VolumeExec']]
        df_pivot_long = pd.pivot_table(df_trades_long, index='SecuCode', columns='time', values=['Volume', 'TradesCos', 'VolumeExec']).fillna(0)
        df_pivot_short = pd.pivot_table(df_trades_short, index='SecuCode', columns='time', values=['Volume', 'TradesCos', 'VolumeExec']).fillna(0)
    else:
        df_trades = df_trades.groupby(['time', 'SecuCode', 'LongShort'])[['TradesCos', 'Volume']].sum().reset_index()
        df_trades_long = df_trades[df_trades['LongShort'] == 0][['SecuCode', 'time', 'TradesCos', 'Volume']]
        df_trades_short = df_trades[df_trades['LongShort'] == 1][['SecuCode', 'time', 'TradesCos', 'Volume']]
        df_pivot_long = pd.pivot_table(df_trades_long, index='SecuCode', columns='time', values=['Volume', 'TradesCos']).fillna(0)
        df_pivot_short = pd.pivot_table(df_trades_short, index='SecuCode', columns='time', values=['Volume', 'TradesCos']).fillna(0)

    union_index = df_pivot_long.index.union(df_pivot_short.index).union(df_1min_mat_close.index).union(df_premv.index).union(intra_weight_diff.index).union(df_quota_pre.index)
    df_close_1min = df_1min_mat_close.copy(deep=True).reindex(union_index, columns=time_list, fill_value=np.nan)
    if df_pivot_short.empty:
        short_value = pd.DataFrame(index=union_index, columns=time_list).fillna(0)
        short_volume = pd.DataFrame(index=union_index, columns=time_list).fillna(0)
    else:
        short_value = df_pivot_short['TradesCos'].reindex(union_index, columns=time_list, fill_value=0)
        short_volume = df_pivot_short['Volume'].reindex(union_index, columns=time_list, fill_value=0)

    if df_pivot_long.empty:
        long_value = pd.DataFrame(index=union_index, columns=time_list).fillna(0)
        long_volume = pd.DataFrame(index=union_index, columns=time_list).fillna(0)
    else:
        long_value = df_pivot_long['TradesCos'].reindex(union_index, columns=time_list, fill_value=0)
        long_volume = df_pivot_long['Volume'].reindex(union_index, columns=time_list, fill_value=0)

    if add_exec:
        long_volume_exec = df_pivot_long['VolumeExec'].reindex(union_index, columns=time_list, fill_value=0)
        short_volume_exec = df_pivot_short['VolumeExec'].reindex(union_index, columns=time_list, fill_value=0)
    df_trades_buy = long_value / pre_mv_sum
    df_trades_sell = short_value / pre_mv_sum
    df_trades_diff = np.abs(df_trades_buy) - np.abs(df_trades_sell)

    df_trades_buy = np.cumsum(df_trades_buy, axis=1)
    df_trades_sell = np.cumsum(df_trades_sell, axis=1)
    df_trades_diff = np.cumsum(df_trades_diff, axis=1)

    # 计算日内实时买卖交易占比，矩阵，以及diff 矩阵, 单边换手率
    long_value = np.cumsum(long_value, axis=1)
    long_volume = np.cumsum(long_volume, axis=1)
    short_value = np.cumsum(short_value, axis=1)
    short_volume = np.cumsum(short_volume, axis=1)

    net_openning_volume = long_volume + short_volume
    if add_exec:
        long_volume_exec = np.cumsum(long_volume_exec, axis=1)
        short_volume_exec = np.cumsum(short_volume_exec, axis=1)
        net_openning_volume_exec = long_volume_exec + short_volume_exec

    which_min = long_volume <= abs(short_volume)
    min_long_short_volume = which_min * long_volume + (which_min - 1) * short_volume

    long_price_avr = (long_value / long_volume).replace([np.inf, -np.inf], np.nan).fillna(0)
    short_price_avr = (short_value / short_volume).replace([np.inf, -np.inf], np.nan).fillna(0)

    unrealized_pnl = (((df_close_1min - long_price_avr * (1 + fee_ratio)) * (net_openning_volume >= 0) +
                       (df_close_1min - short_price_avr * (1 - tax_ratio - fee_ratio)) * (
                               net_openning_volume < 0)) * net_openning_volume) / pre_mv_sum
    realized_pnl = ((short_price_avr * (1 - tax_ratio - fee_ratio) -
                     long_price_avr * (1 + fee_ratio)) * min_long_short_volume) / pre_mv_sum
    trades_pnl_total = unrealized_pnl + realized_pnl

    position_pre_mat = pd.DataFrame(
        np.tile(df_position[['PreCloseVolume']].copy(deep=True).reindex(union_index, fill_value=0), 239),
        index=union_index, columns=time_list)
    price_pre_mat = pd.DataFrame(
        np.tile(df_price[['PreClosePrice']].copy(deep=True).reindex(union_index, fill_value=0), 239),
        index=union_index, columns=time_list)
    weight_pre_mat = pd.DataFrame(
        np.tile(df_premv[['PreWeight']].copy(deep=True).reindex(union_index, fill_value=0), 239),
        index=union_index, columns=time_list)
    quota_weight_pre_mat = pd.DataFrame(
        np.tile(df_quota_pre[['PreQuotaWeight']].copy(deep=True).reindex(union_index, fill_value=0), 239),
        index=union_index, columns=time_list)

    df_pos_ret_mat = (df_close_1min - price_pre_mat) / price_pre_mat * weight_pre_mat
    df_quota_ret_mat = (df_close_1min - price_pre_mat) / price_pre_mat * quota_weight_pre_mat
    df_alpha = np.array((df_pos_ret_mat + trades_pnl_total).sum(axis=0)) - np.array(index_1minret)
    pos_alpha_ret = np.array((df_pos_ret_mat).sum(axis=0)) - np.array(index_1minret)
    quota_alpha_ret = np.array((df_quota_ret_mat).sum(axis=0)) - np.array(index_1minret)
    pos_diff_ret = df_pos_ret_mat - df_quota_ret_mat

    df_weight_mat = (net_openning_volume + position_pre_mat) * df_close_1min
    real_weight_mat = df_weight_mat / df_weight_mat.sum(axis=0)
    if add_exec:
        df_weight_mat_exec = (net_openning_volume_exec + position_pre_mat) * df_close_1min
        real_weight_mat_exec = df_weight_mat_exec / df_weight_mat_exec.sum(axis=0)

    intra_weight = intra_weight.copy(deep=True).rename(rename_bar_dict, axis='columns').reindex(
        index=union_index, columns=time_list, fill_value=np.nan).fillna(method='bfill', axis=1).fillna(0)
    intra_weight_diff = intra_weight_diff.copy(deep=True).rename(rename_bar_dict, axis='columns').reindex(
        index=union_index, columns=time_list, fill_value=np.nan).fillna(method='bfill', axis=1).fillna(0)
    intra_weight_diff = pd.DataFrame(
        np.where(intra_weight_diff > 0, 1, 0), index=intra_weight_diff.index, columns=intra_weight_diff.columns)

    mat_weight_diff = np.abs(real_weight_mat - intra_weight)
    if add_exec:
        mat_weight_diff_exec = np.abs(real_weight_mat_exec - intra_weight) / 2

    mat_weight_diff_long = mat_weight_diff * intra_weight_diff
    mat_weight_diff_short = mat_weight_diff * (1 - intra_weight_diff)
    mat_weight_diff = mat_weight_diff / 2
    
    if intra_weight_quota is not None:
        intra_weight_quota = intra_weight_quota.copy(deep=True).rename(rename_bar_dict, axis='columns').reindex(
            index=union_index, columns=time_list, fill_value=np.nan).fillna(method='bfill', axis=1).fillna(0)
        mat_weight_quota_diff = np.abs(real_weight_mat - intra_weight_quota) / 2
        # (mat_weight_quota_diff[mat_weight_quota_diff.sum(axis=1) != 0] * 10000).to_csv(f'{LOG_TEMP_PATH}temp.csv')
        mat_weight_diff = np.array(mat_weight_diff.sum(axis=0))
        mat_weight_quota_diff = np.array(mat_weight_quota_diff.sum(axis=0))
        weight_diff_tuple = (mat_weight_diff, mat_weight_quota_diff)
    else:
        mat_weight_diff = np.array(mat_weight_diff.sum(axis=0))
        weight_diff_tuple = (mat_weight_diff, )

    if return_mode == 'array':
        trades_pnl_total = np.array(trades_pnl_total.sum(axis=0))

    unrealized_pnl = np.array(unrealized_pnl.sum(axis=0))
    realized_pnl = np.array(realized_pnl.sum(axis=0))
    df_trades_buy = np.array(df_trades_buy.sum(axis=0))
    df_trades_sell = np.array(df_trades_sell.sum(axis=0))
    df_trades_diff = np.array(df_trades_diff.sum(axis=0))
    mat_weight_diff_long = np.array(mat_weight_diff_long.sum(axis=0))
    mat_weight_diff_short = np.array(mat_weight_diff_short.sum(axis=0))
    if add_exec:
        mat_weight_diff_exec = np.array(mat_weight_diff_exec.sum(axis=0))

    if add_exec:
        return pre_mv_sum, df_alpha, pos_alpha_ret, quota_alpha_ret, pos_diff_ret, trades_pnl_total, unrealized_pnl, realized_pnl, \
               df_trades_buy, df_trades_sell, df_trades_diff, \
               weight_diff_tuple, mat_weight_diff_long, mat_weight_diff_short, real_weight_mat, mat_weight_diff_exec
    else:
        return pre_mv_sum, df_alpha, pos_alpha_ret, quota_alpha_ret, pos_diff_ret, trades_pnl_total, unrealized_pnl, realized_pnl, \
               df_trades_buy, df_trades_sell, df_trades_diff, \
               weight_diff_tuple, mat_weight_diff_long, mat_weight_diff_short, real_weight_mat


def calculate_intradaily_vwap_simu_analysis_result(
        df_trade_volume, df_30min_mat_vwap, df_1min_mat_close, time_list,
        init_mv=2 * 10 ** 8, fee_ratio=0.00015, tax_ratio=STAMP_RATE, return_mode='array', bar_mode=8, rename_bar_dict=None, smooth_fee=False):
    union_index = df_trade_volume.index.union(df_30min_mat_vwap.index).union(df_1min_mat_close.index)
    df_1min_close = df_1min_mat_close.copy(deep=True).reindex(union_index, columns=time_list, fill_value=np.nan)
    bar_list = [bar for bar in range(1, bar_mode + 1)]

    if rename_bar_dict is None:
        if bar_mode == 8: rename_bar_dict = DICT_TIME_2_BAR_30MIN
        else: rename_bar_dict = DICT_TIME_2_BAR_5MIN

    df_price30_vwap = df_30min_mat_vwap.copy(deep=True).rename(rename_bar_dict, axis='columns').reindex(union_index, fill_value=np.nan)
    df_trade_volume = df_trade_volume.rename(rename_bar_dict, axis='columns').reindex(union_index, columns=df_price30_vwap.columns, fill_value=np.nan)

    df_trade_pos = np.array(np.cumsum(df_trade_volume, axis=1).reindex(
        union_index, columns=time_list, fill_value=np.nan).fillna(method='ffill', axis=1)) * np.array(df_1min_close)
    df_trade_pos = np.nan_to_num(df_trade_pos)

    mat_fee = np.where(df_trade_volume > 0, 1 + fee_ratio, 1 - tax_ratio - fee_ratio)
    df_volume_trades = np.array(df_trade_volume) * np.array(df_price30_vwap)

    df_quota_tn = pd.DataFrame(df_volume_trades, index=union_index, columns=bar_list).fillna(0)
    df_quota_tn_long = df_quota_tn[df_quota_tn > 0].fillna(0)
    df_quota_tn_short = df_quota_tn[df_quota_tn <= 0].fillna(0)
    quota_turnover = (np.abs(df_quota_tn_long) + np.abs(df_quota_tn_short)).sum(axis=1).to_frame().rename({0: 'QuotaTurnover'}, axis='columns')

    quota_trades_ratio_long = np.array(np.cumsum(df_quota_tn_long, axis=1).rename(rename_bar_dict, axis='columns').reindex(
        union_index, columns=time_list, fill_value=np.nan).fillna(method='ffill', axis=1).sum(axis=0) / init_mv)
    quota_trades_ratio_short = np.array(np.cumsum(df_quota_tn_short, axis=1).rename(rename_bar_dict, axis='columns').reindex(
        union_index, columns=time_list, fill_value=np.nan).fillna(method='ffill', axis=1).sum(axis=0) / init_mv)

    df_trade_cost = df_volume_trades * mat_fee
    df_trade_cost = pd.DataFrame(np.nancumsum(df_trade_cost, axis=1), index=union_index, columns=bar_list).rename(rename_bar_dict, axis='columns')
    df_trade_cost = np.array(df_trade_cost.reindex(union_index, columns=time_list, fill_value=np.nan).fillna(method='ffill', axis=1).fillna(0))

    df_trade_ret = (df_trade_pos - df_trade_cost) / init_mv

    if return_mode == 'array': return np.sum(df_trade_ret, axis=0), quota_trades_ratio_short, quota_trades_ratio_long, quota_turnover

    df_trade_ret = pd.DataFrame(df_trade_ret, index=union_index, columns=time_list)

    return df_trade_ret, quota_trades_ratio_short, quota_trades_ratio_long, quota_turnover


def calculate_intradaily_analysis_result(
        curdate, product, index_1minret, df_price,
        df_1min_mat_close, df_30min_mat_vwap, time_list,
        intra_weight=None, intra_weight_diff=None, intra_weight_diff_volume=None, intra_quota_diff_volume=None, intra_weight_quota=None,
        fee_ratio=0.00015, tax_ratio=STAMP_RATE, add_exec=False, df_position=None, df_trades=None, df_quota_pre=None,
        bar_mode=8, rename_bar_dict=None, ps_mpt_rps=False, qt_ps_ret_mv_real=False, pos_pre_mode=True, return_bps_details=False, backtest_date=None, smooth_fee=False):
    print(product)
    """
    return paras:
        'product':                  product,
        'alpha-ret':                alpha_vector * 10000,
        'trades-simu-ret':          simulate_ret * 10000,
        'trades-ret':               market_vector * 10000,
        'trades-real-ret':          realized_pnl * 10000,
        'trades-unreal-ret':        unrealized_pnl * 10000,
        'trading-ratio-long':       df_trades_buy,
        'trading-ratio-short':      df_trades_sell,
        'trading-ratio-net':        df_trades_diff,
        'weight-diff-mat-long':     mat_weight_diff_long,
        'weight-diff-mat-short':    mat_weight_diff_short,
        'weight-diff-mat':          weight_diff_tuple[0],
    """
    df_price = df_price.copy(deep=True)
    df_1min_mat_close = df_1min_mat_close.copy(deep=True)
    df_30min_mat_vwap = df_30min_mat_vwap.copy(deep=True)
    intra_weight = intra_weight.copy(deep=True)
    intra_weight_diff = intra_weight_diff.copy(deep=True)
    intra_weight_diff_volume = intra_weight_diff_volume.copy(deep=True)
    if intra_weight_quota is not None: intra_weight_quota = intra_weight_quota.copy(deep=True)

    start_time = time.time()
    
    predate = get_predate(curdate, 1)
    gbd = GetBaseData()

    if (df_position is None) and pos_pre_mode: df_position = gbd.get_position_close(predate, product, backtest_date=backtest_date)
    elif (df_position is None) and (not pos_pre_mode): df_position = gbd.get_position_close(curdate, product, backtest_date=backtest_date, pre_mode=True)
    else: df_position = df_position.copy(deep=True)

    if df_trades is None: df_trades = gbd.get_trades_data(curdate, product, backtest_date=backtest_date)
    else: df_trades = df_trades.copy(deep=True)

    if df_quota_pre is None: df_quota_pre = gbd.get_quota_close(predate, product, bar_mode)
    if ps_mpt_rps and df_position.empty: df_position = df_quota_pre

    if df_position.empty or df_trades.empty: return None

    cal_time = time.time()
    ipo_code_list = get_ipo_code_list(curdate, curdate).get(curdate, [])
    if ipo_code_list: df_trades = df_trades[~df_trades['SecuCode'].isin(ipo_code_list)]

    file_path_ex_right = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/ex_right_data/{curdate}_ex_right_data.csv'
    if os.path.exists(file_path_ex_right) and pos_pre_mode:
        df_exit_right = pd.read_csv(file_path_ex_right)
        if not df_exit_right.empty:
            df_exit_right['SecuCode'] = df_exit_right['SecuCode'].apply(lambda x: expand_stockcode(x))
            df_quota_pre = pd.merge(df_quota_pre, df_exit_right, on='SecuCode', how='left')
            df_quota_pre['PreCloseVolume'] *= df_quota_pre['ExRightRatio'].fillna(1)

            df_position = pd.merge(df_position, df_exit_right, on='SecuCode', how='left')
            df_position['PreCloseVolume'] *= df_position['ExRightRatio'].fillna(1)

    df_quota_pre = df_quota_pre[['SecuCode', 'PreCloseVolume']]
    df_position = df_position[['SecuCode', 'PreCloseVolume']]

    intra_return = calculate_intradaily_trading_process_analysis_result(
        time_list, index_1minret,
        df_position, df_quota_pre, df_trades, df_price, df_1min_mat_close,
        intra_weight, intra_weight_diff, intra_weight_quota,
        fee_ratio=fee_ratio, tax_ratio=tax_ratio, add_exec=add_exec, return_mode='df', bar_mode=bar_mode, rename_bar_dict=rename_bar_dict, qt_ps_ret_mv_real=qt_ps_ret_mv_real)
    if add_exec:
        pre_mv_sum, alpha_vector, pos_alpha_ret, quota_alpha_ret, pos_diff_code_ret, market_vector, unrealized_pnl, realized_pnl, \
        df_trades_buy, df_trades_sell, df_trades_diff, \
        weight_diff_tuple, mat_weight_diff_long, mat_weight_diff_short, real_weight_mat, mat_weight_diff_exec = intra_return
    else:
        pre_mv_sum, alpha_vector, pos_alpha_ret, quota_alpha_ret, pos_diff_code_ret, market_vector, unrealized_pnl, realized_pnl, \
        df_trades_buy, df_trades_sell, df_trades_diff, \
        weight_diff_tuple, mat_weight_diff_long, mat_weight_diff_short, real_weight_mat = intra_return

    print('市值:', pre_mv_sum)
    if intra_weight_diff_volume is not None:
        simulate_weight_ret, weight_trades_ratio_short, weight_trades_ratio_long, mat_turnover = calculate_intradaily_vwap_simu_analysis_result(
            intra_weight_diff_volume, df_30min_mat_vwap, df_1min_mat_close, time_list,
            init_mv=1e8, fee_ratio=fee_ratio, tax_ratio=tax_ratio, return_mode='df', bar_mode=bar_mode, rename_bar_dict=rename_bar_dict, smooth_fee=smooth_fee)
    if intra_quota_diff_volume is not None:
        simulate_ret, quota_trades_ratio_short, quota_trades_ratio_long, quota_turnover = calculate_intradaily_vwap_simu_analysis_result(
            intra_quota_diff_volume, df_30min_mat_vwap, df_1min_mat_close, time_list,
            init_mv=pre_mv_sum, fee_ratio=fee_ratio, tax_ratio=tax_ratio, return_mode='df', bar_mode=bar_mode, rename_bar_dict=rename_bar_dict, smooth_fee=smooth_fee)

    pos_diff_quota_ret = pos_alpha_ret - quota_alpha_ret
    market_vector *= 10000
    pos_diff_code_ret *= 10000
    print('用时:', time.time() - start_time, time.time() - cal_time)
    res_array_dict = {
        'product': product,
        'alpha-ret': alpha_vector * 10000,
        'pos-ret': pos_alpha_ret * 10000,
        'quota-ret': quota_alpha_ret * 10000,
        'pos-diff-ret': pos_diff_quota_ret * 10000,
        'trades-ret': np.array(market_vector.sum(axis=0)),

        'trades-real-ret': realized_pnl * 10000,
        'trades-unreal-ret': unrealized_pnl * 10000,

        'trading-ratio-long': df_trades_buy,
        'trading-ratio-short': df_trades_sell,
        'trading-ratio-net': df_trades_diff,

        'mat-long-weight-diff': mat_weight_diff_long,
        'mat-short-weight-diff': mat_weight_diff_short,
        'mat-weight-diff': weight_diff_tuple[0],
    }
    res_df_dict = {
        'trades-ret': market_vector,
        'pos-diff-ret': pos_diff_code_ret
    }
    conlist = [
        market_vector[[time_list[-1]]].rename({time_list[-1]: 'trades-ret'}, axis='columns')
    ]
    if intra_quota_diff_volume is not None:
        union_index = simulate_ret.index.union(market_vector.index)
        simulate_ret = simulate_ret.reindex(union_index) * 10000
        market_vector = market_vector.reindex(union_index)
        real_diff_vwap_simu = market_vector.fillna(0) - simulate_ret.fillna(0)

        conlist.append(simulate_ret[[time_list[-1]]].rename({time_list[-1]: 'trades-simu-ret'}, axis='columns'))
        conlist.append(real_diff_vwap_simu[[time_list[-1]]].rename({time_list[-1]: 'trades-diff-ret'}, axis='columns'))

        res_array_dict['trading-ratio-quota-long'] = quota_trades_ratio_long
        res_array_dict['trading-ratio-quota-short'] = quota_trades_ratio_short
        res_array_dict['trading-ratio-quota-net'] = quota_trades_ratio_short + quota_trades_ratio_long
        res_array_dict['trades-simu-ret'] = np.array(simulate_ret.sum(axis=0))
        res_array_dict['trades-diff-ret'] = np.array(real_diff_vwap_simu.sum(axis=0))

        res_df_dict['trades-simu-ret'] = simulate_ret
        res_df_dict['trades-diff-ret'] = real_diff_vwap_simu

    if intra_weight_diff_volume is not None:
        union_index = simulate_weight_ret.index.union(market_vector.index)
        simulate_weight_ret = simulate_weight_ret.reindex(union_index) * 10000
        market_vector = market_vector.reindex(union_index)
        real_diff_vwap_simu_weight = market_vector.fillna(0) - simulate_weight_ret.fillna(0)

        conlist.append(simulate_weight_ret[[time_list[-1]]].rename({time_list[-1]: 'trades-simu-mat-ret'}, axis='columns'))
        conlist.append(real_diff_vwap_simu_weight[[time_list[-1]]].rename({time_list[-1]: 'trades-diff-mat-ret'}, axis='columns'))

        res_array_dict['weight-trading-ratio-long'] = weight_trades_ratio_long
        res_array_dict['weight-trading-ratio-short'] = weight_trades_ratio_short
        res_array_dict['weight-trading-ratio-net'] = weight_trades_ratio_short + weight_trades_ratio_long
        res_array_dict['trades-simu-mat-ret'] = np.array(simulate_weight_ret.sum(axis=0))
        res_array_dict['trades-diff-mat-ret'] = np.array(real_diff_vwap_simu_weight.sum(axis=0))

        res_df_dict['trades-simu-mat-ret'] = simulate_weight_ret
        res_df_dict['trades-diff-mat-ret'] = real_diff_vwap_simu_weight

    conlist.append(pos_diff_code_ret[[time_list[-1]]].rename({time_list[-1]: 'pos-diff-ret', '150000': 'pos-diff-ret'}, axis='columns'))
    df_bps = pd.concat(conlist, axis=1)
    df_bps = df_bps[np.abs(df_bps).sum(axis=1) != 0]
    
    df_bps_code_sum = np.cumsum(df_bps.fillna(0), axis=0)
    df_bps = df_bps.reset_index().rename({'index': 'SecuCode'}, axis='columns')
    df_bps_price_cum = pd.merge(df_bps, df_price.reset_index()[['SecuCode', 'ClosePrice']], on='SecuCode', how='left').drop('SecuCode', axis=1).sort_values('ClosePrice', ascending=True)
    df_bps_price_cum['ClosePrice'] = df_bps_price_cum['ClosePrice'].astype('str')
    df_bps_price_cum = np.cumsum(df_bps_price_cum.set_index('ClosePrice').fillna(0), axis=0)

    if return_bps_details:
        df_trades['Value'] = df_trades['Volume'] * df_trades['Price']
        mode = '5min' if bar_mode == 48 else '30min'
        df_trades['Bar'] = df_trades['time'].apply(lambda x: time_2_bar_n_min(x, mode=mode))
        df_trade_vol_bar = df_trades.groupby(['SecuCode', 'LongShort', 'Bar'])['Volume'].sum().reset_index()
        df_trade_vol_long = pd.pivot_table(df_trade_vol_bar[df_trade_vol_bar['LongShort'] == 0], index='SecuCode', columns='Bar', values='Volume').reindex(columns=list(range(1, bar_mode + 1)), fill_value=0)
        df_trade_vol_short = pd.pivot_table(df_trade_vol_bar[df_trade_vol_bar['LongShort'] == 1], index='SecuCode', columns='Bar', values='Volume').reindex(columns=list(range(1, bar_mode + 1)), fill_value=0)
        df_trade_vol_short = df_trade_vol_short.rename({bar: f'{bar}_short' for bar in range(1, bar_mode + 1)}, axis='columns')
        df_turnover = df_trades.groupby('SecuCode')['Value'].sum().to_frame().rename({'Value': 'Turnover'}, axis='columns')

        df_bps = pd.concat([
            df_quota_pre.set_index('SecuCode').rename({'PreCloseVolume': 'PreCloseQuota'}, axis='columns'),
            df_position.set_index('SecuCode'),
            intra_quota_diff_volume,
            df_trade_vol_long,
            df_trade_vol_short,
            df_turnover,
            quota_turnover,
            df_bps.set_index('SecuCode')
        ], axis=1).fillna(0)
        df_bps = df_bps[np.abs(df_bps).sum(axis=1) != 0].sort_index().reset_index().rename({'index': 'SecuCode'}, axis='columns')
        df_bps['HoldMV'] = pre_mv_sum

    if intra_weight_quota is not None: res_array_dict['quota-weight-diff'] = weight_diff_tuple[1]
    if add_exec: res_array_dict['mat-exec-weight-diff'] = mat_weight_diff_exec

    res_dict_2 = {
        'real_weight_mat': real_weight_mat
    }

    return res_array_dict, res_dict_2, res_df_dict, df_bps, df_bps_code_sum, df_bps_price_cum


def intra_analysis_plot_market_product(
        curdate, product_infor_list, compare_infor_list=None, mean_compare_date_tuple=None, mode='dest-pos',
        target_paras_list=None, add_exec=False, figure_all_in_one=False, ret_dict=False, mode_nmin=None, bar_mode=8, bar_mode_pre=None,
        ps_mpt_rps=False, qt_ps_ret_mv_real=False, pos_pre_mode=True, drop_bar_delay=True, return_bps_details=True, scale_pqt_bar_mode=False,
        backtest_date=None, output_dir=None, smooth_fee=False, font_size=10):
    if bar_mode_pre is None: bar_mode_pre = bar_mode
    predate = get_predate(curdate, 1)
    df_price, df_1min_mat_close, df_30min_mat_vwap, df_30min_mat_close, df_lastprice, time_list = get_intradaily_analysis_all_price_data(curdate, bar_mode=bar_mode, mode_nmin=mode_nmin)

    if isinstance(product_infor_list, str): product_infor_list = [product_infor_list]
    df_accsumm = get_production_list_trading(curdate, ret_df_data=True, drop_simu=False)
    df_accsumm_pre = get_production_list_trading(predate, ret_df_data=True, drop_simu=False)
    dict_bar_mode = {product: bar_mode for product, bar_mode in zip(df_accsumm['Account'], df_accsumm['bar'])}
    dict_bar_mode_pre = {product: bar_mode for product, bar_mode in zip(df_accsumm_pre['Account'], df_accsumm_pre['bar'])}

    index_ret_dict = get_indexret_1min_dict(curdate)
    result_dict = {}
    for prod in product_infor_list:
        if SimuDict.get(prod) is not None:
            quota_prd = SimuDict[prod].get('quota', prod)

            mkt_prod = SimuDict[prod].get('mkt', prod)
            fee_ratio = production_2_feeratio(mkt_prod)
            proportion = SimuDict[prod].get('proportion', production_2_proportion(mkt_prod))

            alpha_name = SimuDict[prod].get('alpha', get_production_list_trading(curdate, quota_prd, 'alpha', drop_simu=False))
            pre_alpha_name = alpha_name
            quota_path = SimuDict[prod].get('quota_path')
        else:
            quota_prd = prod
            if 20250110 >= int(curdate) > 20241111:
                if prod == 'JQ11': proportion = {'HS300': 0.1, 'ZZ500': 0.1, 'ZZ1000': 0.8}
                elif prod == 'JQ11B': proportion = {'ZZ1000': 1}
                else: proportion = production_2_proportion_date(curdate, prod)
            else: proportion = production_2_proportion_date(curdate, prod)
            fee_ratio = production_2_feeratio(prod)
            pre_alpha_name, alpha_name, quota_path = None, None, None

        if prod in ProductionList_AlphaShort: proportion = {'ZZ500': 0}

        bar_mode, bar_mode_pre = dict_bar_mode.get(prod, bar_mode), dict_bar_mode_pre.get(prod, bar_mode_pre)
        if bar_mode == 8: rename_bar_dict = DICT_TIME_2_BAR_30MIN
        else: rename_bar_dict = DICT_TIME_2_BAR_5MIN

        print(prod, bar_mode, bar_mode_pre)
        intra_weight, intra_weight_pre, intra_weight_diff, intra_weight_diff_volume, intra_quota_diff_volume, intra_quota, intra_weight_quota = (
            get_intradaily_analysis_all_mat_data(
                curdate, predate, quota_prd, df_price, df_lastprice, df_30min_mat_close, mode=mode,
                quota_path=quota_path, pre_alpha_name=pre_alpha_name, alpha_name=alpha_name, bar_mode=bar_mode, bar_mode_pre=bar_mode_pre,
                drop_bar_delay=drop_bar_delay, scale_pqt_bar_mode=scale_pqt_bar_mode))

        index_1minret = np.sum(np.array([prop * index_ret_dict[idx_nm] for idx_nm, prop in proportion.items()]), axis=0)
        res = calculate_intradaily_analysis_result(
            curdate, prod, index_1minret, df_price, df_1min_mat_close, df_30min_mat_vwap, time_list,
            intra_weight=intra_weight, intra_weight_diff=intra_weight_diff, intra_weight_diff_volume=intra_weight_diff_volume, intra_quota_diff_volume=intra_quota_diff_volume, intra_weight_quota=intra_weight_quota,
            fee_ratio=fee_ratio, add_exec=add_exec, bar_mode=bar_mode, rename_bar_dict=rename_bar_dict,
            ps_mpt_rps=ps_mpt_rps, qt_ps_ret_mv_real=qt_ps_ret_mv_real, pos_pre_mode=pos_pre_mode, return_bps_details=return_bps_details, backtest_date=backtest_date, smooth_fee=smooth_fee)

        result_dict[prod] = res

    if ret_dict: return result_dict, time_list

    if compare_infor_list is not None:
        if output_dir is None: output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{curdate}/Simulation_Compare/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        simu_compare_img_list = []
        for left_prod, right_prod in compare_infor_list:
            if result_dict[left_prod] is None or result_dict[right_prod] is None: continue

            file_path = output_dir + f'{curdate}_{left_prod}_{right_prod}_intradaily_analysis.png'
            plt.rcParams['font.size'] = font_size
            dict_left, dict_right = intra_analysis_compare_dict_process(result_dict[left_prod], result_dict[right_prod])
            plot_subplots_compare_dict_data(curdate, time_list, dict_left, dict_right, target_paras_list, file_path)
            simu_compare_img_list.append(file_path)

            if mean_compare_date_tuple is not None:
                start_date, end_date = mean_compare_date_tuple
                if isinstance(start_date, int): start_date = get_predate(end_date, start_date)
                date_list = get_trading_days(start_date, end_date)

                conlist_left, conlist_right = [], []
                for date in date_list:
                    conlist_left.append(pd.read_csv(f'{output_dir.replace(curdate, date)}{date}_{left_prod}_quota_mode.csv'))
                    conlist_right.append(pd.read_csv(f'{output_dir.replace(curdate, date)}{date}_{right_prod}_quota_mode.csv'))

                dict_left = pd.concat(conlist_left, axis=0).groupby(['time', 'product']).mean().to_dict(orient='list')
                dict_left['product'] = left_prod
                dict_right = pd.concat(conlist_right, axis=0).groupby(['time', 'product']).mean().to_dict(orient='list')
                dict_right['product'] = right_prod

                file_path = output_dir + f'{start_date}_{end_date}_{left_prod}_{right_prod}_mean_intradaily_analysis.png'
                plot_subplots_compare_dict_data(
                    curdate, time_list, dict_left, dict_right, target_paras_list, file_path,
                    title_str=f'{start_date}_{end_date} | mean: ')
                simu_compare_img_list.append(file_path)
        return simu_compare_img_list

    img_list = []
    if not figure_all_in_one:
        if output_dir is None: output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{curdate}/intra_analysis_data/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        for prod in product_infor_list:
            drop_bar_list = get_intradaily_delay_drop_bar_list(curdate, prod, bar_mode)

            plt.rcParams['font.size'] = font_size
            alpha_name = get_production_list_trading(curdate, prod, drop_simu=False)
            if prod == 'ZZ10005Msimu': alpha_name = 'zli.weightv9.11mpo_intrav1.0_zz1000_r7.1_10b.3k.5m_2023'
            gbd = GetBaseData()
            dict_alpha_simu = gbd.read_alpha_simu(curdate, alpha_name)
            if SimuDict.get(prod) is not None:
                mkt_prod = SimuDict[prod].get('mkt', prod)
                fee_ratio = production_2_feeratio(mkt_prod)
            else:
                fee_ratio = production_2_feeratio(prod)

            res = result_dict[prod]
            if res is None: continue
            res_dict = intra_analysis_dict_process(res)
            kind_dict = {'bps-pos-scl': 'bar', 'bps-trades-scl': 'bar'}
            code_type_xticks = ['bps-pos-scl', 'bps-trades-scl', 'bps-trades-sort-code-ret', 'bps-diff-sort-code-ret']

            mode_nmin_str = mode_nmin if isinstance(mode_nmin, str) else '_actually'
            file_path = output_dir + f'{curdate}_{prod}_{mode_nmin_str}.png'

            intra_analysis_plot(curdate, time_list, res_dict, target_paras_list, kind_dict=kind_dict, code_type_xticks=code_type_xticks, fpath=file_path,
                                title_str=f'{mode},delay_mode={mode_nmin_str},drop_bar={drop_bar_list},pos_pre_mode={pos_pre_mode},qt_ps_ret_mv_real={qt_ps_ret_mv_real},'
                                          f'scale_pqt_bar_mode={scale_pqt_bar_mode}\n{dict_alpha_simu} | fr={fee_ratio} | {alpha_name}')
            df_bps = result_dict[prod][3]
            outputdir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{curdate}/vwap_simu_temp/'
            if not os.path.exists(outputdir): os.makedirs(outputdir)
            df_bps.to_csv(f'{outputdir}{curdate}_{prod}_vwapsimu.csv', index=False)

            img_list.append(file_path)
    else:
        if output_dir is None: output_dir = f'{PLATFORM_PATH_DICT["v_path"]}StockData/d0_file/IntraAnalysisResults/{curdate}/intra_analysis_data/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        file_path = output_dir + f'{curdate}_production_intradaily_analysis.png'
        if target_paras_list is None: target_paras_list = list(set(list(result_dict.values())[-1][0].keys()) - set('product'))
        print(target_paras_list)

        paras_df_dict = {}
        for paras_name in target_paras_list:
            infor_dict = {product: result_dict[product][0][paras_name] for product in product_infor_list}

            df_paras = pd.DataFrame(infor_dict, index=time_list)
            df_paras.index = df_paras.index.astype('str')
            paras_df_dict[paras_name] = df_paras

        plot_subplots_compare_dict_data(
            curdate, time_list, paras_df_dict, None, target_paras_list, file_path, multi_prods=True, one_figure=False)
        img_list.append(file_path)
    return img_list


def intra_analysis_plot(curdate, time_list, dict_data, target_paras_list=None, kind_dict=None, code_type_xticks=None, fpath=None, title_str='', num_locator=30):
    dict_data = deepcopy(dict_data)

    time_list = [str(x) for x in time_list]
    if target_paras_list is None: target_paras_list = sorted(list(set(dict_data.keys()) - {'product'}))
    print(target_paras_list)

    subx, suby = PLOT_STRUCT_DICT[len(target_paras_list)]
    plt.figure(figsize=(20, 16))
    for iplot, key in enumerate(target_paras_list):
        if key not in target_paras_list: continue
        ax = plt.subplot(subx, suby, iplot + 1)

        if isinstance(dict_data[key], np.ndarray): df = pd.DataFrame({key: dict_data[key]}, time_list)
        elif isinstance(dict_data[key], dict): df = pd.DataFrame(dict_data[key], time_list)
        else: df = dict_data[key]

        if isinstance(df, list):
            df_left, df_right = df
        else:
            df_left = df

        df_left = rename_df_columns_name(df_left, mode='last', precision=3)
        df_left.index.name = None
        df_left.columns.name = None
        df_left.index = df_left.index.astype('str')

        if (kind_dict is None) or (kind_dict.get(key) is None): bar1 = df_left.plot(ax=ax)
        else: bar1 = df_left.plot(ax=ax, kind=kind_dict[key])
        plt.legend(framealpha=0.5)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        
        if len(df_left) > 500: ax.xaxis.set_major_locator(MultipleLocator(len(df_left) // 8))
        else: ax.xaxis.set_major_locator(MultipleLocator(num_locator))
        if key in code_type_xticks:
            code_list = df_left.index.to_list()
            max_num, min_num = np.max(df_left) * 1.2, np.min(df_left) * 1.2
            for c_i, (c1, c2) in enumerate(zip(code_list, code_list[1:])):
                c_i += 1
                if c1.startswith('0') and c2.startswith('3'):
                    plt.plot([c_i, c_i], [min_num, max_num], 'k--')
                if c1.startswith('3') and c2.startswith('6'):
                    plt.plot([c_i, c_i], [min_num, max_num], 'k--')
                if c1.startswith('6') and (not c1.startswith('688')) and c2.startswith('688'):
                    plt.plot([c_i, c_i], [min_num, max_num], 'k--')

        if isinstance(df, list):
            ax_right = ax.twinx()
            df_right = rename_df_columns_name(df_right, mode='last', precision=3)
            df_right.index.name = None
            df_right.columns.name = None
            df_right.index = df_right.index.astype('str')

            if (kind_dict is None) or (kind_dict.get(key) is None): bar2 = df_right.plot(ax=ax_right, linestyle='--')
            else: bar2 = df_right.plot(ax=ax_right, kind=kind_dict[key], linestyle='--')

            ax_right.spines['top'].set_color('none')
            ax_right.spines['left'].set_color('none')
            ax_right.yaxis.set_ticks_position('right')
            ax_right.spines['right'].set_position(('data', len(df_right - 1)))
            if len(df_right) > 500: ax_right.xaxis.set_major_locator(MultipleLocator(len(df_right) // 8))
            else: ax_right.xaxis.set_major_locator(MultipleLocator(num_locator))
            
            handles_left, labels_left = bar1.get_legend_handles_labels()
            handles_right, labels_right = bar2.get_legend_handles_labels()
            handles_left += handles_right
            labels_left += labels_right
            by_label = dict(zip(labels_left, handles_left))
            plt.legend(list(by_label.values()), list(by_label.keys()), loc='upper left', bbox_to_anchor=(0.05, 1.2), ncol=1, fontsize=9)

        plt.legend(framealpha=0.5)
        plt.grid(True, linestyle='-.')
        plt.xticks(rotation=0)
        plt.title(key)

    plt.suptitle(f'[{curdate}]-{title_str}: {dict_data["product"]}')

    plt.tight_layout()
    if fpath is not None: plt.savefig(fpath)


def intra_analysis_dict_process(res, thres=500):
    result_dict, res_dict_2, res_df, df_bps, df_bps_code_sum, df_bps_price_cum = res
    df_bps = df_bps.set_index('SecuCode')
    df_bps['bps-trades-scl'] = df_bps['trades-diff-ret'] * df_bps['HoldMV'] / df_bps['Turnover']
    df_bps['bps-pos-scl'] = df_bps['pos-diff-ret'] * df_bps['HoldMV'] / df_bps['Turnover']
    df_bps['bps-trades-scl'] = np.maximum(np.minimum(df_bps['bps-trades-scl'], thres), -thres)
    df_bps['bps-pos-scl'] = np.maximum(np.minimum(df_bps['bps-pos-scl'], thres), -thres)

    df_bps_code_sum['total-diff-ret'] = df_bps_code_sum['trades-diff-ret'] + df_bps_code_sum['pos-diff-ret']

    result_dict['total-diff-ret'] = result_dict['trades-diff-ret'] + result_dict['pos-diff-ret']

    res_dict = {}
    res_dict['product'] = result_dict['product']
    res_dict['alpha-ret'] = {
        'alpha-ret': result_dict['alpha-ret'],
    }

    res_dict['bps-pos-ret'] = {
        'pos-ret': result_dict['pos-ret'],
        'quota-ret': result_dict['quota-ret'],
    }
    res_dict['bps-trades-ret'] = {
        'trades-ret': result_dict['trades-ret'],
        'trades-simu-ret': result_dict['trades-simu-ret']
    }
    res_dict['bps-trades-sort-code-ret'] = df_bps_code_sum[['trades-ret', 'trades-simu-ret']]
    res_dict['bps-trades-sort-price-ret'] = df_bps_price_cum[['trades-ret', 'trades-simu-ret']]

    res_dict['bps-diff-ret'] = {
        'trades-diff-ret': result_dict['trades-diff-ret'],
        'pos-diff-ret': result_dict['pos-diff-ret'],
        'total-diff-ret': result_dict['total-diff-ret'],
    }
    res_dict['bps-ascending-diff-ret'] = pd.DataFrame({
        'trades-diff-ret': np.cumsum(np.sort(df_bps['trades-diff-ret'].to_list())),
    })
    res_dict['bps-diff-sort-code-ret'] = df_bps_code_sum[['trades-diff-ret', 'pos-diff-ret', 'total-diff-ret']]
    res_dict['bps-diff-sort-price-ret'] = df_bps_price_cum[['trades-diff-ret', 'pos-diff-ret']]

    res_dict['bps-pos-scl'] = df_bps[['bps-pos-scl']]
    res_dict['bps-trades-scl'] = df_bps[['bps-trades-scl']]

    # if (result_dict.get('trades-ret') is not None) and (
    #         result_dict.get('trades-real-ret') is not None) and (
    #         result_dict.get('trades-unreal-ret') is not None):
    #     res_dict['trades-ret'] = {
    #         'trades-ret': result_dict['trades-ret'],
    #         'trades-real-ret': result_dict['trades-real-ret'],
    #         'trades-unreal-ret': result_dict['trades-unreal-ret'],
    #     }

    trading_ratio_dict = {}
    if (result_dict.get('trading-ratio-quota-long') is not None) and (
            result_dict.get('trading-ratio-quota-short') is not None):
        trading_ratio_dict.update({
            'trading-ratio-quota-long': result_dict['trading-ratio-quota-long'],
            'trading-ratio-quota-short': result_dict['trading-ratio-quota-short']
        })

    # if (result_dict.get('weight-trading-ratio-long') is not None) and (
    #         result_dict.get('weight-trading-ratio-short') is not None):
    #     trading_ratio_dict.update({
    #         'weight-trading-ratio-long': result_dict['weight-trading-ratio-long'],
    #         'weight-trading-ratio-short': result_dict['weight-trading-ratio-short']
    #     })

    if (result_dict.get('trading-ratio-long') is not None) and (
            result_dict.get('trading-ratio-short') is not None):
        trading_ratio_dict.update({
            'trading-ratio-long': result_dict['trading-ratio-long'],
            'trading-ratio-short': result_dict['trading-ratio-short'],
        })
    res_dict['trading-ratio'] = trading_ratio_dict

    res_dict['trading-ratio-net'] = {
        'real-trading-ratio-net': result_dict['trading-ratio-net'],
        'trading-ratio-quota-net': result_dict['trading-ratio-quota-net'],
        # 'weight-trading-ratio-net': result_dict['weight-trading-ratio-net']
    }

    if (result_dict.get('mat-long-weight-diff') is not None) and (
            result_dict.get('mat-short-weight-diff') is not None) and (
            result_dict.get('mat-weight-diff') is not None) and (
            result_dict.get('quota-weight-diff') is not None):
        res_dict['weight-diff-mat'] = {
            'mat-weight-diff': result_dict['mat-weight-diff'],
            'mat-long-weight-diff': result_dict['mat-long-weight-diff'],
            'mat-short-weight-diff': result_dict['mat-short-weight-diff'],
        }

    res_dict['weight-diff-quota'] = result_dict['quota-weight-diff']

    return res_dict


def intra_analysis_compare_dict_process(left_dict, right_dict):
    dict_left, res_dict_2_left, res_df_left = left_dict[:3]
    dict_right, res_dict_2_right, res_df_right = right_dict[:3]

    left_real_weight = res_dict_2_left['real_weight_mat']
    right_real_weight = res_dict_2_right['real_weight_mat']
    union_index = left_real_weight.index.union(right_real_weight.index)
    left_real_weight = left_real_weight.reindex(union_index)
    right_real_weight = right_real_weight.reindex(union_index)
    weight_diff_compare = np.array(np.abs(left_real_weight - right_real_weight).sum(axis=0) / 2)
    print(weight_diff_compare)
    dict_left_res, dict_right_res = {}, {}

    dict_left_res['product'] = dict_left['product']
    dict_left_res['alpha-ret'] = dict_left['alpha-ret']
    dict_left_res['pos-ret'] = dict_left['pos-ret']
    dict_left_res['quota-ret'] = dict_left['quota-ret']
    dict_left_res['pos-diff-ret'] = dict_left['pos-diff-ret']
    dict_left_res['pos-ret'] = dict_left['pos-ret']
    dict_left_res['quota-ret'] = dict_left['quota-ret']

    dict_left_res['trades-ret'] = dict_left['trades-ret']
    dict_left_res['trades-simu-ret'] = dict_left['trades-simu-ret']
    dict_left_res['trades-diff-ret'] = dict_left['trades-diff-ret']
    dict_left_res['trades-simu-mat-ret'] = dict_left['trades-simu-mat-ret']
    dict_left_res['trades-diff-mat-ret'] = dict_left['trades-diff-mat-ret']

    dict_left_res['weight-diff-quota'] = dict_left['quota-weight-diff']
    dict_left_res['weight-diff-mat'] = dict_left['mat-weight-diff']
    dict_left_res['weight-diff-compare'] = weight_diff_compare

    dict_left_res['trading-ratio-net'] = dict_left['trading-ratio-net']
    dict_left_res['trading-ratio-long'] = dict_left['trading-ratio-long']
    dict_left_res['trading-ratio-short'] = dict_left['trading-ratio-short']
    dict_left_res['trading-ratio-quota-net'] = dict_left['trading-ratio-quota-net']
    dict_left_res['trading-ratio-quota-long'] = dict_left['trading-ratio-quota-long']
    dict_left_res['trading-ratio-quota-short'] = dict_left['trading-ratio-quota-short']

    dict_right_res['product'] = dict_right['product']
    dict_right_res['alpha-ret'] = dict_right['alpha-ret']
    dict_right_res['pos-ret'] = dict_right['pos-ret']
    dict_right_res['quota-ret'] = dict_right['quota-ret']
    dict_right_res['pos-diff-ret'] = dict_right['pos-diff-ret']
    dict_right_res['pos-ret'] = dict_right['pos-ret']
    dict_right_res['quota-ret'] = dict_right['quota-ret']

    dict_right_res['trades-ret'] = dict_right['trades-ret']
    dict_right_res['trades-simu-ret'] = dict_right['trades-simu-ret']
    dict_right_res['trades-diff-ret'] = dict_right['trades-diff-ret']
    dict_right_res['trades-simu-mat-ret'] = dict_right['trades-simu-mat-ret']
    dict_right_res['trades-diff-mat-ret'] = dict_right['trades-diff-mat-ret']

    dict_right_res['weight-diff-quota'] = dict_right['quota-weight-diff']
    dict_right_res['weight-diff-mat'] = dict_right['mat-weight-diff']

    dict_right_res['trading-ratio-net'] = dict_right['trading-ratio-net']
    dict_right_res['trading-ratio-quota-net'] = dict_right['trading-ratio-quota-net']
    dict_right_res['trading-ratio-long'] = dict_right['trading-ratio-long']
    dict_right_res['trading-ratio-short'] = dict_right['trading-ratio-short']
    dict_right_res['trading-ratio-quota-long'] = dict_right['trading-ratio-quota-long']
    dict_right_res['trading-ratio-quota-short'] = dict_right['trading-ratio-quota-short']

    return dict_left_res, dict_right_res


def calculate_daily_alpha_simu_return(mat_weight_diff, df_vwap_close, bar_mode=8, fee_ratio=0.00015, exch_mode=False):
    mat_weight_diff = mat_weight_diff.copy(deep=True)
    df_vwap_close = df_vwap_close.copy(deep=True)
    union_index = mat_weight_diff.index.union(df_vwap_close.index)

    mat_weight_diff = mat_weight_diff.reindex(union_index, columns=[f'Ti{i}_Weight_Diff' for i in range(1, bar_mode + 1)], fill_value=np.nan)
    df_vwap_close = df_vwap_close.reindex(union_index, fill_value=np.nan)

    vwap_price = np.array(df_vwap_close[[i for i in range(1, bar_mode + 1)]])
    is_normal = np.array(mat_weight_diff > 0)

    position_price = vwap_price * (is_normal * (1 + fee_ratio) + (1 - is_normal) * (1 - STAMP_RATE - fee_ratio))
    close_price_mat = np.tile(np.array(df_vwap_close[['ClosePrice']]), bar_mode)

    df_alpha_simu = pd.DataFrame(np.array(mat_weight_diff) * (close_price_mat - position_price) / vwap_price).replace([-np.inf, np.inf], 0)
    alpha_simu = df_alpha_simu.sum().sum()
    if exch_mode:
        flag_sh = np.tile(np.array(pd.Series(union_index).str.startswith('6').to_frame()), bar_mode)
        alpha_simu_sh = (df_alpha_simu * flag_sh).sum().sum()
        alpha_simu_sz = (df_alpha_simu * (~flag_sh)).sum().sum()
        return alpha_simu, alpha_simu_sz, alpha_simu_sh
    return alpha_simu


def calculate_daily_position_alpha_return_by_weight_mat(
        weight_mat, df_close, index_return, tax_ratio=STAMP_RATE, fee_ratio=0.00015, mode='array', bar_mode=8):
    df_close = deepcopy(df_close)
    index_return = deepcopy(index_return)
    if mode == 'dict':
        alphasimu_dict = {}
        date_list = list(weight_mat.keys())
        for predate, date in zip(date_list, date_list[1:]):
            print(date)
            alphasimu_dict[date] = calculate_daily_position_alpha_return_by_weight_mat(
                weight_mat[predate], df_close[date], index_return[date],
                tax_ratio=tax_ratio, fee_ratio=fee_ratio, mode='array', bar_mode=bar_mode)
        return alphasimu_dict
    else:
        union_index = weight_mat.index.union(df_close.index)
        weight_diff_mat = weight_mat.reindex(union_index, fill_value=0)
        df_vwap_close = df_close.reindex(union_index, fill_value=0)
        df_vwap_close['PriceChange'] = (df_vwap_close['ClosePrice'] - df_vwap_close['PreClosePrice']
                                        ) / df_vwap_close['PreClosePrice']
        alpha_ret = (weight_diff_mat[bar_mode] * df_vwap_close['PriceChange']).sum() - index_return

        return alpha_ret


def calculate_daily_position_alpha_return_by_quota_mat(
        quota_mat, df_close, index_return, dict_holdmv, mode='array', bar_mode=8, pre_mode=True, alpha_mode=True):
    df_close = deepcopy(df_close)
    index_return = deepcopy(index_return)
    if mode == 'dict':
        alphasimu_dict = {}
        date_list = list(quota_mat.keys())
        if pre_mode:
            for predate, date in zip(date_list, date_list[1:]):
                print(date)
                alphasimu_dict[date] = calculate_daily_position_alpha_return_by_quota_mat(
                    quota_mat[predate], df_close[date], index_return[date], dict_holdmv[date], mode='array', bar_mode=bar_mode, alpha_mode=alpha_mode)
        else:
            for date in date_list:
                print(date)
                alphasimu_dict[date] = calculate_daily_position_alpha_return_by_quota_mat(
                    quota_mat[date], df_close[date], index_return[date], dict_holdmv[date], mode='array', bar_mode=bar_mode, alpha_mode=alpha_mode)
        return alphasimu_dict
    else:
        if quota_mat.empty:
            return np.nan

        union_index = quota_mat.index.union(df_close.index)
        quota_mat = quota_mat.reindex(union_index, fill_value=0)
        df_vwap_close = df_close.reindex(union_index, fill_value=0)
        df_vwap_close['PriceChange'] = (df_vwap_close['ClosePrice'] - df_vwap_close['PreClosePrice']) / df_vwap_close['PreClosePrice']
        quota_mat[bar_mode] *= df_vwap_close['PreClosePrice']
        quota_mat[bar_mode] /= dict_holdmv
        
        if alpha_mode:
            alpha_ret = (quota_mat[bar_mode] * df_vwap_close['PriceChange']).sum() - index_return
        else:
            alpha_ret = (quota_mat[bar_mode] * df_vwap_close['PriceChange']).sum()

        return alpha_ret


def calculate_daily_trades_return(dict_trades, dict_price, dict_holdmv, fee_ratio=0.00012, mode='dict'):
    dict_trades = deepcopy(dict_trades)
    dict_price = deepcopy(dict_price)
    dict_holdmv = deepcopy(dict_holdmv)
    if mode == 'dict':
        trades_ret_dict = {}
        date_list = list(dict_trades.keys())
        for date in date_list:
            print(date)
            if isinstance(fee_ratio, dict):
                fee = fee_ratio[date]
            else:
                fee = fee_ratio
            trades_ret_dict[date] = calculate_daily_trades_return(dict_trades[date], dict_price[date], dict_holdmv[date], fee_ratio=fee, mode='array')
        return trades_ret_dict
    else:
        df_merge = pd.merge(dict_trades, dict_price.reset_index(), on='SecuCode', how='left')
        df_merge['Volume'] *= (1 - 2 * df_merge['LongShort'])

        ls_flag = df_merge['Volume'] >= 0
        df_merge['PnL'] = df_merge['Volume'] * (df_merge['ClosePrice'] - df_merge['Price'] * (
                ls_flag * (1 + fee_ratio) + (1 - ls_flag) * (1 - STAMP_RATE - fee_ratio)))
        trades_ret = df_merge['PnL'].sum() / dict_holdmv

        return trades_ret


def calculate_daily_alpha_return(
        dict_position, dict_trades, dict_holdmv, dict_price, dict_index_price, fee_ratio, alpha_mode=True):
    dict_trades_ret = calculate_daily_trades_return(dict_trades, dict_price, dict_holdmv, fee_ratio=fee_ratio, mode='dict')

    if dict_position.get('compare', None) is not None:
        conlist_pos_alpha = []
        for pos_mode in dict_position:
            if pos_mode == 'compare':
                continue
            dict_pos_alpha_ret = calculate_daily_position_alpha_return_by_quota_mat(
                dict_position[pos_mode], dict_price, dict_index_price, dict_holdmv, mode='dict', bar_mode='PreCloseVolume', pre_mode=False, alpha_mode=alpha_mode)

            conlist_pos_alpha.append(pd.DataFrame.from_dict(dict_pos_alpha_ret, orient='index', columns=[pos_mode]))
        df_pos_alpha = pd.concat(conlist_pos_alpha, axis=1)
    else:
        dict_pos_alpha_ret = calculate_daily_position_alpha_return_by_quota_mat(
            dict_position, dict_price, dict_index_price, dict_holdmv, mode='dict', bar_mode='PreCloseVolume', pre_mode=False, alpha_mode=alpha_mode)
        df_pos_alpha = pd.DataFrame.from_dict(dict_pos_alpha_ret, orient='index', columns=['PosAlphaRet'])

    df_ret = pd.concat([
        df_pos_alpha,
        pd.DataFrame.from_dict(dict_trades_ret, orient='index', columns=['TradesRet'])],
        axis=1).reset_index().rename({'index': 'Date'}, axis='columns')

    if dict_position.get('compare', None) is None:
        df_ret['AlphaRet'] = df_ret['PosAlphaRet'] + df_ret['TradesRet']

    return df_ret


def calculate_daily_vwap_simu_return_by_weight_diff(
        weight_diff, df_vwap, tax_ratio=STAMP_RATE, fee_ratio=0.00015, mode='array', formula='alphasimu', splitbar=False, bar_mode=8):
    df_vwap = deepcopy(df_vwap)
    if mode == 'dict':
        alphasimu_dict = {}
        for date in weight_diff.keys():
            print(date)
            alphasimu_dict[date] = calculate_daily_vwap_simu_return_by_weight_diff(
                weight_diff[date], df_vwap[date], tax_ratio=tax_ratio, fee_ratio=fee_ratio,
                mode='array', formula=formula, splitbar=splitbar, bar_mode=bar_mode)
        return alphasimu_dict
    else:
        if isinstance(df_vwap, tuple) or isinstance(df_vwap, list): df_vwap = pd.concat(list(df_vwap), axis=1)
        union_index = weight_diff.index.union(df_vwap.index)

        weight_diff_mat = np.array(weight_diff.reindex(union_index, fill_value=0))
        df_vwap_close = df_vwap.reindex(union_index, fill_value=0)

        vwap_mat = np.array(df_vwap_close[[i for i in range(1, bar_mode + 1)]])
        close_mat = np.tile(np.array(df_vwap_close[['ClosePrice']]), bar_mode)
        if formula == 'alphasimu':
            cost_mat = np.where(weight_diff_mat > 0, 1 + fee_ratio, 1 - tax_ratio - fee_ratio)
            vwap_simu_mat = np.multiply(weight_diff_mat, close_mat / vwap_mat - cost_mat)
        else:
            pre_close_mat = np.tile(np.array(df_vwap_close[['PreClosePrice']]), bar_mode)
            cost_mat = np.multiply(np.where(weight_diff_mat > 0, 1 + fee_ratio, 1 - tax_ratio - fee_ratio), vwap_mat)
            vwap_simu_mat = np.multiply(weight_diff_mat, (close_mat - cost_mat) / pre_close_mat)

        if not splitbar:
            alpha_simu = np.nansum(vwap_simu_mat)
        else:
            alpha_simu = list(np.nansum(vwap_simu_mat, axis=0))
            alpha_simu.append(np.sum(alpha_simu))
        return alpha_simu


def calculate_daily_vwap_simu_return_by_trade_volume(
        date, df_trade_volume, holdmv, df_vwap=None, ret_type='return', tax_ratio=STAMP_RATE, fee_ratio=0.00015, bar_mode=8):
    if isinstance(df_trade_volume, dict):
        dict_res = {}
        for key, df in df_trade_volume.items():
            dict_res[key] = calculate_daily_vwap_simu_return_by_trade_volume(key, df, holdmv[key], df_vwap[key], ret_type=ret_type, tax_ratio=tax_ratio, fee_ratio=fee_ratio[key], bar_mode=bar_mode[key])
        return dict_res

    if df_vwap is None:
        df_vwap = get_n_min_vwap_twap_price(date, bar_mode=bar_mode).set_index('code')
    else:
        df_vwap = df_vwap.copy(deep=True)
        if 'code' in df_vwap.columns.to_list():
            df_vwap = df_vwap.set_index('code')

    union_index = df_trade_volume.index

    trade_volume = np.array(df_trade_volume.reindex(union_index, fill_value=0))
    df_vwap_close = df_vwap.reindex(union_index, fill_value=0)

    vwap_mat = np.array(df_vwap_close[[i for i in range(1, bar_mode + 1)]])
    close_mat = np.tile(np.array(df_vwap_close[['ClosePrice']]), bar_mode)

    cost_mat = np.where(trade_volume > 0, 1 + fee_ratio, 1 - tax_ratio - fee_ratio)
    vwap_simu_mat = np.multiply(trade_volume, close_mat - np.multiply(vwap_mat, cost_mat))

    if ret_type == 'pnl-vwap-bm':
        return pd.DataFrame(
            np.array([
                np.nansum(vwap_simu_mat, axis=1),
                np.nansum(np.abs(trade_volume * vwap_mat), axis=1)
             ]).T, index=union_index, columns=['VwapSimuPnL', 'QuotaDiffValue'])
    elif ret_type == 'pnl-mat':
        return pd.DataFrame(
            vwap_simu_mat, index=union_index, columns=[f'Ti{ibar}_PnL' for ibar in range(1, bar_mode + 1)])
    elif ret_type == 'pnl':
        return np.nansum(vwap_simu_mat)
    elif ret_type == 'return':
        return np.nansum(vwap_simu_mat) / holdmv
    elif ret_type == 'return-mat':
        return pd.DataFrame(
            vwap_simu_mat / holdmv, index=union_index, columns=[f'Ti{ibar}_Ret' for ibar in range(1, bar_mode + 1)])


def get_weight_diff_mat_groupby_price(product, weight_diff_df, df_weight, time_list):
    weight_diff_df = weight_diff_df[np.abs(weight_diff_df).sum(axis=1) != 0]
    weight_diff_df = weight_diff_df.reset_index().rename({'index': 'SecuCode'}, axis='columns')
    mat_weight_diff_price = pd.merge(weight_diff_df, df_weight, on='SecuCode', how='outer').fillna(0)
    mat_weight_diff_price = mat_weight_diff_price.sort_values('PreClosePrice')
    mat_weight_diff_price['Weight'] = np.nancumsum(mat_weight_diff_price['Weight'])

    quantile_array = np.linspace(0, 1, 11)
    mat_weight_diff_price['WeightClass'] = mat_weight_diff_price['Weight'].apply(
        lambda x: np.searchsorted(quantile_array, x))
    mat_weight_diff_price['WeightClass'] = mat_weight_diff_price['WeightClass'].apply(
        lambda x: 1 if x == 0 else x)
    mat_weight_diff_price['WeightClass'] = mat_weight_diff_price['WeightClass'].apply(
        lambda x: 10 if x >= 11 else x)
    mat_weight_diff_price['NumClass'] = 1

    agg_dict = {col: 'sum' for col in time_list}
    agg_dict.update({'PreClosePrice': 'mean', 'Weight': lambda x: x.tail(1), 'NumClass': 'sum'})
    mat_weight_diff_price = mat_weight_diff_price.groupby('WeightClass').agg(agg_dict).reset_index()
    mat_weight_diff_price['Product'] = product

    return mat_weight_diff_price
