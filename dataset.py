#!/bin/env python


def load_daily_infos(filename):
    daily_infos = []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')       
            fields = fields[1:]
            daily_info = [float(field) for field in fields]
            daily_infos.append(daily_info)
    return daily_infos


def generate_train_set(daily_infos, size):
    history_share_inputs = []
    date_feature_inputs = []
    outputs = []
    for i in range(size, len(daily_infos)):
        history_share_input = daily_infos[i-size:i]
        share = daily_infos[i]
        date_feature = '0' if share[4] == 0 else '1'
        
        history_share_inputs.append(history_share_input)
        date_feature_inputs.append(date_feature)
        outputs.append('%d' % round((share[2] / history_share_input[-1][2] - 1) * 100))
    
    return history_share_inputs, date_feature_inputs, outputs


if __name__ == '__main__':
    daily_infos = load_daily_infos('002027.tab')
    history_share_inputs, date_feature_inputs, outputs = generate_train_set(daily_infos, 3)
    print history_share_inputs[:10]
    print date_feature_inputs[:10]
    print outputs[:10]
    
