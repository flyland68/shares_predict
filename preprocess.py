#!/bin/env python

import sys
import datetime
import csv


def readCSV():
    reader = csv.DictReader(sys.stdin)
    return dict([(row['date'], row) for row in reader])


def print_share_info(share_info):
    print "%(date)s\t%(open)s\t%(high)s\t%(close)s\t%(low)s\t%(volume)s\t%(amount)s" % share_info
    
    
def print_no_share_info(current, last_close):
    print '%(date)s\t%(last_close)s\t%(last_close)s\t%(last_close)s\t%(last_close)s\t0\t0' % {'date':current, 'last_close':last_close}


def process(start_date, end_date):
    shares_info = readCSV()
    last_info = None
    for delta_days in range((end_date - start_date).days):
        current_date = start_date + datetime.timedelta(delta_days)
        current = current_date.strftime('%Y-%m-%d')
        if current in shares_info:
            last_info = shares_info[current]
            print_share_info(last_info)
        else:
            print_no_share_info(current, last_info['close'])
            
            
if __name__ == '__main__':
    start_date = datetime.datetime.strptime(sys.argv[1], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d')
    process(start_date, end_date)
