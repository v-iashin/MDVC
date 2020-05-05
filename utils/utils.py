from time import strptime, localtime, mktime
import sys
import os

def timer(timer_started_at):
    # 20190911133748 (YmdHMS) -> struct time
    timer_started_at = strptime(timer_started_at, '%y%m%d%H%M%S')
    # struct time -> secs from 1900 01 01 etc
    timer_started_at = mktime(timer_started_at)
    
    now = mktime(localtime())
    timer_in_hours = (now - timer_started_at) / 3600
    
    return round(timer_in_hours, 2)
    
class HiddenPrints:
    '''
    Used in 1by1 validation in order to block printing of the enviroment 
    which is surrounded by this class 
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout