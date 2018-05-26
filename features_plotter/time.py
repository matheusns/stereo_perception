from time import sleep
from datetime import datetime

start_time = datetime.now()
print('My Start Time', start_time)
sleep(65)

stop_time = datetime.now()
print('My Stop Time', stop_time)

elapsed_time = stop_time - start_time
print('My Elapsed Time', elapsed_time)  