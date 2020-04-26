from icodrops import IcodropsEnded
import multiprocessing
import os
import time
from multiprocessing import Pool
import pandas as pd


if __name__ == '__main__':

    # Display Options
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option("display.max_rows", 999)

    # Python Extensions
    chromedriver_path = "C:/PythonExtensions/chromedriver_win32/chromedriver.exe"

    # Folders settings
    base_path = 'C:/PythonPrjFiles/forecasting_winner_icos/'
    icodrops_path = base_path + 'icodrops/'
    icodrops_ended_list_path = icodrops_path + '01_icodrops_ended_list.csv'

    # Functions settings
    debug = False
    ico_list = None
    level = 'info'

    start_time = time.time()

    icodrops = IcodropsEnded(icodrops_ended_list_path, chromedriver_path, debug=debug, ico_list=ico_list, level=level)

    n_cores = multiprocessing.cpu_count() - 1
    pool = Pool(n_cores)
    icodrops.load_icos()

    data = pool.map(icodrops.get_details, icodrops.ico_list)
    pool.close()
    pool.join()

    df = pd.DataFrame(data)

    print(df.head(5)) if debug else False
    print(icodrops.__repr__()) if debug else False

    if not os.path.exists(icodrops_path):
        os.mkdir(icodrops_path)
    df.to_csv(icodrops_ended_list_path, index_label='index')


    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))