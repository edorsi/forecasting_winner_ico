from ico_tools import convert_whitepaper
from ico_tools import topic_modeling
from ico_tools import ner
from ico_tools import ico_date_process
from ico_tools import load_whitepapers
from ico_tools import aff_prop
from ico_tools import sentiment
import os

base_filepath = r'C:\PythonPrjFiles\forecasting_winner_ico'
whitepaper_downloadpath = 'whitepapers\\'

def ico_preprocess(convert_flag=False, topic_flag=False, ner_flag=False, date_flag=False, load_flag=False, aff_flag=False, sentiment_flag=False):

    # download_whitepaper(icodrops_download_path, icodrops_ended_details_path)
    #
    #
    # check_whitepaper_download(icodrops_ended_details_path, icodrops_download_path, icodrops_ended_checked_path)
    #
    #
    if convert_flag:
        # multiprocessing
        convert_whitepaper(input_csv=os.sep.join([base_filepath, '03_icodrops_ended_list_checked.csv']),
                           output_csv=os.sep.join([base_filepath, '04_icodrops_whitepaper_converted.csv']),
                           download_path=os.sep.join([base_filepath, whitepaper_downloadpath]))

    if topic_flag:
        # multiprocessing
        topic_modeling(input_csv=os.sep.join([base_filepath, '04_icodrops_whitepaper_converted.csv']),
                       output_csv=os.sep.join([base_filepath, '05_icodrops_topic.csv']))
    #
    #
    if ner_flag:
        # multiprocessing
        ner(input_csv=os.sep.join([base_filepath, '05_icodrops_topic.csv']),
            output_csv=os.sep.join([base_filepath, '06_icodrops_ner.csv']))
    #
    if date_flag:
        # multiprocessing
        ico_date_process(input_csv=os.sep.join([base_filepath, '06_icodrops_ner.csv']),
                         output_csv=os.sep.join([base_filepath, '07_icodrops_feature.csv']),
                         bitusd_csv=os.sep.join([base_filepath, 'BTC-USD_yahoo_finance.csv']))
    #
    if load_flag:
        # multiprocessing
        load_whitepapers(input_csv=os.sep.join([base_filepath, '07_icodrops_feature.csv']),
                         output_csv=os.sep.join([base_filepath, '30_icodrops_wp_analysis.csv']))
    #
    if aff_flag:
        aff_prop(input_csv=os.sep.join([base_filepath, '30_icodrops_wp_analysis.csv']),
                 output_csv=os.sep.join([base_filepath, '32_icodrops_af_analysis.csv']))
    #
    #
    # sentiment(icodrops_af_analysis_path, icodrops_sentiment_path)

if __name__ == '__main__':
    ico_preprocess(convert_flag=False, topic_flag=False, ner_flag=False, date_flag=False, load_flag=False, aff_flag=True, sentiment_flag=False)
