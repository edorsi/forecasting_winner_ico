from ico_tools import ico_date_process
from ico_tools import convert_whitepaper
from ico_tools import topic_modeling
import os

base_filepath = r'C:\PythonPrjFiles\forecasting_winner_ico'
whitepaper_downloadpath = 'whitepapers\\'


def ico_preprocess(convert_flag=False, topic_flag=False):

    # download_whitepaper(icodrops_download_path, icodrops_ended_details_path)
    #
    #
    # check_whitepaper_download(icodrops_ended_details_path, icodrops_download_path, icodrops_ended_checked_path)
    #
    #
    if convert_flag:
        convert_whitepaper(input_csv=os.sep.join([base_filepath, '03_icodrops_ended_list_checked.csv']),
                           output_csv=os.sep.join([base_filepath, '04_icodrops_whitepaper_converted.csv']),
                           download_path=os.sep.join([base_filepath, whitepaper_downloadpath]))

    if topic_flag:
        topic_modeling(input_csv=os.sep.join([base_filepath, '04_icodrops_whitepaper_converted.csv']),
                       output_csv=os.sep.join([base_filepath, '05_icodrops_topic.csv']))
    #
    #
    # ner(icodrops_topic_path, icodrops_ner_path, remove_ner, remove_ner_person, remove_ner_organization, remove_ner_location)
    #
    #
    # ico_date_process(icodrops_ner_path, icodrops_feature_path, current_year, bitcoin_path)
    #
    #
    # load_whitepapers(icodrops_feature_path, icodrops_wp_analysis_path, stop_words)
    #
    #
    # aff_prop(icodrops_wp_analysis_path, icodrops_af_analysis_path, icodrops_w2v_not_processed, do_plot=False)
    #
    #
    # sentiment(icodrops_af_analysis_path, icodrops_sentiment_path)


ico_preprocess(convert_flag=False, topic_flag=True)
