from ico_tools import ico_date_process





download_whitepaper(icodrops_download_path, icodrops_ended_details_path)


check_whitepaper_download(icodrops_ended_details_path, icodrops_download_path, icodrops_ended_checked_path)


convert_whitepaper(icodrops_ended_checked_path, icodrops_download_path, icodrops_whitepaper_converted_path)


topic_modeling(icodrops_whitepaper_converted_path, icodrops_topic_path, tech_words)


ner(icodrops_topic_path, icodrops_ner_path, remove_ner, remove_ner_person, remove_ner_organization, remove_ner_location)


icodrops_feature(icodrops_ner_path, icodrops_feature_path, current_year, bitcoin_path) ico_date_process


load_whitepapers(icodrops_feature_path, icodrops_wp_analysis_path, stop_words)


aff_prop(icodrops_wp_analysis_path, icodrops_af_analysis_path, icodrops_w2v_not_processed, do_plot=False)


sentiment(icodrops_af_analysis_path, icodrops_sentiment_path)