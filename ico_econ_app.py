import pandas as pd
import numpy as np
import re


def token_utility(element):
    utility = ['utility', 'voucher', 'redeem']
    element = str(element).replace(' ', '')
    element = element.split(',')
    u = 0
    for s in element:
        if s.lower() in utility:
            u = 1
        else:
            pass
    return u


def sector(element):
    entertainment = ['Gambling', 'Gaming', 'Media', 'Social', 'Social Network']
    finance = ['Crowdfunding', 'Currency', 'Exchange', 'Finance', 'Payments', 'Trading']
    infrastructure = ['Artificial Intelligence', 'Blockchain', 'Blockchain Platform', 'Blockchain Service',
                      'Cloud Computing', 'Cloud Storage', 'Dapp', 'Data Service', 'IOT', 'Mobile', 'Network',
                      'Protocol', 'Security', 'Smart Contract', 'Verification']
    if element in entertainment:
        sec = 'Entertainment'
    elif element in finance:
        sec = 'Finance'
    elif element in infrastructure:
        sec = 'Infrastructure'
    else:
        sec = 'Other'
    return sec


def country(element):
    countries_us = ['United States of America', 'United States', 'USA', 'Texas', 'Tampa', 'St Louis', 'Seattle',
                    'Salt Lake City', 'San Diego', 'San Diego ', 'San Francisco', 'San Francisco Bay Area', 'San Jose',
                    'Santa Barbara', 'Santa Clara', 'Pittsburgh', 'Philadelphia', 'Pennsylvania', 'Ottawa', 'Orlando',
                    'North America', 'North Amercia', 'Nevada', 'New Orleans', 'New York', 'New York City',
                    'New York Citys',
                    'New York NY', 'Massachusetts', 'Los Angeles', 'Las Vegas', 'Jersey', 'Illinois', 'Greater Boston',
                    'Georgia', 'Florida', 'Denver Colorado', 'Denver', 'Delaware', 'Dallas', 'Cupertino', 'Chicago',
                    'California ', 'California', 'Boston', 'Austin', 'Arizona', 'America'
                    ]
    countries_eu = ['United Kingdom', 'United Kingdom of Great Britain', 'Switzerland', 'Sweden', 'Spain', 'Slovenia',
                    'Sicily',
                    'Sheffield', 'Serbia', 'Scotland', 'Tallinn', 'Stockholm', 'Rotterdam', 'Rome', 'Poland',
                    'Portugal',
                    'Prague', 'Oxford', 'Norway', 'Netherland', 'Netherlands', 'Munich', 'Monte Carlo', 'Malta',
                    'Luxembourg',
                    'London', 'Lithuania', 'Lisbon', 'Kingdom of Norway', 'Italy', 'Hamburg', 'Greece', 'Germany',
                    'Geneva',
                    'Frankfurt', 'France', 'Florence', 'Finland', 'European Union', 'Europe', 'Estonia', 'England',
                    'Eastern Europe', 'Dusseldorf', 'Dublin', 'Denmark', 'Czech Republic', 'Cyprus', 'Crimea',
                    'Cologne',
                    'Central Europe', 'Brussels', 'Bosnia', 'Berlin', 'Belgrade', 'Belgium', 'Bavaria', 'Barcelona',
                    'Austria', 'Athens', 'Amsterdam', 'Alicante', 'Albania'
                    ]
    element = [str(element).strip(',')]
    count_us = count_eu = count_oth = 0
    for s in element:
        if s in countries_us:
            count_us = count_us + 1
        elif s in countries_eu:
            count_eu = count_eu + 1
        else:
            count_oth = count_oth + 1
    if count_us >= count_eu >= count_oth:
        country_flag = 'US'
    elif count_eu >= count_us >= count_oth:
        country_flag = 'EU'
    else:
        country_flag = 'Other'
    return country_flag


def goal_received_log(element):
    element = re.sub(r'[^0-9]', '', element)
    try:
        element = np.log(element)
    except:
        element = np.nan
    return element


def ico_econ_app_preparation(input_csv, output_csv, verbose=False):
    df = pd.read_csv(input_csv, index_col=0)
    print(df.info()) if verbose else False
    print(df.head(5)) if verbose else False

    prefix = 'ea_model_'

    df_model = pd.DataFrame()
    df_model[prefix + 'wp_tokens'] = df['tokens_in_sentiment_analysis']
    df_model[prefix + 'goal_received_log'] = df['goal_received'].apply(
        lambda element: goal_received_log(element))

    df_model[prefix + 'patent_dummy'] = 0  # No patents recorded in PATSTAT
    df_model[prefix + 'wp_tech_dummy'] = df['tech_sen_pct'].fillna(0).astype(np.float64)
    df_model[prefix + 'wp_tech_dummy'] = df_model[prefix + 'wp_tech_dummy'].apply(
        lambda element: 1 if element >= 0.5 else 0)
    df_model[prefix + 'github_dummy'] = df['github'].apply(lambda element: 0 if element == 'None' else 1)

    df_model[prefix + 'tokens_offered_share'] = df['available_for_sale'].apply(
        lambda element: 0 if element == 'None' else element)
    df_model[prefix + 'tokens_offered_share'] = \
        df_model[prefix + 'tokens_offered_share'].replace('46 (40 on IEO)%', '46%')
    df_model[prefix + 'tokens_offered_share'] = \
        df_model[prefix + 'tokens_offered_share'].replace('45 (1.3 on IEO)%', '45%')
    df_model[prefix + 'tokens_offered_share'] = \
        df_model[prefix + 'tokens_offered_share'].replace('4 (Total 35)%', '35%')
    df_model[prefix + 'tokens_offered_share'] = df_model[prefix + 'tokens_offered_share'].apply(
        lambda element: str(element).replace(',', '.', ))
    df_model[prefix + 'tokens_offered_share'] = df_model[prefix + 'tokens_offered_share'].apply(
        lambda element: str(element).replace('%', '', ))
    df_model[prefix + 'tokens_offered_share'] = df_model[prefix + 'tokens_offered_share'].fillna(0).astype(np.float64)
    df_model[prefix + 'tokens_offered_share'] = df_model[prefix + 'tokens_offered_share'] / 100

    df_model[prefix + 'presale_dummy'] = df['presale'].apply(lambda element: 0 if element == 'None' else 1)
    df_model[prefix + 'duration_in_days'] = df['duration_days']

    df_model[prefix + 'token_utility'] = df['topic_words_000'].astype(str) + ', ' \
                                         + df['topic_words_001'].astype(str) + ', ' \
                                         + df['topic_words_002'].astype(str) + ', ' \
                                         + df['topic_words_003'].astype(str) + ', ' \
                                         + df['topic_words_004'].astype(str) + ', ' \
                                         + df['topic_words_005'].astype(str) + ', ' \
                                         + df['topic_words_006'].astype(str) + ', ' \
                                         + df['topic_words_007'].astype(str) + ', ' \
                                         + df['topic_words_008'].astype(str) + ', ' \
                                         + df['topic_words_009'].astype(str)
    df_model[prefix + 'token_utility'] = df_model[prefix + 'token_utility'].apply(
        lambda element: token_utility(element))

    df_model[prefix + 'token_supply_log'] = df['total_tockens'].apply(
        lambda element: 1 if element == 'None' else element)
    df_model[prefix + 'token_supply_log'] = df_model[prefix + 'token_supply_log'].apply(
        lambda element: re.sub(r'[^0-9]', '', str(element)))
    df_model[prefix + 'token_supply_log'] = df_model[prefix + 'token_supply_log'].fillna(0).astype(np.int64)
    df_model[prefix + 'token_supply_log'] = np.log(df_model[prefix + 'token_supply_log'])

    df_model[prefix + 'ETH_based_dummy'] = df['token_type'].apply(
        lambda element: 1 if element == 'ERC20' else 0)
    df_model[prefix + 'bitcoin_price_start'] = df['open_start']
    df_model[prefix + 'time_dummy'] = 'Q' + df['quarter_start'].astype(str) + '-' + df['year_start'].astype(str)
    df_model[prefix + 'twitter_dummy'] = df['twitter'].apply(
        lambda element: 0 if element == 'None' else 1)
    df_model[prefix + 'wp_team_dummy'] = df['person'].fillna('not available')
    df_model[prefix + 'wp_team_dummy'] = df_model[prefix + 'wp_team_dummy'].apply(
        lambda element: 0 if element == 'not available' or element == '' else 1)
    df_model[prefix + 'wp_sentence_count'] = df['sentences'].apply(
        lambda element: np.nan if element == 'not available' else element)
    df_model[prefix + 'location'] = df['location'].apply(
        lambda element: country(element))
    df_model[prefix + 'sector'] = df['ico_category_name'].apply(
        lambda element: sector(element))
    df = pd.concat([df, df_model], axis='columns', join='outer', ignore_index=False, sort=False)

    # review
    df_model[prefix + 'goal_received'] = df['goal_received'].apply(
        lambda element: re.sub(r'[^0-9]', '', element))
    df_model[prefix + 'goal_received'] = df_model[prefix + 'goal_received'].apply(
        lambda element: np.nan if element == '' else element)
    df_model[prefix + 'goal_received'] = df_model[prefix + 'goal_received'].astype(np.float64)
    df_model[prefix + 'goal'] = df['goal'].apply(
        lambda element: re.sub(r'[^0-9]', '', element))
    df_model[prefix + 'goal'] = df_model[prefix + 'goal'].apply(
        lambda element: np.nan if element == '' else element)
    df_model[prefix + 'goal'] = df_model[prefix + 'goal'].astype(np.float64)
    df_model[prefix + 'goal_pct'] = df_model[prefix + 'goal_received'] / df_model[prefix + 'goal']

    df.to_csv(output_csv, index_label='index')
    print(df_model.info()) if verbose else False
    print(df_model) if verbose else False
