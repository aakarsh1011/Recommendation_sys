############## features file ################
######## feature extraction ################
import pickle
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn import preprocessing
import click
import json


## function to convert price into bins ##
def price_bin(Ad_Data):
    """
    Converts Price of the products in Ads into bins 
    Makes the encoding easier
    """
    Ad_Data['Price'].replace(np.nan, -1, inplace=True)
    Ad_Data['Price'] = pd.to_numeric(Ad_Data['Price'])
    Ad_Data['Price'] = pd.cut(Ad_Data['Price'], bins=25)

    return Ad_Data


## function to create lookup table ##
def ad_look_up(df):
    # lock-up table
    item_dict = {}
    df2 = df[['AdID', 'Title']].sort_values('AdID').reset_index()

    for i in range(df2.shape[0]):
        item_dict[str(df2.loc[i, 'AdID'])] = df2.loc[i, 'Title']

    return item_dict


## function to dummify the data ##
def dummify_data(Ad_Data):
    # dummify categorical features
    Ad_Data = pd.get_dummies(
        Ad_Data, columns=['Level_loc', 'Level_cat', 'Price'])

    Ad_Data = Ad_Data.sort_values('AdID').reset_index().drop('index', axis=1)
    return Ad_Data


## creating user_ad_interaction matrix ##
def user_ad_interaction(Interaction_Data):
    interactions = pd.pivot_table(
        Interaction_Data, index='UserID', columns='AdID', values='IsClick')

    # fill missing values with 0
    interactions = interactions.fillna(0)

    return interactions


## creating user dictionary ##
def mk_user_dct(Interaction_Data):
    UserID = list(Interaction_Data.index)
    user_dict = {}
    counter = 0
    for i in UserID:
        user_dict[i] = counter
        counter += 1

    return user_dict


## create CSR matrix ##
def csr_mat(Ad_Data, Interaction_Data):
    # convert Ad_data to csr matrix
    Ad_Data_transformed = np.array(Ad_Data, dtype=float)
    Ad_Data_transformed = pd.DataFrame(Ad_Data_transformed)
    Ad_Data_csr = csr_matrix(Ad_Data_transformed.values)

    # convert interaction data to csr matrix
    user_ad_interaction_csr = csr_matrix(Interaction_Data.values)

    return Ad_Data_csr, user_ad_interaction_csr

## making data for position of ad ##


def ad_pos_data(Ad_Data_pos):
    # label encoding all the IDs
    lbl_ad = preprocessing.LabelEncoder()
    lbl_search = preprocessing.LabelEncoder()
    lbl_location = preprocessing.LabelEncoder()
    lbl_category = preprocessing.LabelEncoder()
    lbl_parentCategory = preprocessing.LabelEncoder()
    lbl_subCategory = preprocessing.LabelEncoder()
    lbl_price = preprocessing.LabelEncoder()

    Ad_Data_pos.SearchID = lbl_search.fit_transform(Ad_Data_pos.SearchID)
    Ad_Data_pos.AdID = lbl_ad.fit_transform(Ad_Data_pos.AdID)
    Ad_Data_pos.LocationID = lbl_location.fit_transform(Ad_Data_pos.LocationID)
    Ad_Data_pos.CategoryID = lbl_category.fit_transform(Ad_Data_pos.CategoryID)
    Ad_Data_pos.ParentCategoryID = lbl_parentCategory.fit_transform(
        Ad_Data_pos.ParentCategoryID)
    Ad_Data_pos.SubcategoryID = lbl_subCategory.fit_transform(
        Ad_Data_pos.SubcategoryID)
    Ad_Data_pos.Price = lbl_price.fit_transform(Ad_Data_pos.Price)

    pickle.dump(lbl_ad, open('lbl_ad.pkl', 'wb'))
    pickle.dump(lbl_search, open('lbl_search.pkl', 'wb'))
    pickle.dump(lbl_location, open('lbl_location.pkl', 'wb'))
    pickle.dump(lbl_category, open('lbl_category.pkl', 'wb'))
    pickle.dump(lbl_parentCategory, open('lbl_parentCategory.pkl', 'wb'))
    pickle.dump(lbl_subCategory, open('lbl_subCategory.pkl', 'wb'))
    pickle.dump(lbl_price, open('lbl_price.pkl', 'wb'))

    return Ad_Data_pos


@click.command()
@click.option('--data-path', default='E:\sapient\personalize\Final_Master_Database.csv')
def feature_ext(data_path):
    """
    Takes in a path of csv file 

    Returns:
    1. CSR_matrix data of Ads for recommendation sys
    2. User-Ad csr matrix for recommendation sys
    3. Item look up dictionary for recommendation sys
    4. User look up dictionary for recommendation sys
    5. Data for Ad_position model

    """
    df = pd.read_csv(data_path)

    # recommendation data set
    Ad_Data = df[['SearchID', 'AdID', 'IsClick',
                  'LocationID', 'CategoryID', 'Level_loc', 'Level_cat',
                  'ParentCategoryID', 'SubcategoryID', 'Price', 'UserID']]

    Interaction_Data = df[['UserID', 'AdID', 'IsClick']]

    # position of ad
    # using original data, no dummies
    Ad_Data_pos = df[['SearchID', 'AdID',
                      'LocationID', 'CategoryID', 'Level_loc', 'Level_cat',
                      'ParentCategoryID', 'SubcategoryID', 'Price', 'Position', 'HistCTR']]

    # Converting prices of products into bins
    Ad_Data = price_bin(Ad_Data)
    item_dict = ad_look_up(df)
    Ad_Data = dummify_data(Ad_Data)
    interactions = user_ad_interaction(Interaction_Data)
    user_dict = mk_user_dct(interactions)
    Ad_Data_csr, user_ad_csr = csr_mat(Ad_Data, interactions)
    Ad_Data_pos = ad_pos_data(Ad_Data_pos)

    # Ad_Data_csr, user_ad_csr, interactions, item_dict, user_dict, Ad_Data_pos
    # save artifacts
    sparse.save_npz("Ad_Data_csr.npz", Ad_Data_csr)  # Ad_Data_csr
    sparse.save_npz("user_ad_csr.npz", user_ad_csr)  # user_ad_csr

    interactions.to_pickle("interaction.pkl")  # interactions

    pickle.dump(user_dict, open('user_dict.pkl', 'wb'))
    #pickle.dump(item_dict, open('item_dict.pkl', 'wb'))

    with open('item_dict.json', 'w') as fp:
        json.dump(item_dict, fp)  # item_dict

    #with open('user_dict.json', 'w') as fp:
     #   json.dump(user_dict, fp)  # user_dict

    Ad_Data_pos.to_pickle("Ad_Data_pos.pkl")  # Ad_Data_pos

    return


if __name__ == "__main__":
    feature_ext()
