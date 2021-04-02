import env
import pandas as pd

import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_connection(db, user=env.user, host=env.host, password=env.password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Acquire
def get_zillow_data():
    '''
    Grab our data from path and read as dataframe
    '''
    
    df = pd.read_sql('''
                        SELECT *
                        FROM   properties_2017 prop  
                               INNER JOIN (SELECT parcelid,
                                                  logerror,
                                                  Max(transactiondate) transactiondate 
                                           FROM   predictions_2017 
                                           GROUP  BY parcelid, logerror) pred
                                       USING (parcelid) 
                               LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
                               LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
                               LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
                               LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
                               LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
                               LEFT JOIN storytype story USING (storytypeid) 
                               LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
                       WHERE transactiondate like '2017-%%-%%'
                               AND prop.latitude IS NOT NULL
                               AND prop.longitude IS NOT NULL
                               
                               AND propertylandusetypeid between 260 AND 266
                               OR propertylandusetypeid between 273 AND 279
                               AND NOT propertylandusetypeid = 274
                               AND unitcnt = 1;''', get_connection('zillow'))
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Prepare

def drop_50_pct_null(df):
    '''This function takes in the zillow df
    removes all columns and rows with 50% nulls or more
    returns df'''
    # Drop columns with 50% or more missing values
    df = df.dropna(axis = 1, thresh = 0.5 * len(df.index))
    # went from 67 columns down to 33
    # drop rows with 50% or more missing vlaues
    df.dropna(axis = 0, thresh = 0.5 * len(df.columns))
        # ended up not dropping any rows 
            # will remain in the function in case anything were to change later on
    return df

def clean_zillow(df):
    '''This function takes in the df
    applies all the cleaning funcitons previously created
    drops columns
    renames columns'''
    # assuming null value for pool means no pool
    df.poolcnt.fillna(0, inplace = True)
    # assuming null calie for fireplace means no null
    df.fireplacecnt.fillna(0, inplace = True)
    # drop features/rows with more than 50% null values
    df = drop_50_pct_null(df)
    # create dummy variables and add them to the df
    dummy_df =  pd.get_dummies(df['fips'])
    dummy_df.columns = ['in_los_angeles', 'in_orange_county', 'in_ventura']
    df = pd.concat([df, dummy_df], axis=1)
    df['fips'] = df.fips.replace('6,037.00', 6037)
    df['fips'] = df.fips.replace('6,059.00', 6059)
    df['fips'] = df.fips.replace('6,111.00', 6111)
    #create new feature house_age
    today = pd.to_datetime('today')
    df['house_age'] = today.year - df['yearbuilt']
    #create new feature tax_rate which is the monthyl taxes
    df['tax_rate'] = df.taxvaluedollarcnt / df.taxamount
    # create new feature for log_error_levels
    df['level_of_log_error'] = pd.qcut(df.logerror, q=5, labels=['L1', 'L2', 'L3', 'L4', 'L5'])
    # create new feature acres
    df['acres'] = df.lotsizesquarefeet/43560
    #drop features
    df = df.drop(['propertycountylandusecode', 'propertyzoningdesc', 
                 'heatingorsystemdesc', 'transactiondate',
                  'finishedsquarefeet12', 'id', 'censustractandblock',
                 'rawcensustractandblock', 'calculatedbathnbr', 
                 'assessmentyear'], axis=1)
    #rename features
    df = df.rename(columns={'heatingorsystemtypeid':'has_heating_system', 
                           'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 
                           'buildingqualitytypeid':'quality',
                           'calculatedfinishedsquarefeet':'square_feet', 
                           'fullbathcnt':'full_bathrooms',
                           'lotsizesquarefeet':'lot_square_feet', 
                           'propertylandusetypeid':'land_type',
                           'regionidcity':'city', 'regionidcounty':'county',
                           'regionidzip':'zip_code', 'roomcnt':'room_count',
                           'structuretaxvaluedollarcnt':'structure_tax_value',
                           'taxvaluedollarcnt':'tax_value', 
                           'landtaxvaluedollarcnt':'land_tax_value', 
                           'fireplacecnt':'has_fireplace',
                           'poolcnt':'has_pool'})
    # assuming that null means no heating bc it is southern CA
    df.has_heating_system.fillna('13', inplace = True)
    # change has_heating_system to binary
    df['has_heating_system'] = df.has_heating_system.replace([2.0, 7.0, 24.0, 6.0, 20.0, 13.0, 18.0, 1.0, 10.0, 11.0], 1)
    df['has_heating_system'] = df.has_heating_system.replace('13', '0')
    # drop all remaining null values
    # all of these are 1 unit counts
    df.unitcnt.fillna(1, inplace = True)
    df['unitcnt'] = df.unitcnt.replace([2.0, 3.0, 4.0, 6.0], 1)
    # replacing null in quality feature with its median range (6)
    df.quality.fillna(6.0, inplace = True)
    # replacing null in square_feet with its median
    df.lot_square_feet.fillna(7313, inplace = True)
     # replacing null in quality feature with its median
    df.square_feet.fillna(1511, inplace = True)
     # replacing null in quality feature with its median
    df.full_bathrooms.fillna(2, inplace = True)
     # replacing null in quality feature with its median
    df.yearbuilt.fillna(1970, inplace = True)
     # replacing null in quality feature with its median
    df.structure_tax_value.fillna(134871, inplace = True)
     # replacing null in quality feature with its median
    df.house_age.fillna(51, inplace = True)
     # replacing null in quality feature with its median
    df.city.fillna(25218, inplace = True)
     # replacing null in quality feature with its median
    df.zip_code.fillna(96410, inplace = True)
    # change has_fireplace to a binary
    df['has_fireplace'] = df.has_fireplace.replace([2.0, 3.0, 4.0, 5.0], 1)
    df['has_fireplace'] = df.has_fireplace.replace(0.0, 0)
    #fix has_pool to int
    df['has_pool'] = df.has_fireplace.replace(1.0, 1)
    df['has_pool'] = df.has_fireplace.replace(0.0, 0)
    # change unitcnt and has_heating_system to integers istead of objects
    df['has_heating_system'] = (df['has_heating_system'] == True ).astype(int)
    df['unitcnt'] = (df['unitcnt'] == True ).astype(int)
    #drop remaining null values
    df = df.dropna()
    # change la, oc, and vent into int
    df['in_los_angeles'] = (df['in_los_angeles'] == True ).astype(int)
    df['in_orange_county'] = (df['in_orange_county'] == True ).astype(int)
    df['in_ventura'] = (df['in_ventura'] == True ).astype(int)
    # set index as parcelid
    df = df.set_index('parcelid')
    # drop random column that came up
    df = df.drop(['Unnamed: 0'], axis=1)
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df
    
def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            # stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       # stratify=train_validate[target]
                                      )
    return train, validate, test


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def minmax_scale(train, validate, test):
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # We fit on the training data
    # in a way, we treat our scalers like our ML models
    # we only .fit on the training data
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # turn the numpy arrays into dataframes
    train = pd.DataFrame(train_scaled, columns=train.columns)
    validate = pd.DataFrame(validate_scaled, columns=train.columns)
    test = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~