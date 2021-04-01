import env
import pandas as pd

import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_connection(db, user=env.user, host=env.host, password=env.password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_zillow_data():
    '''
    Grab our data from path and read as dataframe
    '''
    
    df = pd.read_sql('''
                        SELECT *

                        FROM   properties_2017 prop  
                               INNER JOIN (SELECT parcelid,
                                                  Max(logerror),
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
                               AND unitcnt = 1;

                        ''', get_connection('zillow'))
    
    
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

def data_split(df, stratify_by='taxvaluedollarcnt'):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['taxvaluedollarcnt'])
    y_train = train['taxvaluedollarcnt']
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['taxvaluedollarcnt'])
    y_validate = validate['taxvaluedollarcnt']
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['taxvaluedollarcnt'])
    y_test = test['taxvaluedollarcnt']
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def minmax_scale(X_train, X_validate, X_test):
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # We fit on the training data
    # in a way, we treat our scalers like our ML models
    # we only .fit on the training data
    scaler.fit(X_train)
    
    train_scaled = scaler.transform(X_train)
    validate_scaled = scaler.transform(X_validate)
    test_scaled = scaler.transform(X_test)
    
    # turn the numpy arrays into dataframes
    X_train = pd.DataFrame(train_scaled, columns=X_train.columns)
    X_validate = pd.DataFrame(validate_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(test_scaled, columns=X_train.columns)
    
    return X_train, X_validate, X_test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~