# Zillow Log Error Project

# <a name="top"></a>Finding Log Error for Zillow - README.md
![Zillow Logo](https://github.com/Zillow-Project/zillow_project_2021/blob/main/Caitlyn/photos/Screen%20Shot%202021-04-01%20at%205.57.59%20PM.png?raw=true)
​
***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Acquire & Prep](#acquire_and_prep)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___
​
​
## <a name="project_description"></a>Project Description:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Description
- Log error is based on our Zestimate price minus the actual sales price of a home, and then we take the log of the difference. But what is causing our errors? Thats what we are here to find out!

### Goals
- Uncovering what the drivers of the error in the Zestimate.
- Utilize clustering models to find these drivers.
- Presenting our finding to the Zillow data science team.

### Where did you get the data?
- Within the Zillow database found in the Codeup Sequel server, we joined specific tables onto our main data set (properties_2017). We also made specific parameters, within sequel, to fit our teams needs for this project.

</details>
    
    
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

### Projet Outline:
    
- Acquisiton of data through Codeup SQL Server, using env.py file with username, password, and host
- Prepare and clean data with python - Jupyter Labs
- Explore data
    - if value are what the dictionary says they are
    - null values
        - are the fixable or should they just be deleted
    - categorical or continuous values
    - Make graphs that show 
- Run statistical analysis
- Modeling
    - Make multiple models
    -Pick best model
    - Test Data
    - Conclude results
        
### Hypothesis
- Living in Los Angeles may be causing log error because of majorly different economic standings within the area.
- Latitude and Longitude are drivers of log error because one home may be in the hills, another may be on the coast, another may be in the dessert.
- Homes with heating systems are drivers of log error, because normally it is important but in southern California it is just nice to have but not a necessity.

### Target variable
- logerror

</details>

    
## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
​
### Explore:
- What are your key findings from explore?
​
​
### Stats
- What are your key findings from stats?
​
### Modeling:
- Model results?
​
​
***
​
    
</details>

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Data Used
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| acres  |  How many acres the property has | float |     
| acre_bins |  How many acres the property has binned | category |    
| bathrooms | Number of bathrooms in home including fractional bathrooms | float |
| bedrooms | Number of bedrooms in home | float |
| city |   City in which the property is located (if any) | float |
| county |   County in which the property is located) | float |
| fips |   Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details | float |
| full_bathrooms |  Number of full bathrooms (sink, shower + bathtub, and toilet) present in home | int |    
| has_fireplace |  If the house has a fireplace or not | int |
| has_heating_system |  If the house has a heating system or not | int |
| has_pool |  If the house has a pool or not | float |    
| house_age | year_built minus current year | int |
| in_los_angeles |  If the house is in Los Angeles or not | int |
| in_orange_county |  If the house is in Orange County or not | int |    
| in_ventura |  If the house is in Ventura or not | int |
| land_tax_value | The assessed value of the land area of the parcel  | float |    
| land_type |  Type of land use the property is zoned for | float |
| latitude | Latitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |
| level_of_log_error |  The log of the zestimate minus actual sold price of house binned| category |
| logerror* |  The log of the zestimate minus actual sold price of house | float |    
| longitude | Longitude of the middle of the parcel multiplied by 10<sup>6</sup> | float |    
| lot_square_feet |   Area of the lot in square feet | float |    
| lot_sqft_bins |  Area of the lot in square feet binned. | category |       
| quality |   Overall assessment of condition of the building from best (lowest) to worst (highest) | float | | room_count |  Total number of rooms in the principal residence | float |   
| structure_tax_value |  The assessed value of the built structure on the parcel | float |    
| square_feet | Calculated total finished living area of the home | float |
| square_feet_bins |  Calculated total finished living area of the home binned | category | 
| taxamount	|  The total property tax assessed for that assessment year | int | 
| tax_rate |  Rate of tax in the area | float |  
| tax_value | The total tax assessed value of the parcel | float  |
| untcnt |   Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...) | int |

    
\*  Indicates the target feature in this Zillow data.

***
</details>

## <a name="acquire_and_prep"></a>Acquire & Prep:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- Gather data from zillow database in the Codeup Sequel server.
    - Code to do this can be found in the wrangle.py file under the `get_zillow_data()` function

### Prepare Data
- To clean the data we had to:
    - Dop columns and rows with 50% or more null values 
    - Replace NULL values
    - Encode features
    - Create new features
    - Drop features
    - Rename features
    - Turn some features into binary features
    - Change some features to int64
    - Handle Outliers
    - Bin some larger features
- From here we :
    - Split the data into train, validate, and test
    - Split train, validate, and test into X and y
    - Scaled the data

​
| Function Name | Purpose |
| ----- | ----- |
| acquire_functions | DOCSTRING | 
| prepare_functions | DOCSTRING | 
| wrangle_functions() | DOCSTRING |
​
***
​

    
</details>



## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
- wrangle.py 

### Findings:
- 
    
    
| Function Name | Definition |
| ------------ | ------------- |
| select_kbest | This function takes in a dataframe, the target feature as a string, and an interger (k) that must be less than or equal to the number of features and returns the (k) best features |
| rfe | This function takes in a dataframe, the target feature as a string, and an interger (k) that must be less than or equal to the number of features and returns the best features by making a model, removing the weakest feature, then, making a new model, and removing the weakest feature, and so on. |
| train_validate_test_split | This function takes in a dataframe, the target feature as a string, and a seed interger and returns split data: train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test |
| get_object_cols() | This function takes in a dataframe and identifies the columns that are object types and returns a list of those column names |
| get_numeric_cols(X_train, object_cols) | This function takes in a dataframe and list of object column names and returns a list of all other columns names, the non-objects. |
| min_max_scale(X_train, X_validate, X_test, numeric_cols) | This function takes in 3 dataframes with the same columns, a list of numeric column names (because the scaler can only work with numeric columns), and fits a min-max scaler to the first dataframe and transforms all 3 dataframes using that scaler. It returns 3 dataframes with the same column names and scaled values. 
​
​
### Function1 used:
- Outcome of the use of the function 
​
### Function2 used:
- Outcome of the use of the function 
​
***
​
</details>    

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

​
### Stats Test 1:
 - What is the test?
 - Why use this test?
 - What is being compared?
​
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is... 
- The alternate hypothesis (H<sub>1</sub>) is ...
​
​
#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05
​
#### Results:
 - Results of statistical tests
​
 - Summary:
     - In depth take-a-ways from the results
​
### Stats Test 2 
 - What is the test?
 - Why use this test?
 - What is being compared?
​
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is... 
- The alternate hypothesis (H<sub>1</sub>) is ...
​
​
#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05
​
#### Results:
 - Results of statistical tests
​
 - Summary:
     - In depth take-a-ways from the results
​
### Stats Test 3
 - What is the test?
 - Why use this test?
 - What is being compared?
​
#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is... 
- The alternate hypothesis (H<sub>1</sub>) is ...
​
​
#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05
​
#### Results:
 - Results of statistical tests
​
 - Summary:
     - In depth take-a-ways from the results
​
***
​
    
</details>    

## <a name="model"></a>Modeling:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>
​
Summary of modeling choices...
​
### Baseline
​
​
- What is the first step?
    
```json
{
Input code here if you want...
}
```
- Next Step:
​
```json
{
Code...
}
```
​
- Baseline Results: 
    - What are the numbers we are trying to beat with our model.
        
***
​
### Models and R<sup>2</sup> Values:
- Will run the following models:
    - Model 1
        - brief summary of what the model does.
    - Model 2 
        - brief summary of what the model does.
    - etc.
​
- Other indicators of model performance with breif defiition and why it's important:
    - R<sup>2</sup> Value is the coefficient of determination, pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable. 
    - Essentially it is a statistical measure of how close the data are to the fitted regression line.
#### Model 1:
​
```json 
{
Model 1 code:
}
```
- Model 1 results:
    - Metric for Model 1:
        - Training/In-Sample:  **Results**
        - Validation/Out-of-Sample:  **Results**
    - Other metrics: (R<sup>2</sup> Value = )
​
​
### Model 2 :
​
```json 
{
Model 2 code:
}
```
- Model 2 results:
    - Metric for Model 1:
        - Training/In-Sample:  **Results**
        - Validation/Out-of-Sample:  **Results**
    - Other metrics: (R<sup>2</sup> Value = )
​
​
### Eetc:
​
## Selecting the Best Model:
​
### Use Table below as a template for all Modeling results for easy comparison:
​
| Model | Training/In Sample RMSE | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- | ---- |
| Baseline | 271194.48 | 272149.78 | -2.1456 x 10<sup>-5</sup> |
| Linear Regression | 217503.9051 | 220468.9564 | 0.3437 |
| Tweedie Regressor (GLM) | 217516.6069 | 220563.6468 | 0.3432 |
| Lasso Lars | 217521.8752 | 220536.3882 | 0.3433 |
| Polynomial Regression | 211227.5585 | 214109.6968 | 0.3810 |
​
- Why did you choose this model?
- 
​
## Testing the Model
```json
{
Model Testing Code...
}
```
- Model Testing Results
     - Out-of-Sample Performance:  **Results**
​
​
***
​
</details>  

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>
​
Reiterate explore findings, statistical analysis, and modeling take-a-ways
​
What could be done to improve the model?
What would you do with more time? 
​
Anything else of note worth adding? Add it here.
</details>  

![Folder Contents](https://github.com/Zillow-Project/zillow_project_2021/blob/main/Caitlyn/photos/ScreenShot2021-04-06at12.45.49PM.png?raw=true)


>>>>>>>>>>>>>>>
.
