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
- The stuctures age effects log error because a homes age may make someone think that it is of lower quality but it may have been refurbished causeing a under valued home.
- Latitude and Longitude are drivers of log error because regions may be a higher priced area versus another.
- A homes quality is probably effecctiing logerror because homes with a lower quality may be being priced way undervalued.
- Tax vlaues may be affecting logerror because not all home prices are directly correlated to tax values. Although it is affects the price it does not lead to an exact number.

### Target variable
- logerror

</details>

    
## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Explore:
- We learned:
    - We have 46,416 accurate home values and 4,791 which are inaccurate.
        - Only about 9.36% are innacurate overall.
    - Reading percentages:
        - Accurate Percentages (between -.15 and 0.15)
            - ~69% in Los Angeles
            - ~23% in Oragne County
            - ~8% in Ventura
        - Over Percentages (between 0.15 and 1)
            - ~75% in Los Angeles
            - ~18% in Oragne County
            - ~7% in Ventura
        - Way Under Percentages (between -1 and -5)
            - ~67% in Los Angeles
            - ~12% in Oragne County
            - ~21% in Ventura
        - Under Percentages (between -0.15 and -1)
            - ~81% in Los Angeles
            - ~14% in Oragne County
            - ~5% in Ventura
        - Way Over Percentages (between 1 and 5)
            - ~52% in Los Angeles
            - ~38% in Oragne County
            - ~10% in Ventura
    - For the quality house age cluster we found that the majority of way under valued homes comes from home with a quality of 0.
    - The majority of over valued homes are older and of lower quality.
    - Homes with a low to medium structure tax value and a low land tax value tend to have a higher logerror than other homes.
    - North Downtown LA have no homes that have no undervalued homes.
    - Overall North Downtown LA has lowest logerror out of all areas in Southern California
    - Homes in Ventura are more often overvalued compared to their surrounding areas.
    
    
**Please note that LA has a significantly higher home population than both Orange county and Ventura**

### Stats
- Stat Test: Land and Structure Taxes
    - **Anova Test**:
        - Showed that there was a difference between log error of at least one of the 6 cluster created.
    - **T-Testing**:
        - Showed that the homes with low to medium amount of structure with low land tax value have a correlation to what is effecting our log error.
- Stats test: Latitude, Longitude, and House Age
    - **Anova Test**:
        - Shows that there is a difference between the log error of at least one of the 5 clusters created.
    - **T-Testing**:
        - Showed that Ventura and North Downtown LA were the most significant when it came to log error.
- Stats test: Quality, House Age, Room Count
    - **Anova Test**:
        - Shows that there is a difference between the log error of at least one of the 5 clusters created.
    - **T-Testing**:
        - Showed that when a homes quality equaled zero, newer homes with higher quality, and older homes with high quality had a relationship to logerror compared to others.

### Modeling:
- So our model performed better than the baseline.
    - Our R Squared OLS Baseline performed at a -0.004585 compared to our OLS R Squared score of 0.0000516.
        - You may be saying "Wow thats not that great"
            - BUT, relativly speaking it is a decent find. Because we are looking at log error, our goal is to be as close to 0 as we can.


***

    
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


### Stats Test 1:
- What is the test?
    - Anova
- Why use this test?
    - Find out if a cluster has significance to the logerror
- What is being compared?
    - Quality, house age, and room count

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - There is no difference between the log error means of each individual cluster
- The alternate hypothesis (H<sub>1</sub>) is ...
    - There is a difference between the log error means of at least one clusters.


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null
- move forward with Alternative Hypothesis 

- Summary:
    - F score of:
        - 4.478
    - P vlaue of:
        - 0.0012

### Stats Test 2: 
- What is the test?
    - T Test
- Why use this test?
    - To find statistical differences between the means of 2 or more clusters
- What is being compared?
    - Winning cluster of Latittude, Longitude, and House Age Anova Test

#### Results:
 - House quality = 0, old homes with high Quality, and new homes with high quality were affecting logerror to a degree.

### Stats Test 3:
- What is the test?
    - Anova
- Why use this test?
    - Find out if a cluster has significance to the logerror
- What is being compared?
    - Structure tax value and land tax value

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - There is no difference between the log error means of each individual cluster
- The alternate hypothesis (H<sub>1</sub>) is ...
    - There is a difference between the log error means of at least one clusters.


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null
- move forward with Alternative Hypothesis 

- Summary:
    - F score of:
        - 5.3376
    - P vlaue of:
        - 6.587e-05

### Stats Test 4: 
- What is the test?
    - T Test
- Why use this test?
    - To find statistical differences between the means of 2 or more clusters
- What is being compared?
    - Winning cluster of taxes

#### Results:
 - Homes with low to medium structure tax value and low land tax value affect logerror to some degree.
    

### Stats Test 5:
- What is the test?
    - Anova
- Why use this test?
    - Find out if a cluster has significance to the logerror
- What is being compared?
    - Latitude, Longitude, and House age

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - There is no difference between the log error means of each individual cluster
- The alternate hypothesis (H<sub>1</sub>) is ...
    - There is a difference between the log error means of at least one clusters.


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null
- move forward with Alternative Hypothesis 

- Summary:
    - F score of:
        - 6.6776
    - P vlaue of:
        - 0.000228

### Stats Test 2: 
- What is the test?
    - T Test
- Why use this test?
    - To find statistical differences between the means of 2 or more clusters
- What is being compared?
    - Winning cluster of Latittude, Longitude, and House Age Anova Test

#### Results:
 - Ventura, and North Downtown LA had an impact on log error.
    
***
​
    
</details>    

## <a name="model"></a>Modeling:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...

### Baseline

- Baseline Results: 
    - Median In sample = 0.16
    - Median Out of sample = 0.15
        
### Models and R<sup>2</sup> Values:
- Will run the following models:
    - Linear regression OLS Model
    - Lasso Lars
    - Tweedie Regressor
    - Polynomail Degree 2
    - Ploynomial Degree 3

- Other indicators of model performance
    - R<sup>2</sup> Baseline Value
        - -0.004585
    - R<sup>2</sup> OLS Value 
        - 0.00005159



### RMSE using Mean
    
Train/In-Sample:  0.16 
    
Validate/Out-of-Sample:  0.15
    

### RMSE using Median
Train/In-Sample:  0.16 
Validate/Out-of-Sample:  0.15

### RMSE for OLS using LinearRegression
    
Training/In-Sample:  0.15698193096987265 
    
Validation/Out-of-Sample:  0.1518694361646674
    

### RMSE for Lasso + Lars
    
Training/In-Sample:  0.012348907010552293 
    
Validation/Out-of-Sample:  0.011532822479710627
    

    
### RMSE for GLM using Tweedie, power=0 and alpha=0
    
Training/In-Sample:  0.01234045919349956 
    
Validation/Out-of-Sample:  0.011536767590909373
    

    
### RMSE for Polynomial Model, degrees=2
    
Training/In-Sample:  0.012288891953326782 
    
Validation/Out-of-Sample:  0.011543443686491118
    

    
### RMSE for Polynomial Model, degrees=3
    
Training/In-Sample:  0.012288891953326782 
    
Validation/Out-of-Sample:  0.011543443686491118


### Eetc:

## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Training/In Sample RMSE | Validation/Out of Sample RMSE | R<sup>2</sup> Value |
| ---- | ----| ---- | ---- |
| Baseline | 0.16  | 0.15 | -0.004585 |
| Linear Regression |  0.15698193096987265  | 0.1518694361646674 | 0.00005159 |
| Tweedie Regressor (GLM) | 0.01234045919349956  | 0.011536767590909373 | n/a |
| Lasso Lars | 0.012348907010552293  | 0.011532822479710627 | n/a |
| Polynomial Regression D2| 0.012288891953326782  | 0.011543443686491118 | n/a |
| Polynomial Regression D3| 0.012288891953326782  | 0.011543443686491118 | n/a |

- Why did you choose this model?
    - It was closer to 0 than our baseline.

## Testing the Model

- Model Testing Results
     - Out-of-Sample Performance:  0.1518694361646674


***

</details>  

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

We found that only about 9.36% of log error was inaccurate. Meaning that it was below -0.15 or above 0.15 rendering it inaccurate.

This gave us a small amount to work with. But in the end we were able to create a model to find certain drivers of the inaccurate log error.
Our model performed better than the baseline by a decent amount. With a R baseline of ~-0.0046 and our model performing at ~0.000052. Meaning we were able to get closer to 0 than our baseline.

We found that Ventura, north downtown LA, tax values, home quality, and a homes age affect loerror within their resepective cluster.

With further time we would like to look further into geographical location and tax values to see if there is a more specific reason for log error.

We recommend using our OLS model to be used within the field, in order to establish a closer zestimate score to what the selling price may be, in order to service our custoemrs even better.


    

</details>  

![Folder Contents](https://github.com/Zillow-Project/zillow_project_2021/blob/main/Caitlyn/photos/ScreenShot2021-04-06at12.52.26PM.png?raw=true)


>>>>>>>>>>>>>>>
.
