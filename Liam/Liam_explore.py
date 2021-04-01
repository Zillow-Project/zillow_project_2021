import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

def train_validate_test_split(df, target, seed=123):
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

###################### ________________________________________
### Univariate

def explore_univariate(train, cat_vars, quant_vars):
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats
    
def freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table

###################### ________________________________________
#### Bivariate


def explore_bivariate(train, categorical_target, continuous_target, binary_vars, quant_vars):
    '''
    This function makes use of explore_bivariate_categorical and explore_bivariate_quant functions. 
    Each of those take in a continuous target and a binned/cut version of the target to have a categorical target. 
    the categorical function takes in a binary independent variable and the quant function takes in a quantitative 
    independent variable. 
    '''
    
    for binary in binary_vars:
        explore_bivariate_categorical(train, categorical_target, continuous_target, binary)
    for quant in quant_vars:
        explore_bivariate_quant(train, categorical_target, continuous_target, quant)

###################### ________________________________________
## Bivariate Categorical

def explore_bivariate_categorical(train, categorical_target, continuous_target, binary_var):
    '''
    takes in binary categorical variable and binned/categorical target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the binary categorical variable. 
    '''
    print(binary_var, "\n_____________________\n")
    ct = pd.crosstab(train[binary_var], train[categorical_target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, binary_var, categorical_target)
    mannwhitney = compare_means(train, continuous_target, binary_var, alt_hyp='two-sided')
    p = plot_cat_by_target(train, categorical_target, binary_var)
    print("\nMann Whitney Test Comparing Means: ", mannwhitney)
    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
    

    
def run_chi2(train, binary_var, categorical_target):
    observed = pd.crosstab(train[binary_var], train[categorical_target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected


def plot_cat_by_target(train, categorical_target, binary_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(categorical_target, binary_var, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[binary_var].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

    
def compare_means(train, continuous_target, binary_var, alt_hyp='two-sided'):
    x = train[train[binary_var]==0][continuous_target]
    y = train[train[binary_var]==1][continuous_target]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

###################### ________________________________________
## Bivariate Quant

def explore_bivariate_quant(train, categorical_target, continuous_target, quant_var):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(categorical_target)[quant_var].describe().T
    spearmans = compare_relationship(train, continuous_target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, categorical_target, quant_var)

    plt.show()
    scatter = plot_scatter(train, categorical_target, continuous_target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nSpearman's Correlation Test:\n", spearmans)
    print("\n____________________\n")


def compare_relationship(train, continuous_target, quant_var):
    return stats.spearmanr(train[quant_var], train[continuous_target], axis=0)

def plot_swarm(train, categorical_target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=categorical_target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, categorical_target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=categorical_target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_scatter(train, categorical_target, continuous_target, quant_var):
    p = sns.scatterplot(x=quant_var, y=continuous_target, hue=categorical_target, data=train)
    p = plt.title(quant_var)
    return p


######################### ____________________________________

### Multivariate


def explore_multivariate(train, categorical_target, binary_var, quant_vars):
    pair = sns.pairplot(data=train, vars=quant_vars, hue=categorical_target)
    plt.show()
    plot_all_continuous_vars(train, categorical_target, quant_vars)
    plt.show()    


def plot_all_continuous_vars(train, categorical_target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing survived. 
    '''
    my_vars = [item for sublist in [quant_vars, [categorical_target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=categorical_target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=categorical_target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()
    
def plot_violin_grid_with_color(train, categorical_target, binary_var, quant_vars):
    for quant in quant_vars:
        sns.violinplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_var, palette="Set2")
        plt.show()
        
def plot_swarm_grid_with_color(train, categorical_target, binary_var, quant_vars):
    for quant in quant_vars:
        sns.swarmplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_var, palette="Set2")
        plt.show()
                

# 1. Write a function that will take, as input, a dataframe and a list containing the column names of all ordered numeric variables. It will output a pairplot and a heatmap.
# 2. Write 2+ other functions for plotting in plots that will contribute to your exploration of what is driving the logerror.  Some ideas: 
#    - seaborn's relplot where you can use 4 dimensions: x & y are continuous or continuous-like, color is discrete with limited number of values, size is numeric & ordered. 
#    - seaborn's swarmplot (x, y, color)
#    - logerror is normally distributed, so it is a great opportunity to use the t-test to test for significant differences in the logerror by creating sample groups based on various variables.  e.g. Is logerror significantly different for properties in Los Angeles County vs Orange County (or Ventura County)?  Is logerror significantly different for properties that are delinquent on their taxes vs those that are not?  Is logerror significantly different for properties built prior to 1960 than those built later than 2000?
# 4. Chi-Squared test:  run at least one at least 1 $\chi^{2}$ test and summarize conclusions.  Ideas: If you split logerror into quartiles, you can expect the overall probability of falling into a single quartile to be 25%.  Now, add another variable, like bedrooms (and you can bin these if you want fewer distinct values) and compare the probabilities of bedrooms with logerror quartiles.  See the example in the Classification_Project notebook we reviewed on how to implement chi-squared.
# 5. write a function that will take a dataframe and a list of continuous-like column names to plot each combination of variables in the chart type of your choice.
# 6. Write a function that will take a dataframe and a list of continuous-like column names, limited discrete/categorical column names (<10), and names of columns with limited discrete/categorical values that could be used as color/hue (<5 or so).  It will plot each combination of variables in the chart type of your choice (that takes those types and number of dimensions) 

import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(df):
    p = sns.pairplot(df)
    return p

def heat_map(df):
    plt.figure(figsize=(8,6))
    q = sns.heatmap(df.corr(), cmap='RdYlBu', annot=True, center=0)
    return q