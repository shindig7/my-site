
# A Study of Eating Habits and Obesity
I've taken a recent interest in nutrition and the study of obesity, so in order to practice my data science skills and learn more about both subjects, I'll be doing some basic exploratory analysis of the BLS American Time Use Survey's Health Module Data.
### A Short Note on BMI
Body Mass Index (BMI) scores are often considered to be an inaccurate way to measure the health and body composition of individuals. Scores can often end up in the overweight or obese range for those with large amounts of muscle. That being said, these individuals are relatively rare in American society, and BMI was always meant as a score for measuring populations, not individuals. As I am studying over 10,000 individuals, I feel justified in using BMI as a heuristic for health.


```python
# Data Processing and Manipulation
import numpy as np
import pandas as pd
# Data Visualization
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
% matplotlib inline
# Statistics
import scipy.stats as stat
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from scipy.stats import chisquare
from math import sqrt

full_df = pd.read_csv('../input/ehresp_2014.csv')
full_df['CorpIndex'] = full_df['euhgt'] / (full_df['euwgt'] ** (1. / 3.) )
full_df['sqrtbmi'] = np.sqrt(full_df.loc[:,'erbmi'])

full_df.describe()
```

Just starting with the absolute basics here. According to the data dictionary, anyone with a bodyweight (euwgt) of -5 is pregnant. As I know less about the pregnancy than I do about nutrition and I'm looking for general insights, I'll be removing them from the dataset. I also removed anyone with a height (euhgt) less than 56 inches or a bodyweight (euwgt) less than 98 pounds, the bottom-coded values.


```python
full_df = full_df[np.logical_and(full_df['euwgt'] >= 98, full_df['euhgt'] >= 56)]  #euwgt is bottom-coded to 98 and euhgt is bottom-coded to 56
full_df.describe()
```

## The Basics



```python
bmi = full_df['erbmi']

total = float(full_df.shape[0])

underwght = full_df[full_df['erbmi'] < 18.5]
total_underwght = float(underwght.shape[0])

normal = full_df[np.logical_and(full_df['erbmi'] >= 18.5, full_df['erbmi'] < 25)]
total_normal = float(normal.shape[0])

overwgt = full_df[np.logical_and(full_df['erbmi'] >= 25, full_df['erbmi'] < 30)]
total_overwgt = float(overwgt.shape[0])

obese1 = full_df[np.logical_and(full_df['erbmi'] >= 30, full_df['erbmi'] < 35)]
total_obese1 = float(obese1.shape[0])

obese2 = full_df[np.logical_and(full_df['erbmi'] >= 35, full_df['erbmi'] < 40)]
total_obese2 = float(obese2.shape[0])

obese3 = full_df[full_df['erbmi'] > 40]
total_obese3 = float(obese3.shape[0])

total_non_normal = float(full_df[np.logical_or(full_df['erbmi'] >= 25, full_df['erbmi'] < 18.5)].shape[0])

print('Total Subjects: {}'.format(total))
print('Total Underweight: {} or {}%'.format(total_underwght, round(total_underwght*100/total, 2)))
print('Total Normal: {} or {}%'.format(total_normal, round(total_normal*100/total, 2)))
print('Total Overweight: {} or {}%'.format(total_overwgt, round(total_overwgt*100/total,2)))
print('Total Class 1 Obese: {} or {}%'.format(total_obese1, round(total_obese1*100/total, 2)))
print('Total Class 2 Obese: {} or {}%'.format(total_obese2, round(total_obese2*100/total, 2)))
print('Total Class 3 Obese: {} or {}%'.format(total_obese3, round(total_obese3*100/total, 2)))
print()
print('Total Subjects Outside Normal: {} or {}%'.format(total_non_normal, round(total_non_normal*100/total, 2)))

style.use('default')
fig=plt.figure(figsize=(10,7), dpi= 90, facecolor='w', edgecolor='b')
_ = plt.bar(np.arange(6), [total_underwght, total_normal, total_overwgt, total_obese1, total_obese2,
                       total_obese3], tick_label=['Underweight', 'Normal', 'Overweight',
                                                  'Class 1 Obese', 'Class 2 Obese', 'Class 3 Obese'])
plt.show()


```

As you can see, an large majority of Americans fall outside of the normal BMI range, and the number (66.15 %) is similar to the 2/3 number often referenced in media on the subject.


```python
fig=plt.figure(figsize=(10,7), dpi= 90, facecolor='w', edgecolor='b')
plt.hist(bmi, bins=61) # bin size was .97, so I just use 1; 61 is max - min bmi
plt.show()

test_stat, p_val = chisquare(bmi)
print('P-value is {} at test statistic {}'.format(round(p_val, 4), test_stat))
_, _, mean, var, skew, kurt = stat.describe(bmi)
print('Mean: {}'.format(round(mean, 2)))
print('Variance: {}'.format(round(var, 2)))
print('Standard Dev: {}'.format(round(sqrt(var), 2)))
print('Skewness: {}'.format(round(skew, 2)))
print('Kurtosis: {}'.format(round(kurt, 2)))
```

A p-value of 0.0 from the Chi squared test means that the data is normally distributed. The kurtosis of about 3 indicates this as well. There is a relatively high amount of skewness to the right, but this is to be expected as higher BMIs are possible and more likely than lower BMIs.

With this mean and standard deviation we can conclude that:
* About 68% of people have a BMI between 21.6 and 33.94
* About 95% of people have a BMI between 15.43 and 40.11

I'd also like to point out that the mean BMI is 27.77, which is firmly in the overweight category.

## Exercise and BMI
First up is a study of the relationship between exercise and the frequency of exercise and BMI. We're often told to exercise in order to decrease our weight


```python
exercise_df = full_df[full_df['euexfreq'] >= 0]  # Dataframe of only those who exercised at have an exercise frequency

exercise_freq = exercise_df['euexfreq']  # All exercise frequencies
exercising_bmi = exercise_df['erbmi']  # All BMI score

fig=plt.figure(figsize=(10,7), dpi= 100, facecolor='w', edgecolor='b')  # formatting
plt.scatter(exercise_freq, exercising_bmi, marker='.')
plt.title('Exercise Frequency vs. BMI')
plt.ylabel('Body Mass Index')
plt.xlabel('Exercise Frequency (# of Times in Last Week)')
plt.show()
```


```python
r = pearsonr(exercising_bmi, exercise_freq)
print(r)
```

So clearly there is no linear correlation between exercise frequency and BMI, but could there be other correlations?


```python
freq_mean_dict = {}
for value in np.ndenumerate(exercise_freq):
    if value[1] in freq_mean_dict.keys():
        pass
    else:
        freq_mean_dict[value[1]] = exercise_df[exercise_df['euexfreq'] == value[1]]['erbmi'].mean()

X = np.array(list(freq_mean_dict.keys()))
y = np.array(list(freq_mean_dict.values()))

# Graph size, style, and labels
fig=plt.figure(figsize=(13,7), dpi= 100, facecolor='w', edgecolor='b')
style.use('fivethirtyeight')
data, = plt.plot(X,y, 'o', label='Avg BMI')
plt.xticks(np.arange(min(X), max(X)+1, 1.0))
plt.title('Exercise Frequency vs. Avg. BMI')
plt.ylabel('Average Body Mass Index')
plt.xlabel('Exercise Frequency (# of Times in Last Week)')

# Cubic function for curve_fit
def cubic_fit(t, a, b, c, d):
    #return a*(t**2) + b*t + c
    return a * (t**3) + b * (t**2) + c*t + d
    #return a*(t**4) + b*(t**3) + c*(t**2) + d*t + e
    #return  a*(t**5) + b*(t**4) + c*(t**3) + d*(t**2) + e*t + f

# Linear function for curve_fit
def linear_fit(t, m, b):
    return m*t + b

# Fit the cubic equation to the data using curve_fit and plotting it as cube_fit
cpopt, cpcov = curve_fit(cubic_fit, X, y)
cube_fit, = plt.plot(X, cubic_fit(X, *cpopt), 'o')

lpopt, lpcov = curve_fit(linear_fit, X, y)
lin_fit, = plt.plot(X, linear_fit(X, *lpopt), 'o')

plt.legend((data, lin_fit, cube_fit), ('BMI', 'Linear Fit', 'Cubic Fit'), loc=2, fontsize='large')

plt.show()

print(cpopt)
#print(lpopt)
pearson_coeff = pearsonr(X,y)
print('Pearson Correlation Coefficient: {}'.format(round(pearson_coeff[0], 4)))
```

Instead of all the data, I tried graphing the mean BMI of each level of exercise frequency. I then tried fitting two lines, one cubic and one linear, to the data. The linear approximation does not seem to be very accurate and seems to show a positive correlation between exercise frequency and BMI score. This is confirmed by the Pearson correlation coefficient of 0.48, which shows a mild positive correlation.  The cubic approximation seems to be a little more accurate, especially in the lower range of frequencies, but the ramifications of using this approximation is a bit disheartening, as BMI only falls into the healthy range when exercising between 7 and 14 times per week, or 1-2 times per day. Anything more or less would result in a relatively steep increase in BMI score.
   
I considered removing the three points in the upper right of the graph, as they were clearly skewing the fit, however, the BMI scores recorded in these points are entirely possible and within 2 standard deviations of the mean. One would only need to be 280 lbs at a height of 5'8" to have a BMI of 42.6.

More unusual to me is the level of exercise reported. I am skeptical of those who claimed that they exercised 4-5 times per day, every day, in the last week. 

If any conclusion is to be drawn from the data, and specifically from the cubic fit line, it's that exercise is only beneficial to a certain point.


```python
style.use('fivethirtyeight')
fig=plt.figure(figsize=(10,7), dpi= 100, facecolor='w', edgecolor='b')
freq_dict = {value[1]: list(exercise_df[exercise_df['euexfreq'] == value[1]]['sqrtbmi']) for value in np.ndenumerate(exercise_freq)}
#_ = plt.boxplot(x='euexfreq', y='erbmi', data=exercise_df)
data = [freq_dict[k] for k in sorted(freq_dict.keys())]
_ = plt.boxplot(data)
plt.yticks(np.arange(3.5, 9, .5))
plt.ylabel('Square Root of BMI Score')
plt.xlabel('Exercise Frequency (Times Per Week)')
plt.show()
```

To make sure I wasn't missing anything by simply looking at the mean, I graphed the box plots of exercise frequency vs the square root of BMI\*.  There is tremendous variability in the BMI scores with no noticeable trendline in either direction. 

* I use square root of BMI for the box and violin plots to minimize the number of outliers represented in the plots.


```python
#exercise_bmi = list(np.sqrt(full_df[full_df['euexercise'] == 1]['erbmi']))
#print(type(exercise_bmi[0]))
#no_exercise_bmi = list(np.sqrt(full_df[full_df['euexercise'] == 2]['erbmi']))
exercise = (full_df[np.logical_and(full_df['euexercise'] > 0, full_df['erbmi'] > 0)])
exercise['sqrtbmi'] = np.sqrt(exercise.loc[:,'erbmi'])

labels = [1, 2]
fig=plt.figure(figsize=(5,10), dpi= 90, facecolor='w', edgecolor='b')

_ = sns.violinplot(x='euexercise', y='sqrtbmi', data=exercise, zorder=1)
plt.axhline(y=4.3, linewidth=2, color='black', zorder=0)
plt.axhline(y=5, linewidth=2, color='black', zorder=0)
plt.axhline(y=5.48, linewidth=2, color='black', zorder=0)
plt.axhline(y=5.9, linewidth=2, color='black', zorder=0)
plt.axhline(y=6.32, linewidth=2, color='black', zorder=0)
plt.xlabel('Exercised Last Week?')
plt.ylabel('Square Root BMI')
plt.xticks([0,1], ['Yes', 'No'])
plt.yticks(np.arange(3, 9.5, .5))
#plt.yticks(np.arange(3, 9, 0.25))
plt.show()
```

Finally, I plotted whether or not participants said they exercised in the last week *at all* against their BMI score. I used a violin plot instead of a box plot as it better shows where the majority of the score lay on the spectrum. For better visual reference, I added lines to represent the different categories defined by BMI; below the first line is considered underweight, between that and the next line is considered normal, etc. As we can see, the majority of people in both categories lie above normal range

As you can see, 
