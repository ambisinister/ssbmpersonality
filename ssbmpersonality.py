import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import csv

ps = pd.read_csv("ps.csv")
ps2 = pd.read_csv("split.csv")
ps3 = pd.read_csv("split2.csv")

def every_character(data):
    all_characters = {}
    for i, characters in enumerate(data['Character']):
        if not pd.isna(characters):
            for character in characters.split(','):
                if character.strip() in all_characters.keys():
                    all_characters[character.strip()].append(i)
                else:
                    all_characters.update({character.strip(): [i]})

    return all_characters

def split_personalities(data, splits):
    charvalues = []
    for key, values in splits.iteritems():
        print key
        for i, value in enumerate(values):
            row = data.iloc[value]
            row[2] = key
            charvalues.append({'Character':row[2], 'Openness':row['Openness'], 'Conscientiousness':row['Conscientiousness'], 'Extraversion':row['Extraversion'], 'Agreeableness':row['Agreeableness'], 'Neuroticism':row['Neuroticism']})

    dataframes = pd.DataFrame.from_dict(charvalues)
    dataframes.to_csv('split2.csv')

def anova_chars(data):
    factors = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    miniframes = split_personalities(ps, every_character(ps))
    for key,value in miniframes.iteritems():
        value['Character'] = key
        return value
    

def histo_metric(data, factor, binno=5):
    fig, ax = plt.subplots()
 
    ax.hist(data[factor].dropna(), density=1, bins=binno)   
    ax.set_xlabel('Confidence')
    ax.set_ylabel('% Responses')
    ax.set_title('SSBM Players - {0}'.format(factor))
    plt.xticks(range(1,binno+1))
    

    plt.show()


def all_personalities(data):
    axes = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    results = []
    for x in axes:
        omean = data.loc[:,x].mean()
        ostd = data.loc[:,x].std()
        results.append([x, omean, ostd])

    return results

def t_test_vs_norm(data, control):
    results1s = []
    results = []

    for axis in control:
        smash_group = data[axis[0]].dropna()

        #just confirm dist is normal, does nothing without prints
        k2, p = stats.normaltest(smash_group)

        #1 sample T test
        results1s.append(stats.ttest_1samp(smash_group, axis[1]))

        #2 sample T test using randomly generated sample from control group
        N = len(smash_group)
        control_group = np.random.normal(axis[1], axis[2], N)
        results.append(stats.ttest_ind(control_group,smash_group))

    return [results1s, results]


def barchart_all(control, smashers):

    factors = zip(*control)[0]
    meanssmash = zip(*smashers)[1]
    meanscontrol = zip(*control)[1]
    stdsmash = zip(*smashers)[2]
    stdcontrol = zip(*control)[2]
    semsmash = [x/np.sqrt(1689) for x in stdsmash]
    semcontrol = [x/np.sqrt(1689) for x in stdcontrol]

    fig, ax = plt.subplots()
    bar_width = 0.4
    opacity = 0.3
    error_config = {'ecolor': '0.3'}
    
    index = np.arange(len(factors))
    rects1 = ax.bar(index, meanssmash, bar_width,
                alpha=opacity, color='b',
                yerr=stdsmash, error_kw=error_config,
                label='SSBM')
    rects2 = ax.bar(index + bar_width, meanscontrol, bar_width,
                alpha=opacity, color='r',
                    yerr=stdcontrol, error_kw=error_config,
                label='Control')

    ax.set_xlabel('Factor')
    ax.set_ylabel('Scores')
    ax.set_title('SSBM Players vs Control')
    ax.set_xticklabels(('O', 'C', 'E', 'A', 'N'))
    ax.set_xticks(index+bar_width/2)
    ax.legend()

    fig.tight_layout()
    plt.show()


def barchart_variable(values, stds, sizes, factor, name, vals):

    sems = [x/np.sqrt(sizes[i]) for i, x in enumerate(stds)]

    fig, ax = plt.subplots()
    bar_width = 0.6
    opacity = 0.3
    error_config = {'ecolor': '0.3'}
    colors = ['b', 'r', 'c', 'g', 'm']
    
    index = np.arange(len(values))
    print values

    for i, group in enumerate(values):
        ax.bar(index[i]+bar_width, values[i], bar_width,
               alpha=opacity, color=colors[i%5],
               yerr=sems[i], error_kw=error_config,
               label=vals[i])
    
    ax.set_ylabel('{} Scores'.format(factor))
    ax.set_xlabel('{} - Self-Report'.format(name))
    ax.set_title('{} vs {}'.format(factor, name))
    #ax.set_xticklabels(vals)
    ax.set_xticklabels(['Very Below \n Average', '', '', '', 'Very Above \n Average'])
    ax.set_xticks(index+bar_width)

    fig.tight_layout()
    plt.show()


def barchart_specific(factors, chars, data):

    fig, ax = plt.subplots()
    bar_width = 0.4
    opacity = 0.3
    error_config = {'ecolor': '0.3'}
    colors = ['b', 'r', 'g', 'c', 'm']
    
    index = np.arange(len(factors))

    iter1 = 0
    iter2 = 0
    for factor in factors:
        print factor
        for i, char in enumerate(chars):
            print char
            print index+(bar_width*iter2)+(bar_width*iter1)
            print data[data['Character'] == char][factor].mean()
            ax.bar(index+(bar_width*iter2)+(bar_width*iter1), data[data['Character'] == char][factor].mean(), bar_width,
                   alpha=opacity, color=colors[i%5],
                   yerr=data[data['Character'] == char][factor].std(), error_kw=error_config,
                   label=char)
            iter2 += 1
        iter1 += 10

    ax.set_xlabel('Factor')
    ax.set_ylabel('Scores')
    ax.set_title('Character Selection vs Conscientiousness')
    ax.set_xticklabels(('Conscientiousness'))
    ax.set_xticks(index+bar_width*2)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
def anova_data(data, metric, filters, axes, chart=False):
    datasets = []
    
    for ax in axes:
        for char in filters:
            if not pd.isnull(char):
                #print "{} {}: {} (std {})".format(char, ax, data[data[metric]==char][ax].mean(), data[data[metric]==char][ax].std())
                datasets.append(data[data[metric]==char][ax].values)
        fv, pv = stats.f_oneway(*datasets)
        if pv < 0.05:
            print "p value for {} and {}: {}".format(metric, ax, pv)
            print "***"
            means = data.groupby([metric]).mean()[ax].tolist()
            stds = data.groupby([metric]).std()[ax].tolist()
            counts = data.groupby([metric]).count()[ax].tolist()
            vals = data[metric].unique()
            vals.sort()
            print data.groupby([metric]).mean()[ax]
            if chart:
                barchart_variable(means, stds, counts, ax, metric, vals)
        #print "p value for {} and {}: {}".format(metric, ax, pv)
        datasets = []
    
def char_representation(data):
    labels = []
    values = []
    for char in data.Character.unique():
        labels.append(char)
        values.append(len(data[data['Character'] == char]))

    pair = zip(labels, values)
    pair.sort(key=lambda x: x[1])
    for i, (x,y) in enumerate(pair):
        labels[i] = x
        values[i] = y
        

    index = np.arange(len(labels))
    plt.barh(labels, values, color='b')
    plt.xlabel('Frequency')
    plt.title('Character Selection Among Participants - Filtered')

    plt.show()
    
    print labels
    print values

def relationships(data, axes):
    
    exploreme = ['Skill', 'Referred', 'Playstyle', 'Intuition', 'Techskill', 'Cool', 'Honest', 'Ultimate', 'Approach', 'Projectiles', 'Wobbling', 'Activity', 'Grab', 'Drive', 'Monitors', 'Wakeup', 'Race', 'Class', '6x9', 'Belief']
    exploremetest = ['Playstyle']

    for group in exploreme:
        print group
        print "***"
        anova_data(data.dropna(subset=([group]+axes)), group, data[group].unique(), axes)

        
control = [["Openness", 3.92, .66],
           ["Conscientiousness", 3.45, .73],
           ["Extraversion", 3.25, .90],
           ["Agreeableness", 3.64, .72],
           ["Neuroticism", 3.32, .82]]

smashers = all_personalities(ps)

axes = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

print relationships(ps, axes)
