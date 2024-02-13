import pandas
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statistics
'''
col_names = ['Home_Won', 'Predicted DVOA Differential']
data = pandas.read_excel('Game Results 2009 No Covid.xlsx')
for i in range(0,20):
    ###Let's predict matchups without AGL first
    X = data['Predicted DVOA Differential'].array.reshape(-1, 1)
    y = data['Home_Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=i)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    TP = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TN = cnf_matrix[1][1]
    print('state {}, Non-AGL Model predicts {}% of games correctly'.format(i,(TP+TN)/sum([TP,FP,TN,FN])*100))
'''
###Now, let's do it again incorporating our new injury metric.
###To do this, we wil take the excel file, load in injury columns, and then run logreg again.
###We will not createa new excel sheet until the very end, when we are sure of our final model.
def z_score(mean, stdev, value):
    return ((value - mean)/stdev)

def calculate_INJ_score(string):
    #print(string)
    if string == 'BY':
        return 0
    answer = 0
    value = dict()
    #Changing the values here will alter the injury score calculation.
    #each entry is of the form value[position] = [Position_Importance, Threshold]
    #The Threshold is the depth chart value for which we consider a player to be important.
    #For example, most teams only play 1 QB all game so that threshold is 1.
    #Most teams play at least 3 WR though, so that is set to 3.
    #Feel free to change any of these values, and do some research on positional value to try and find alternative weights here
    value['FB'] = [1.25,1]
    value['LT'] = [6.00,1]
    value['RT'] = [6.00,1]
    value['ROLB'] = [8.6,1]
    value['LE'] = [5.9,1]
    value['WR'] = [21.0,3]
    value['CB'] = [26.9,4]
    value['HB'] = [2.99,2]
    value['QB'] = [53.3,1]
    value['C'] = [2.53, 1]
    value['RE'] = [5.9,1 ]
    value['DT'] = [4.31,2]
    value['TE'] = [5.37,1]
    value['LOLB'] = [8.6,1]
    value['RG'] = [4.89,1]
    value['P'] = [0.25,1]
    value['MLB'] = [9.86,2]
    value['SS'] = [25.3,1]
    value['K'] = [0.25,1]
    value['LG'] = [4.89,1]
    value['FS'] = [25.3,1]
    value['Unknown'] = [5,1]
    players = string.split('@')
    for player in players:
        sep = player.split(':')
        pos = sep[0]
        rank = int(sep[1])
        z_ovr = 3 + float(sep[2])
        #print(z_ovr)
        #Square to cancel out negative numbers)
        #print('{} {} {}'.format(pos, rank, z_ovr))
        if z_ovr < 1:
            z_ovr = 0
            #This is because if a player is bad enough, his loss dosen't matter.
            #We also don't want to count losing bad players as a net gain
        if rank <= value[pos][1]:
            answer += (value[pos][0]*z_ovr)
        else:
            answer += (value[pos][0]*(z_ovr))
    return answer

def Arizona_Correct():
    injuries = pandas.read_excel('Final Logistic Regression Data.xlsx')
    for i in range(len(injuries)):
        if injuries.loc()[i]['Home'] == 'ARI':
            injuries.loc()[i,'Home'] = 'AZ'
        if injuries.loc()[i]['Away'] == 'ARI':
            injuries.loc()[i,'Away'] = 'AZ'
        if injuries.loc()[i]['Winner'] == 'ARI':
            injuries.loc()[i,'Winner'] = 'AZ'
    injuries.to_excel('AZ Final Logistic Regression.xlsx', index = False)
            
    

def get_normalized_scores():
    injuries = pandas.read_excel('Comprehensive Injury Report.xlsx')
    output = pandas.read_excel('Final Logistic Regression Data.xlsx')
    scores = []
    final = dict()
    for i in range(len(injuries)):
        curr = injuries.loc()[i]
        if curr['Year'] not in final:
            final[curr['Year']] = dict()
        if curr['Team'] not in final[curr['Year']]:
            final[curr['Year']][curr['Team']] = dict()
        for week in range(1,23):
            score = calculate_INJ_score(curr[week])
            if score != 0:
                scores.append(score)
    avg = statistics.mean(scores)
    std = statistics.stdev(scores)
    for i in range(len(injuries)):
        curr = injuries.loc()[i]
        curr_t = curr['Team']
        curr_y = curr['Year']
        for week in range(1,23):
            score = calculate_INJ_score(curr[week])
            if score != 0:
                final[curr_y][curr_t][week] = (z_score(avg, std, score))
            else:
                final[curr_y][curr_t][week] = (100*100) #This indicates nothing - These values should never be considered anyway because the team isn't playing if InjReport is empty.
    
    return final


                
                    
                    
                
                    
                    
                
                
                
    
        

def get_array():
    bible = get_normalized_scores()
    output = pandas.read_excel('Final Logistic Regression Data.xlsx')
    HOME = []
    AWAY = []
    DIFF = []
    for i in range(len(output)):
        curr = output.loc()[i]
        year = curr['Year']
        home = curr['Home']
        away = curr['Away']
        week = int(curr['Week'])
        h_score = bible[year][home][week]
        a_score = bible[year][away][week]
        diff = h_score - a_score
        HOME.append(h_score)
        AWAY.append(a_score)
        DIFF.append(diff)
    answer = output.copy()
    answer['Home INJ'] = HOME
    answer['Away INJ'] = AWAY
    answer['INJ Differential'] = DIFF
    return answer
    
    

####It works. Now, let's import our logistic regression code and see what we get
####This model runs LR with 2 inputs, while the next model combines the two categories into one input and runs it there
ATP = 0
AFP = 0
AFN = 0
ATN = 0
ATTP = 0
ATFP = 0
ATFN = 0
ATTN = 0
data = get_array()
for i in range(30):
    X = data[['Predicted DVOA Differential','INJ Differential']]
    y = data['Home_Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=i)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    train_y_pred = logreg.predict(X_train)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    TP = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TN = cnf_matrix[1][1]
    train_mat = metrics.confusion_matrix(y_train, train_y_pred)
    TTP = train_mat[0][0]
    TFP = train_mat[0][1]
    TFN = train_mat[1][0]
    TTN = train_mat[1][1]
    ATP += cnf_matrix[0][0]
    AFP += cnf_matrix[0][1]
    AFN += cnf_matrix[1][0]
    ATN += cnf_matrix[1][1]
    ATTP += train_mat[0][0]
    ATFP += train_mat[0][1]
    ATFN += train_mat[1][0]
    ATTN += train_mat[1][1]
    #print('state {}, Two-Input Model predicts {}% of test games correctly'.format(i,(TP+TN)/sum([TP,FP,TN,FN])*100))
    #print('state {}, Two-Input Model predicts {}% of train games correctly'.format(i,(TTP+TTN)/sum([TTP,TFP,TTN,TFN])*100))
print('{} Run Avg, Two-Input Model predicts {}% of test games correctly'.format(i+1,(ATP+ATN)/sum([ATP,AFP,ATN,AFN])*100))
print('{} Run-Avg, Two-Input Model predicts {}% of train games correctly'.format(i+1,(ATTP+ATTN)/sum([ATTP,ATFP,ATTN,ATFN])*100))


###Now let's try for a one input model
###This model combines two variables into a single input score, and then runs LR.
###As you'll see when you run it, it performs much worse.
ATP = 0
AFP = 0
AFN = 0
ATN = 0
ATTP = 0
ATFP = 0
ATFN = 0
ATTN = 0
data_2 = get_array()
COMBINED = []
a = .75  #RAMACHANDRA: Changing this alpha value from a decimal in the range(0,1) will change how much each variable impacts the final single variable.
###a = 1 means this is just Predicted DVOA Differential
###a = 0 means this is just INJ Differential
###The model preforms poorly, similar to just using INJ Differential, for values of a not close to 1
for i in range(len(data_2)):
    P1 = data_2.loc()[i]['Predicted DVOA Differential']
    P2 = data_2.loc()[i]['INJ Differential']
    metric = (a*P1) + ((1-a)*P2)
    COMBINED.append(metric)
data_2['Var'] = COMBINED
for i in range(30):
    X = data_2['Var'].array.reshape(-1,1)
    y = data_2['Home_Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=i)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    train_y_pred = logreg.predict(X_train)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    TP = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TN = cnf_matrix[1][1]
    train_mat = metrics.confusion_matrix(y_train, train_y_pred)
    TTP = train_mat[0][0]
    TFP = train_mat[0][1]
    TFN = train_mat[1][0]
    TTN = train_mat[1][1]
    ATP += cnf_matrix[0][0]
    AFP += cnf_matrix[0][1]
    AFN += cnf_matrix[1][0]
    ATN += cnf_matrix[1][1]
    ATTP += train_mat[0][0]
    ATFP += train_mat[0][1]
    ATFN += train_mat[1][0]
    ATTN += train_mat[1][1]
    #print('state {}, One-Input Model predicts {}% of test games correctly'.format(i,(TP+TN)/sum([TP,FP,TN,FN])*100))
    #print('state {}, One-Input Model predicts {}% of train games correctly'.format(i,(TTP+TTN)/sum([TTP,TFP,TTN,TFN])*100))
print('{} Run Avg, One-Input Model predicts {}% of test games correctly'.format(i+1,(ATP+ATN)/sum([ATP,AFP,ATN,AFN])*100))
print('{} Run-Avg, One-Input Model predicts {}% of train games correctly'.format(i+1,(ATTP+ATTN)/sum([ATTP,ATFP,ATTN,ATFN])*100))

#Finally, we set up a model to test the effectiveness of our injury score metric
#This one only runs LogReg on the injury score variable. It is only for testing out how well we are tuning parameters.
ATP = 0
AFP = 0
AFN = 0
ATN = 0
ATTP = 0
ATFP = 0
ATFN = 0
ATTN = 0
for i in range(30):
    X = data_2['INJ Differential'].array.reshape(-1,1)
    y = data_2['Home_Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=i)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    train_y_pred = logreg.predict(X_train)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    TP = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    TN = cnf_matrix[1][1]
    train_mat = metrics.confusion_matrix(y_train, train_y_pred)
    TTP = train_mat[0][0]
    TFP = train_mat[0][1]
    TFN = train_mat[1][0]
    TTN = train_mat[1][1]
    ATP += cnf_matrix[0][0]
    AFP += cnf_matrix[0][1]
    AFN += cnf_matrix[1][0]
    ATN += cnf_matrix[1][1]
    ATTP += train_mat[0][0]
    ATFP += train_mat[0][1]
    ATFN += train_mat[1][0]
    ATTN += train_mat[1][1]
    #print('state {}, One-Input Model predicts {}% of test games correctly'.format(i,(TP+TN)/sum([TP,FP,TN,FN])*100))
    #print('state {}, One-Input Model predicts {}% of train games correctly'.format(i,(TTP+TTN)/sum([TTP,TFP,TTN,TFN])*100))
print('{} Run Avg, INJ-Differential predicts {}% of test games correctly'.format(i+1,(ATP+ATN)/sum([ATP,AFP,ATN,AFN])*100))
print('{} Run-Avg, INJ-Differential predicts {}% of train games correctly'.format(i+1,(ATTP+ATTN)/sum([ATTP,ATFP,ATTN,ATFN])*100))



