import re
import json
import pandas as pd
# import datetime
import re
from collections import Counter
import numpy as np
import time
import math
import sys
from prefixspan import PrefixSpan
quiz = pd.read_csv(filepath_or_buffer= "C:\\Users\\zhazhang\\Desktop\\GitHub\\quizzResult.csv")
data1 = pd.read_csv(filepath_or_buffer="C:\\Users\\zhazhang\\Desktop\\GitHub\\event_semester1.csv")
exam = pd.read_csv(filepath_or_buffer="C:\\Users\\zhazhang\\Desktop\\GitHub\\exam_result.csv")
data1 = (data1[data1['action'].isin(['Viewed'])])
quiz.score = quiz.score.map(lambda x: x*10)
exam['score'] = exam.score.map(lambda x: (x/18*100))
quiz['action'] = "Submitted"
event = pd.concat([data1[['userId','objectId','action','Timestamp']],quiz[['userId','objectId','action','Timestamp']]])
exam['action']='FinalExam'
exam['Timestamp'] = 1542835584754#1542835584754 # this is the right timestamp of exam
exam['objectId'] = 11209 # I faked an
event = event[event.objectId != 23133]
event = event[event.Timestamp < 1542835584755] # here cut the data at the time just after exam.
"""
Here is the treat of event which add the correlated quiz and exam score 
"""
user_list = list(set(event['userId']))
next_score = []
quiz1 = list(set(quiz.objectId))[0]
quiz2 = list(set(quiz.objectId))[1]
exam_time = exam.iloc[0]['Timestamp']
"""
judge if the learner is participate all the quizzes
"""
user_quized = []
for i in user_list:
    if ((len(quiz[quiz.userId==i])) == 2) & (i in list(exam.userId)):
        #quiz=quiz[quiz.userId!=i]
        user_quized.append(i)
#print(len(user_list))
event=event.sort_values(by='Timestamp', ascending=True)
event = event[event.userId.isin(user_quized)]

"""
Here we group the students in three groups
""" 
s_good = [] #top 30%
s_normal = []# middle 40%
s_bad = []# next 30%
exam=exam.sort_values(by='score', ascending=False)
for i in range(len(exam)):
    if i < 30:
        s_good.append(exam.iloc[i]["userId"])
    if i>=30 & i<70:
        s_normal.append(exam.iloc[i]["userId"])
    if i>=70 & i<100:
        s_bad.append(exam.iloc[i]["userId"])
        
print("student level dividing done")

def get_key(dic, value):
    """
    know the value, get keys
    """
    a = [k for k, v in dic.items() if v == value]
    return a[0]
 

"""
Here is the training data divided by 3 groups of students
"""
l_good = []
l_bad = []
l_normal = []
for user in s_good:
    l_path = []
    for i in range(len(event)):
        if user==event.iloc[i]["userId"]:
            l_path.append(event.iloc[i]["objectId"])
    l_good.append(l_path)
for user in s_normal:
    l_path = []
    for i in range(len(event)):
        if user==event.iloc[i]["userId"]:
            l_path.append(event.iloc[i]["objectId"])
    l_normal.append(l_path)
for user in s_bad:
    l_path = []
    for i in range(len(event)):
        if user==event.iloc[i]["userId"]:
            l_path.append(event.iloc[i]["objectId"])
    l_bad.append(l_path)
 
 

"""
Here is the recommendation process
"""
resourcelist= []
def recommend(trainingset=l_good, s_group=s_good, student=s_good[0], path_length=9, rl=resourcelist):
    
    # Here we put the influence or this stdent's learning log bigger. x10
    for i in range(30):
        trainingset.append(trainingset[s_group.index(student)])
    ps = PrefixSpan(trainingset)
    
    pattern = ps.topk(1000, filter=lambda patt, matches: len(patt)>1)# pattern lenth should bigger than 1
    pattern_time = {} #Here stores all pattern with appear times
    
    for i,element in enumerate(pattern):
        l_s = []# store pattern in this element
        s = ""
        for i in range(len(element[1])):
            if i==0:
                s=str(element[1][i])
            else:
                l_s.append(s+","+str(element[1][i]))
                s = str(element[1][i])
        for j in l_s:
            if j in pattern_time.keys():
                pattern_time[j]+=element[0]
            else:
                pattern_time[j]=element[0]
                
    # ordered pattern in list            
    pattern_time = sorted(pattern_time.items(),key = lambda pattern_time:pattern_time[1],reverse=True)
    print("pattern with time:",pattern_time)
    # delete repeat part
    #print(len(pattern_time))
    
    """
    Here is deduplication.
    we can't delete the item of list in for cycle. It will have 'index out of range problem'. 
    So we store the repeat index and delete after
    """ 
    delete_indice = [] 
    for k1 in range(len(pattern_time)):
        starter = pattern_time[k1][0].split(",")[0]
        ender = pattern_time[k1][0].split(",")[1]
        if starter == ender:
            delete_indice.append(k1)
        if pattern_time[k1]==pattern_time[-1]:
            break
            
        for k2 in range(k1+1,len(pattern_time)):
            #print(pattern_time[k2])
            temps_start = pattern_time[k2][0].split(",")[0]
            temps_end = pattern_time[k2][0].split(",")[1]
            if starter == temps_start:
                delete_indice.append(pattern_time[k2])
            if ender == pattern_time[k2][0].split(",")[1]:
                delete_indice.append(pattern_time[k2])    
                
    for  i in set(delete_indice):
        if i in pattern_time:
            pattern_time.remove(i)
        
    """
    Here we organise the path from pattern list.
    We should firstly find the head then finish the path.
    """
    element = []
       
    pattern_result = [x[0] for x in pattern_time] # delete pattern show times, keep pattern
    #print("unique pattern:",pattern_result)  
    store = []
    for i in range(len(pattern_result)):
        for j in range(len(pattern_result)):
            if i==j:
                continue
            if pattern_result[i].split(",")[0] in pattern_result[j]:
                store.append(pattern_result[i])
    path =list(set(pattern_result).difference(set(store)))[0]
    print("begin_node of path:",path)
    compt=0
    c_b = 0
    l_change = 2
    while compt < path_length-2: # First node has two element, so we should add path_length-2
        c_b+=1
        for i in pattern_result:
            if i.split(",")[0]==path.split(",")[-1]:
                path+=","+i.split(",")[-1]
                compt+=1
       
        if l_change==len(path):
            c_b+=1
        else:
            l_change=len(path)
        if c_b>100000:
             break
    print("path:",path)
    return path 

path = recommend()

"""
Bad student
"""
b_pm = []
for i in s_bad[:4]:
    #trainingset=l_normal, s_group=s_normal, student=s_normal[0], path_length=5, rl=resourcelist
    b_pm.append(recommend(l_bad,s_bad, i,10,resourcelist)) 
n_pm = []
"""
Bad student
"""
for i in s_normal[:4]:
    #trainingset=l_normal, s_group=s_normal, student=s_normal[0], path_length=5, rl=resourcelist
    n_pm.append(recommend(l_normal,s_normal, i,10,resourcelist)) 
g_pm = []
"""
Bad student
"""
for i in s_good[:4]:
    #trainingset=l_normal, s_group=s_normal, student=s_normal[0], path_length=5, rl=resourcelist
    g_pm.append(recommend(l_good,s_good, i,10,resourcelist)) 