import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import copy
import time
import random 

import pandas as pd
import json
import pandas as pd
import re
from collections import Counter
import numpy as np
import time
import math

used_data_types_dict = {
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id':'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

train_df = pd.read_csv(
    'train.csv',
    usecols=used_data_types_dict.keys(),
    dtype=used_data_types_dict
)

#初步读取question数据
questions_df = pd.read_csv('questions.csv')


# 这里筛选100个最火的lecture以及和他们对应的quiz
# 注意lecture的tag和id是不一样的。
train_l = train_df[train_df['content_type_id']==1]
lec_dis = train_l['content_id'].value_counts()
lec_dis.index
lec_top100 = list(lec_dis.index)[:100]

#w将数据按照quiz分成五个训练章节
part_1 = copy.deepcopy(questions_df[questions_df["part"]==1])
part_2 = copy.deepcopy(questions_df[questions_df["part"]==2])
part_3 = copy.deepcopy(questions_df[questions_df["part"]==3])
part_4 = copy.deepcopy(questions_df[questions_df["part"]==4])
part_5 = copy.deepcopy(questions_df[questions_df["part"]==5])


#判断两个list是否有交集
def inter(a,b):
    return list(set(a)&set(b))
lecture_df = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

state = []
state_q1 = part_1["question_id"].tolist()
lec = []#lec_top100#存放出现过的相关的lecture
for indx,q in enumerate(state_q1):
    temp = part_1[part_1['question_id']==q]['tags'].tolist()
    temp = temp[0].split(' ')
    temp = [int(x) for x in temp]
    for j in temp:
        temp_l = lecture_df[lecture_df['tag']==j]['lecture_id'].tolist()
        
        if inter(temp_l, lec_top100):
            state.append(q)
            
state = state+ lec_top100

print("total states for part 1:",len(state))
#挑选两万个随机的用户，并将其存在了这个表格中。 
f = open('user_list.txt')
line = f.readline()
f.close()
temp = line.split(' ')
user = [int(x) for x in temp[1:-2]]

# 将traindf缩减为原来的十分之一。
train = train_df[train_df.user_id.isin(user)]
#del train_df
train = train[train.content_id.isin(state)]

pd.options.display.float_format = '{:.11g}'.format #这里是修改输出模式


t_q = train[train["content_type_id"]==0]
t_l = train[train["content_type_id"]==1]
print("question and lecture's length",len(t_q),len(t_l))

t_q.sort_values(by=['user_id','timestamp'])
user_question = t_q['user_id'].value_counts()
print(statistics.mean(user_question))
print(statistics.median(user_question))
t_l.sort_values(by=['user_id','timestamp'])
user_lecture = t_l['user_id'].value_counts()
print(statistics.mean(user_lecture))
print(statistics.median(user_lecture))
print("max for q:",user_question)
print("max for q:",user_lecture)

user = random.sample(user, 1000)#先随机取100个用户试试水。 
# 将traindf缩减为原来的十分之一。
train_10 = train_df[train_df.user_id.isin(user)]
#del train_df
train_10 = train_10[train_10.content_id.isin(state)]

train_10 = train_10.sort_values(by=['user_id','timestamp'])


class Student_problem:
    def __init__(self, S, A,Omega,M,event_count,resourceIndex,event,userid = " "):
        self.S = S
        self.A = A
        self.Omega = Omega
        self.M = M
        self.gamma = 0.95
        self.event_count = event_count
        self.resourceIndex = resourceIndex
        self.event = event
        self.userid = userid
        
    #def similar_student(self, userid, event, mentor):
        
    def transition_function(self, state, action):
        """
        指定当前的s和a得到的下一个s'的概率分布字典
        这里需要一个计算M的矩阵。
        """
        transition = {}
        # resource, visited, control level
        s_c = state.split('level')[0]
        s_n = action.split('to')[1]
        """
        if terminate_state == s_c:
        # Add terminate state here.
            proba = 1
            level = score_to_control_level(int(self.event[(self.event.objectId==get_key(resourceIndex ,int(s_c))) &
            (self.event.usertId==self.userid)].nearest_score.iloc[0]))
            return {transition[]}
        else:"""
        proba = M[int(s_c)][int(s_n)] # the % of each action
        for j in range(3):
#               返回值应该是state和对应的概率 的字典
            t_temp = (s_n+"level"+str(j))
            transition[t_temp] = (proba/3)  # Here 我们返回的是3个状态的平均值
#         if str(resourceIndex[11209]) in s_n:
#             return 0
        return transition
    
    
    def reward_function(self, state, action):
        global reward_table
        a = action.split('to')
        s = state.split('level')[0]
        return reward_table[int(a[0])][int(a[0])]
        
    def observation_function(self, state, action):
        """
        on sais state nextet a, on calcule o et le met dans une dict
        
        Score high, 50% level2, score low, 50% level0, score middle, 50% 1
        
        """
        # Control level, It should be related with quiz.
        O = {} # Observation space
        e = self.event[(self.event.userId==self.userid)]
        s_n= state.split('level')[0]
        
        reward = self.reward_function(state, action)
        
        if s_n == action.split('to')[1]:
            
            if reward >= 20:
                O[("observation " + s_n +'level0')] = 0.1
                O[("observation " + s_n +'level1')] = 0.4
                O[("observation " + s_n +'level2')] = 0.50
            elif (reward < 20) & (reward>=0):
                O[("observation " + s_n +'level0')] = 0.25
                O[("observation " + s_n +'level1')] = 0.5
                O[("observation " + s_n +'level2')] = 0.25
            else:
                O[("observation " + s_n +'level0')] = 0.5
                O[("observation " + s_n +'level1')] = 0.4
                O[("observation " + s_n +'level2')] = 0.1
        return O
    
    
    def merge_dict(self, x, y):
        for k, v in x.items():
            if k in y.keys():
                y[k] += v
            else:
                y[k] = v
        return y

    def get_key(self, dic, value):
        """
        know the value, get keys
        """
        a = [k for k, v in dic.items() if v == value]
        return a[0]

    
class Solver():
    def __init__(self, pomdp, horizon,action_previous,result,policy):
        self.pomdp = pomdp
        self.gamma = 0.95
        self.error = 0.001
        # Create a table to store the value for every s at each time step k
        self.max_iter = 10000
        self.horizon = horizon
        self.action_previous = action_previous
        self.result = result
        self.policy = policy
        
    # ------------belief update-------------------------
    def belief_update(self, action, obs, b):
        b_new = {}
        total = 0
        dict_next_state_merged = { }
        # First compute what's the possible next states and their probs
        for state in b:
            dict_next_state = self.pomdp.transition_function(state, action)
            if dict_next_state==0:
                print("terminal state")
                return 
            # compute Σs in S p(s'|s, a)b(s) for each s', storing the dict in the dict_next_state_merged
            for s_new in dict_next_state:
                dict_next_state[s_new] *= b[state]
            dict_next_state_merged = self.pomdp.merge_dict(dict_next_state, dict_next_state_merged)
#         print('dict_next_state_merged',dict_next_state_merged)
        # for each s', compute the prob of having obs, then compute the product
        for next_state in dict_next_state_merged:
            dist_obs = self.pomdp.observation_function(next_state, action)
            pr_obs = 0
            if obs in dist_obs:
                pr_obs = dist_obs[obs]
            summation = dict_next_state_merged[next_state]
            b_new[next_state] = (pr_obs * summation)
#         print('b_new',b_new)

    # --- normalize the whole probability----
        for s in b_new:
            total += b_new[s]
        for s in b_new:
            b_new[s] = b_new[s]/total

        return b_new

    # ------------ calc expected reward ----------
    def exp_reward(self, b, a,ap):
        # for each state in the prob distribution of b
        exp_r = 0
        for s in b:
            reward = self.pomdp.reward_function(s, a,ap)
            # calc the sum
            exp_r += b[s] * reward
        return exp_r

    # ------------ derive possible states from a given belif b----------
    def possible_states(self, b):
        poss_states = []
        for elem in b:
            poss_states.append(elem)
        return poss_states

    # ----------- derive poss --------------
    
    # ----------- derive poss --------------
    def get_policy(self):
        return self.policy
    
    def get_actions(self):
        return self.action_previous
#------------Recursive dynamic programming-----------------------
    def recur_dyn(self, b, k):
        global difficulty
        ap = self.action_previous
#         if 'result' not in locals.keys():
#             result = {}
        # return value_future
        if k == self.horizon:
#             print("return 0")
            return 0
        if k < self.horizon:
            Vmax = -10000
            Vmax2 = -10000
            
            # Calculate the expected reward, the sum of the product of the prob of each state with their reward
            # rw_exp = self.exp_reward(b, a)
            # print("--------------")
            # print("rw_exp_vec",rw_exp)
            
            for a in self.pomdp.A:
                
                
#                 print("a is ", a)
                # rw_exp = self.exp_reward(b, a)
                
                # 这里控制所有action形成一条链，并能减少计算复杂度。
#                 a_last = self.action_previous[-1]
#                 if a != a_last:
#                     if a.split('to')[0] != a_last.split('to')[-1]:
#                         continue 
#                 else:
#                     continue
#                 if a=='0to1':
#                     self.pomdp.A.remove('0to1')
                # 1. compuet reward
                # for o in (possible observations)
                #     possible observations need to be derived from possible states (b)
                
                rw_exp = 0
                for state in b:
                    if state.split("level")[0] != a.split("to")[0]:
                        continue
                    # calc the reward
                    reward = self.pomdp.reward_function(state, a)
                    # calc the exp reward
#                     print("state",state)
                    # print("reward",reward)
                    rw_exp += b[state] * reward
    
                    # store best reward expectation with action_state
#                     state_without_level = state.split("level")[0]
#                     if state_without_level not in self.result.keys():
#                         self.result[state_without_level] = [a, rw_exp]
#                     else:
#                         if a not in self.result[state_without_level] and self.result[state_without_level][-1] < rw_exp :
#                             self.result[state_without_level] = [a,rw_exp]
#                         elif a not in self.result[state_without_level] and self.result[state_without_level][-1] == rw_exp:
#                             self.result[state_without_level].insert(1,a)
#                         elif a in self.result[state_without_level] and self.result[state_without_level][-1] < rw_exp:
#                             self.result[state_without_level] = [a,rw_exp]
                # 2. compute all o available
                obs_possible = []
                for state in b:
                    if state.split("level")[0] != a.split("to")[0]:
                        continue
                    dist_next_state = self.pomdp.transition_function(state, a)
                    
                    for next_state in dist_next_state:
                        dist_obs = self.pomdp.observation_function(next_state, a)
                        for obs in dist_obs:
                            # add to P(obs) b(s)*P(s'|s,a)*P(o|s')
                            if obs not in obs_possible:
                                obs_possible.append(obs)
                                # 因为observation possible是空的所以出现了问题。
                        # raise Exception("to correct obs dict")
#                 print('dist_obs',dist_obs)
#                 print("length of obs_possible", len(obs_possible))
#                 print("obs_possible[0]", obs_possible[:1])
                # 3. compute V(b,a)
                value_future = 0
                for obs in obs_possible:
                    if (obs.split(" ")[1]).split('level')[0] != a.split("to")[1]:
                        continue
                    # compute V(b,a,o), update belief state.
                    # print("************* UPDATE ************")
                    b_new = self.belief_update(a, obs, b)
                    v_b_new = self.recur_dyn(b_new, k + 1)

                    # compute P O
                    PO = 0
                    for state in b:
                        dist_next_state = self.pomdp.transition_function(state, a)
                        for next_state in dist_next_state:
                            dist_obs = self.pomdp.observation_function(next_state, a)
                            # 这里有一些问题
                            PO += b[state]*dist_next_state[next_state]*dist_obs[obs]
#                     print("PO:",PO)
#                     print("v_b_new:",v_b_new)
                    # add expectation on o value
                    value_future += PO*v_b_new


                Value = (rw_exp + self.gamma*value_future)
                # print("***")
                # print("depth:", k, " action:", a)
                # print("value ", Value, " reward", rw_exp, " future", value_future)
                
                if Value > Vmax:
                    #print("V_max is ", Value)
                    a_max = a
                    #print("action optimal",a_max)
                    Vmax = Value
                    # if policy == 0:
#                     if a_max not in policy[k]:
#                         policy[k].append([a_max,Vmax])
#                 if (Value < Vmax) and (Value >=Vmax2) and (a != policy[k][0]):
#                     policy[k][1]=a_max
    ###----------------------------ADD SPACED REPETITION----------------------------###
        if k==(self.horizon)-1:
            if a_max not in self.policy:
                self.policy.append(a_max)

        if a_max not in self.action_previous:
            self.action_previous.append(a_max)
#*********************************************
#         print("----")
#         print("depth:", k, " action:", a)
#         print("V_max is ", Vmax)
#         print("action optimal", a_max)
#*********************************************        
        with open('1_result.txt', 'a') as file:
            file.write("Optimal for horizon"+str(k)+":\n")
            file.write("*********action optimal*********%s\n" %a_max)
            #file.write(json.dumps(self.result))
            #file.write("\n")
        file.close()
        return Vmax
        
        
        
S = list(resourceIndex.values())
s_knowledge = []
knowledge_level = [0,1,2]
for k_s in (S):
    for j in knowledge_level:
        s_knowledge.append(( str(k_s) + 'level' + str(j)))

    

actions = []
for index_i,i in enumerate(M):
    for index_j,j in enumerate(i):
        #给这里改了，现在是没有重复数据的，比如1to1这种。要改成有重复的
        if (M[index_i][index_j] != 0)& (index_i!=index_j): #:
            actions.append(str(index_i)+"to"+str(index_j))
            
O = {} # Observation space
for s in s_knowledge:
    for action in actions:
        s_c= s.split('level')[0]
        if s_c == action.split('to')[1]:
            s_n = action.split('to')[1]
            for j in range(3):
                    O[("observation " + s_n +'level'+str(j))] = 1/3
                    
print("\train running time %s second\n" %(time.time()-start_time))

target = 1828007622   
print("target user id is:",target)
#target = input("please input user")

start_state = train_10[train_10.user_id == 1828007622]#.loc[0]["content_id"]
print(start_state.head())


learning_path = []
target_path = []
for u in user:
    l_path = list(train_10[train_10.user_id==u].content_id)
    l_path = [resourceIndex[x] for x in l_path]
    #l_path.append(list(exam[exam.userId==user].score)[0]-average_score)
    learning_path.append(l_path)
    if len(l_path)>5:
        data_path.append(l_path)
    if target == user:

        target_path = l_path

horizons = 5
policy = ["to"+str(start_state)]
pomdp_student = Student_problem(s_knowledge, actions, O, M,resourceIndex,train_10,userid=target)
#     b0 = give_intial_belief(arg)
b0 = {}
start_time = time.time()
b0[str(start_state)+"level0"]=1/3
b0[str(start_state)+"level1"]=1/3
b0[str(start_state)+"level1"]=1/3
#     for s in s_knowledge:
#         if s.split("level")=='0':
#             b0[s]=0.7
#         else:    
#             b0[s] =0.3/(len(s_knowledge))
# lanuch the alog for several horizons

values = np.zeros([horizons])
for i in range(horizons):
    if i==0:
        action_previous = ['to'+str(target_path[0])]
        solve_pomdp = Solver(pomdp_student, i,action_previous,{},policy)
    else:
        action_previous = solve_pomdp.get_actions()
        solve_pomdp = Solver(pomdp_student, i,action_previous,{},policy)
    values[i] = solve_pomdp.recur_dyn(b0,0)
    policy = solve_pomdp.get_policy()    
    print(policy)
    print("\nrunning time %s second\n" %(time.time()-start_time))