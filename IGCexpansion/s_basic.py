import numpy as np
import scipy as sc
from numpy import random
import pandas as pd


# di is length of Q matrix
# whether do mc computation

# GLS is function to simulate history on each colunm

def GLS(Q,t,ini,end,di,length):

# Q_iiii is diagnal of Q
# max is biggest change in a line
    Q_iiii=np.ones((di))
    Q_O=Q
    max_number = 10
    time_matrix = 100*np.ones(shape=(length, max_number))
    state_matrix = np.zeros(shape=(length, max_number))


    for ii in range(di):

        Q_iiii[ii] = -Q[ii, ii]


    for d in range(di):
        Q[d,d]=0
        Q[d] = Q[d]/Q_iiii[d]

# how many change in history
# big_number how to trunct the matrix
    effect_number=0
    big_number=0

    for ll in range(length):
        # most transfer 5 times
        curent_state=ini[ll]
        i=0
        time = [0]
        state = [0]

        while(curent_state!=end[ll]):
            u=0
            i=0
            time = [100]
            state = [0]
            while(u<=t):
                # state may from 0 or 1
                i=i+1
                u1=random.exponential(Q_iiii[curent_state,])
                u=u+u1
                time.append(u)
                a = np.random.choice(range(di), 1, p=Q[curent_state,])[0]
                curent_state=a
                state.append(curent_state)

            curent_state=state[i-1]
        print(state[0:i])



        if(i>max_number):
            big_number=i
            time_matrix_old=time_matrix
            state_matrix_old=state_matrix
            time_matrix = np.zeros(shape=(length, i))
            state_matrix = np.zeros(shape=(length, i))
            time_matrix[0:length,0:max_number]=time_matrix_old
            state_matrix[0:length,0:max_number]=state_matrix_old
            time_matrix[ll,0:i]=time[0:i]
            state_matrix[ll,0:i]=state[0:i]

            max_number=big_number
        else:
            big_number=max(big_number,i)
            if(i>0):
                print(i)
                effect_number = (i-1)+effect_number
            time_matrix[ll,0:i]=time[0:i]
            state_matrix[ll,0:i]=state[0:i]
            state_matrix[ll,0]=ini[ll]
    time_matrix=time_matrix[0:length,0:big_number]
    state_matrix=state_matrix[0:length,0:big_number]

    return time_matrix,state_matrix,effect_number,big_number

# GLS_m is unsed for MC
#repeat is about how many times mc needed
def GLS_m(Q,t,ini,end,di,length,repeat=15):
    Q_iiii=np.ones((di))
    Q_O=Q
    max_number = 10

    time_list = []
    state_list =[]
    dis_matrix = np.ones(shape=(length))

    for i in range(length):
        if(ini[i]!=end[i]):

            time_list.append(0)
            state_list.append(0)
            dis_matrix[i]=0
        else:
            time_list.append(1)
            state_list.append(1)
    effect_matrix = np.zeros(shape=(length,repeat))
    big_number_matrix=np.zeros(shape=(length,repeat))



    for ii in range(length):
        if(time_list[ii]!=1):

            time_matrix = 100*np.ones(shape=(repeat, max_number))
            state_matrix = np.zeros(shape=(repeat, max_number))

            for jj in range(repeat):
                    # most transfer 5 times
                    curent_state = ini[ii]
                    i = 0
                    time = [0]
                    state = [0]


                    effect_number = 0
                    big_number = 0

                    while (curent_state != end[ii]):
                        u = 0
                        i = 0
                        time = [100]
                        state = [0]
                        while (u <= t):
                            # state may from 0 or 1
                            i = i + 1
                            u1 = random.exponential(Q_iiii[curent_state,])
                            u = u + u1
                            time.append(u)
                            a = np.random.choice(range(di), 1, p=Q[curent_state,])[0]
                            curent_state = a
                            state.append(curent_state)

                        curent_state = state[i - 1]

                    if (i > max_number):
                        big_number = i
                        time_matrix_old = time_matrix
                        state_matrix_old = state_matrix
                        time_matrix = np.zeros(shape=(length, i))
                        state_matrix = np.zeros(shape=(length, i))
                        time_matrix[0:length, 0:max_number] = time_matrix_old
                        state_matrix[0:length, 0:max_number] = state_matrix_old
                        time_matrix[jj, 0:i] = time[0:i]
                        state_matrix[jj, 0:i] = state[0:i]
                    else:
                        big_number = max(big_number, i)
                        if (i > 0):
                            effect_number = (i - 1) + effect_number
                        time_matrix[jj, 0:i] = time[0:i]
                        state_matrix[jj, 0:i] = state[0:i]
                        state_matrix[jj, 0] = ini[ii]
                    effect_matrix[ii,jj]=int(effect_number)
                    big_number_matrix[ii,jj]=int(big_number)
            time_list[ii]=time_matrix
            state_list[ii]=state_matrix

    return time_list,state_list,effect_matrix,big_number_matrix,dis_matrix

# making one sample for mc, use results from GLS_M

def GLS_s(time_list,state_list,effect_matrix,big_number_matrix, dis_matrix ,length,repeat):
    max_number = 10
    time_matrix = 100*np.ones(shape=(length, max_number))
    state_matrix = np.zeros(shape=(length, max_number))
    effect_number=0
    big_number=0
    for i in range(length):
        a = np.random.choice(range(repeat), 1, p=(1 / float(repeat)) * np.ones(shape=(repeat)))[0]
        print (a)
        if(dis_matrix[i]!=1):
            if(big_number_matrix[i,a]<=max_number):

                time_matrix[i, 0:max_number] = time_list[i][a,]
                state_matrix[i, 0:max_number] = state_list[i][a,]
                big_number=max(big_number_matrix[i,a],big_number)
                effect_number=effect_number+effect_matrix[i,a]
            else:
                big_number=max(big_number_matrix[i,a],big_number)
                effect_number=effect_number+effect_matrix[i,a]

                time_matrix_old = time_matrix
                state_matrix_old = state_matrix
                time_matrix = np.zeros(shape=(length, big_number))
                state_matrix = np.zeros(shape=(length, big_number))
                time_matrix[0:length, 0:max_number] = time_matrix_old
                state_matrix[0:length, 0:max_number] = state_matrix_old
                time_matrix[i, 0:big_number] = time_list[i][a,]
                state_matrix[i, 0:big_number] = state_list[i][a,]

    time_matrix = time_matrix[0:length, 0:int(big_number)]
    state_matrix = state_matrix[0:length, 0:int(big_number)]

    return time_matrix, state_matrix, int(effect_number), int(big_number)









# sort time and state matrix into a long history
# di is a dictionary indicate the difference
# ini stat will be changed at each time

def rank_ts(time,state,di,ini,length,effect_number):
    difference=0
    time_new=0
    for i in range(length):
        difference = difference+di[ini[i]]

# 0 last difference number ,1 next difference number, 2 last time, 3 next time
# 4 time difference is important, 5 location ,6 last state, 7 next state,8 paralog type
    history_matrix = np.zeros(shape=(effect_number+1, 9))
    for jj in range(effect_number+1):
        coor = np.argmin(time)
        history_matrix[jj,0]=difference
        time_old=time_new
        history_matrix[jj, 2] = time_old
        time_new=np.min(time)
        history_matrix[jj, 3] = time_new
        history_matrix[jj, 4] = time_new-time_old
# track state matrix
        d = time.shape[1]
        x_aixs = coor / d
        y_aixs = coor % d
        history_matrix[jj, 5]=x_aixs
        history_matrix[jj, 6] = ini[x_aixs]
        history_matrix[jj, 7]=state[x_aixs, y_aixs]

        history_matrix[jj, 1]=difference-di[int(history_matrix[jj, 6])]+di[int(history_matrix[jj, 7])]
        difference=history_matrix[jj, 1]
        # renew time matrix and ini stat
        time[x_aixs, y_aixs]=100
        ini[x_aixs]=history_matrix[jj, 7]


    return(history_matrix)






# method can be selected, "simple"  means select by unimormly partition

def divide_Q(history_matrix,effect_number, method="simple", simple_state_number=5):
    if(method =="simple"):
        quan = 1/float(simple_state_number)
        quan_c=1 / float(simple_state_number)
        stat_rank = pd.DataFrame(history_matrix[:, 1])
        stat_vec = np.zeros(shape=(simple_state_number-1, 1))
        for i in range(simple_state_number-1):
            stat_vec[i] = np.quantile(stat_rank,quan_c)
            quan_c=quan_c+quan
        for ii in range(effect_number+1):
            if( history_matrix[ii,1] <= stat_vec[0] ):
                history_matrix[ii, 8]=0
            elif(history_matrix[ii,1]>stat_vec[simple_state_number-2]):
                history_matrix[ii, 8] = simple_state_number-1
            else:
                for iii in range(simple_state_number-1):
                    if(history_matrix[ii,1]<=stat_vec[iii+1] and history_matrix[ii,1]>stat_vec[iii]):
                        history_matrix[ii, 8] = iii+1
                        break
        type_number=simple_state_number



    return(history_matrix,type_number)


#type_number is from divide_Q, which indicate how many type of paralog
def esitimate_Q(history_matrix,di,type_number):
    q_list=[]
    for i in range(type_number):
        mtrx = np.zeros((i, 2))




    return (history_matrix)












m1=np.array([[-3.0,1.0,1.0,1.0],[1.0,-3.0,1.0,1.0],[1.0,1.0,-3.0,1.0],[1.0,1.0,1.0,-3.0]])
m2=np.array([[-3.0,1.0,1.0,1.0],[1.0,-3.0,1.0,1.0],[1.0,1.0,-3.0,1.0],[1.0,1.0,1.0,-3.0]])
Q_iiii = np.ones((4))
for ii in range(4):
    Q_iiii[ii] = -m1[ii, ii]

for d in range(4):
    m1[d, d] = 0
    m1[d] = m1[d] / Q_iiii[d]

a = np.random.choice(range(4), 1, p=m1[1,])

t=1
ini=np.array([2,1,1,2,3,3,2,1,0,0,3,2,1,0])
end=np.array([1,1,2,3,1,3,1,1,0,3,0,1,2,3])
di=4
length=14



aa=GLS(m2,t,ini,end,di,length)


print(aa)

aaa=GLS_m(m2,t,ini,end,di,length,repeat=3)

aaaa=GLS_s(time_list=aaa[0],state_list=aaa[1],effect_matrix=aaa[2],big_number_matrix=aaa[3],dis_matrix=aaa[4],
           length=length,repeat=3)


time=np.array(aaaa[0])
state=np.array(aaaa[1])
effect=np.array(aaaa[2])
dic={0:1,1:2,2:3,3:4}

print(time)
print(state)
print("eff")
print(effect)
print("history:")

zz=rank_ts(time=time,state=state,di=dic,ini=ini,length=length,effect_number=effect)
stat_rank=pd.DataFrame(zz[:,1])
a=np.quantile(stat_rank,1/3)
print(zz)
print("stat:")
print(stat_rank)
print("aaa")
print(a)

mm=divide_Q(history_matrix=zz,effect_number=effect,simple_state_number=3)
print(mm)





coor=np.argmin(time)
d=time.shape[1]
x_aixs=coor/d
y_aixs=coor%d
print(np.min(time))
print(time[x_aixs,y_aixs])

print (np.zeros(shape=(2,2)))


aaa=np.random.choice(range(4), 1, p=[0.1,0.9,0,0 ])[0]
print (aaa)

print (np.random.uniform(0,1))
