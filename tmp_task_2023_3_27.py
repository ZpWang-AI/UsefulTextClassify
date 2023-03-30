import pandas as pd
import numpy as np

from tqdm import tqdm

from utils import *
from corpus import *


def get_dic():
    init_reply = pd.read_csv('./data/txt1.csv')
    init_reply = np.array(init_reply['Reply'])
    # print(init_reply[:10])
    init_ans = pd.read_csv('./data/result1_txt1.csv')
    init_ans = np.array(init_ans['non_answer'])
    # print(init_ans[:10])
    # dic = {}
    # for k, v in zip(init_reply, init_ans):
    #     dic[k] = v
    dic = dict(zip(init_reply, init_ans))
    print(init_ans.shape, init_reply.shape, len(dic))
    # 4159641 4159641 3292914
    return dic


target_file = './data/result2_txt1.csv'

def main():
    new_data = pd.read_csv('./data/EasyIR(QnA)(wP).csv')
    new_data = np.array(new_data)
    # id, HasReplied, Qsubj, Reply
    # 4404093

    dic = get_dic()
    no_dealt = 0
    with open(target_file, 'w')as f:
        f.write('id,non_answer\n')
        for pid, has_replied, query, reply in tqdm(new_data):
            if has_replied:
                non_answer = dic.get(reply, -1)
                if non_answer == -1:
                    no_dealt += 1
            else:
                non_answer = -1
            f.write(f'{pid},{non_answer}\n')
    print(no_dealt)


def check():
    ans = pd.read_csv(target_file)
    ans = np.array(ans)[:,1]
    print(np.sum(ans==-1), np.sum(ans==0), np.sum(ans==1))
    print(ans.shape)
    # for a, b in ans:
    #     if b not in [0,1]:
    #         print('???')
    #         break
    # else:
    #     print('well')

if __name__ == '__main__':
    # main()
    check()
    
    pass