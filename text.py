import time
import numpy as np

dis = [[1, 2, 3], [4, 5, 6]]
dic = {}

name = 'agent_1'

if __name__ == '__main__':
    dic['%s/q_loss' % name] = 23
    w1 = 0
    w2 = 0
    ww = np.logaddexp(w1, w2)
    print(ww)
    print(np.exp(ww))
    print(dis)
    print(dis[1][0])
