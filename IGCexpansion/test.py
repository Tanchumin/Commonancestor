from __future__ import print_function
import numdifftools as nd
import numpy as np


class test:
    def __init__(self):
        self.index = 0


    def sum_branch_test1(self):

        step = nd.step_generators.MaxStepGenerator(base_step=[1,2])
        print(step)
        print(nd.Hessian(self.testfunction, step=step)([0,0]))


    def testfunction(self,x):
        print(x)
        self.index += 1
        print(self.index)
        return ((x[0] ** 2+x[1]**2 +x[1]*x[0]+2*x[0]))

    def print11(self):
        step_gen = nd.step_generators.BasicMaxStepGenerator(base_step=2.0, step_ratio=2,
                                                            num_steps = 4)

        for s in step_gen():
            print(s)

        print(np.log1p(np.abs(4)).clip(min=1.0))

    def test1(self):
        basic = 0.4
        step = nd.step_generators.MaxStepGenerator(base_step=basic)

        H = nd.Hessian(self.testfunction, step=step)([2,2])
        print(H)


    def test3(self):
        dic={"d":1,"dd":2,"ddd":4}
        print(sum(list(dic.values())[1:]))





if __name__ == '__main__':
    dd=test()
    dd.test3()





