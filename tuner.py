from __future__ import print_function, division
import random
import pdb
import time

# __author__ = 'WeiFu'


class DE(object):
    """
    :parameter
    ===========
    :param learner: data minier to be used to predict
    :param paras_distribution: dictionary type, key is the name, value is a
    list showing range
    :param train_data: training data sets, panda.DataFrame type
    :param tune_data: tuning data sets, panda.DataFrame type
    :param goal: tuning goal, can be "PD, PF, F, PREC, G" ect
    :param num_population: num of population in DE
    :param repeats: num of repeats,
    :param life: early termination.
    :param f: prob of mutation a+f*(b-c)
    :param cr: prob of crossover
    """

    def __init__(self, learner, params_distribution, goal,
                 num_population=10, repeats=60, f=0.75, cr=0.3, life=3):
        self.np = num_population
        self.repeats = repeats
        self.f = f
        self.cr = cr
        self.life = life
        self.learner = learner
        self.params_distribution = params_distribution
        print(params_distribution)
        self.goal = goal
        self.evaluation = 0
        self.scores = {i:-100.0 for i in range(num_population)}
        self.frontier = [self.generate() for _ in range(self.np)]
        self.bestconf, self.bestscore = None,-100.0

    def generate(self):
        candidate = {}
        for key, val in self.params_distribution.items():
            if isinstance(val[0], float):
                candidate[key] = round(random.uniform(val[0], val[1]), 3)
            elif isinstance(val[0], bool):
                candidate[key] = random.random() <= 0.5
            elif isinstance(val[0], str):
                candidate[key] = random.choice(val)
            elif isinstance(val[0], int):
                candidate[key] = int(random.uniform(val[0], val[1]))
            elif isinstance(val[0], list) and isinstance(val[0][0], int):
                candidate[key] = [int(random.uniform(each[0], each[1])) for each in
                                  val]
            else:
                raise ValueError("type of params distribution is wrong!")
        # if "random_state" in self.params_distribution.keys():
        #   candidate["random_state"] = 1  ## set random seed here
        return candidate

    def best(self):
        sortlst = sorted(self.scores.items(), key=lambda x: x[1])
        bestconf = self.frontier[sortlst[-1][0]]
        bestscore = sortlst[-1][-1]
        return bestconf, bestscore

    def gen3(self, n, f):
        seen = [n]

        def gen1(seen):
            while 1:
                k = random.randint(0, self.np - 1)
                if k not in seen:
                    seen += [k]
                    break
            return self.frontier[k]

        a = gen1(seen)
        b = gen1(seen)
        c = gen1(seen)
        return a, b, c

    def trim(self, n, x):
        if isinstance(self.params_distribution[n][0], float):
            return max(self.params_distribution[n][0],
                       min(round(x, 2), self.params_distribution[n][1]))
        elif isinstance(self.params_distribution[n][0], int):
            return max(self.params_distribution[n][0],
                       min(int(x), self.params_distribution[n][1]))
        else:
            raise ValueError("wrong type here in parameters")

    def mutate(self, index, old):
        newf = {}
        a, b, c = self.gen3(index, old)
        for key, val in old.items():
            if isinstance(self.params_distribution[key][0], bool):
                newf[key] = old[key] if self.cr < random.random() else not old[key]
            elif isinstance(self.params_distribution[key][0], str):
                newf[key] = random.choice(self.params_distribution[key])
            elif isinstance(self.params_distribution[key][0], list):
                temp_lst = []
                for i, each in enumerate(self.params_distribution[key]):
                    temp_lst.append(old[key][i] if self.cr < random.random() else
                                    max(self.params_distribution[key][i][0],
                                        min(self.params_distribution[key][i][1],
                                            int(a[key][i] +
                                                self.f * (b[key][i] - c[key][i])))))
                newf[key] = temp_lst
            else:
                newf[key] = old[key] if self.cr < random.random() else self.trim(key, (
                    a[key] + self.f * (b[key] - c[key])))

        return newf

    def tune(self, train_X, train_Y, tune_X, tune_Y, goal):
        changed = False
        for it in range(self.repeats):
            #print(time.strftime("%Y%m%d_%H:%M:%S"), "###","Now life is: " ,self.life, it)
            if self.life <= 0:
                break
            nextgeneration = []
            for index, f in enumerate(self.frontier):
                new = self.mutate(index, f)
                self.learner.set_params(**new)
                self.learner.fit(train_X, train_Y)
                newscore = goal(tune_Y, self.learner.predict(tune_X))
                self.evaluation += 1
                if newscore > self.scores[index]:
                    nextgeneration.append(new)
                    self.scores[index] = newscore
                else:
                    nextgeneration.append(f)
            self.frontier = nextgeneration[:]
            newbestconf, newbestscore = self.best()
            if newbestscore > self.bestscore:
                self.bestscore = newbestscore
                self.bestconf = newbestconf
                changed = True  #self.life = 3 again?
            if not changed:
                self.life -= 1
            changed = False
        #print("TUNING DONE !")
        return (self.bestconf, self.evaluation)
