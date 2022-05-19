import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sympy import symbols, SOPform, POSform, simplify_logic
from src.utils.utils import get_reference1_2_3


with torch.no_grad():
    class Sat_expression_v2:

        def __init__(self, args, model, path_save_exp, flag_all = True, flag_filter= True):
            self.args = args
            self.model = model
            self.path_save_exp = path_save_exp
            self.reference1, self.reference2, self.reference3 = get_reference1_2_3()
            self.flag_all = flag_all
            self.flag_filter = flag_filter

        def iterate_over_block_real(self):
            with torch.no_grad():
                for layer, (name, module) in enumerate(self.model._modules.items()):
                    if name == "layer":
                        for layer2, (name2, module2) in enumerate(module._modules.items()):
                            self.blockici = layer2
                            print()
                            print("Block ", layer2)
                            print()
                            if layer2 == 0:
                                channelinterest = int(self.args.nchannel / self.args.groups[0])
                                self.x_input_f2, self.df = self.generate_all_inputv2(0)

                            else:
                                channelinterest = int(self.args.filters[layer2-1] / self.args.groups[layer2])
                                self.x_input_f2, self.df = self.generate_all_inputv2(layer2)

                            res_numpy = self.get_res_numpy(module2)
                            self.iterate_over_filter(res_numpy, channelinterest)

        def get_res_numpy(self, module2):
            self.x_input_f2 = 2 * self.x_input_f2 - 1
            _ = module2(self.x_input_f2)
            if not module2.last:
                res = (module2.outputblock + 1) / 2
            else:
                res = module2.outputblock
            res_numpy = res.squeeze(-1).squeeze(-1).numpy()
            return res_numpy

        def iterate_over_block_general(self):
            with torch.no_grad():
                for blockici in range(len(self.args.filters)):
                    for layer, module in enumerate(self.model.features):
                        if layer == 6*blockici+3:
                            self.blockici = blockici
                            if blockici ==0:
                                channelinterest = int(self.args.nchannel / self.args.groups[blockici])
                            else:
                                channelinterest = int(self.args.filters[blockici-1] / self.args.groups[blockici])
                            self.x_input_f2, self.df = self.generate_all_inputv2(blockici)
                            resb = module(self.x_input_f2)
                        elif layer == 6*blockici+4 or layer == 6*blockici+5 or layer == 6*blockici+6 or layer == 6*blockici+7:
                            resb = module(resb)
                        elif layer == 6*blockici+8:
                            res_numpy = module(resb)
                            res_numpy = res_numpy.squeeze(-1).squeeze(-1).numpy()

                            self.iterate_over_filter(res_numpy, channelinterest)
                            del res_numpy




        def iterate_over_filter(self, res_numpy, channelinterest):
            nkici = channelinterest * self.args.kernelsizes[self.blockici]**2
            for filterici in tqdm(range(0, self.args.filters[self.blockici])):
                self.filterici = filterici
                if self.args.filter_occurence == filterici:
                    resici = res_numpy[:, filterici]
                    unique = np.unique(resici)
                    if len(unique)==1:
                        exp_DNF, exp_CNF = "True", "True"
                        coef = unique[0]
                        exp_CNF3 = str(coef)
                        #print(exp_CNF3)
                        with open(self.path_save_exp + 'table_outputblock_' +
                                  str(self.blockici) + '_filter_' + str(self.filterici) +
                                  '_coefdefault_' +
                                  str(coef) + ".txt", 'w') as f:
                            f.write(str(exp_CNF3))
                        if self.flag_all:
                            table = np.chararray((3 ** nkici, 2 ** nkici), itemsize=10)
                            table[:][:] = str(coef)
                            np.save(self.path_save_exp + 'table_outputblock_' +
                                str(self.blockici) + '_filter_' + str(self.filterici) +
                                '_value_' + str(coef) + '_coefdefault_' +
                                str(coef) + '.npy', table)

                    else:
                        coef_default = unique[0]
                        unique2 = unique[1:]
                        for unq2 in unique2:
                            exp_DNF, exp_CNF, exp_CNF3 = self.for_1_filter(unq2, resici)

                            if self.flag_filter:
                                exp_DNF, exp_CNF = self.filter_exp(exp_DNF, exp_CNF)

                            #print(exp_CNF3)
                            if self.flag_all:
                                table = self.for_1_filter_table(nkici, exp_DNF, unq2, exp_CNF)
                                np.save(self.path_save_exp + 'table_outputblock_' +
                                    str(self.blockici) + '_filter_' + str(self.filterici) +
                                    '_value_' + str(unq2) + '_coefdefault_' +
                                    str(coef_default) + '.npy', table)
                            with open(self.path_save_exp + 'table_outputblock_' +
                                            str(self.blockici) + '_filter_' + str(self.filterici) +
                                              '_coefdefault_' +
                                            str(unq2) +".txt", 'w') as f:
                                f.write(str(exp_CNF3))


        def for_1_filter_table(self, nkici, exp_DNF, unq2, exp_CNF):
            table = np.chararray((3 ** nkici, 2 ** nkici), itemsize=nkici + 1)
            table[:][:] = "-1"
            for enter in tqdm(range(3 ** nkici)):
                value_input = np.base_repr(enter, base=3)
                value_input2 = (nkici - len(value_input)) * '0' + value_input
                evaluate = {}
                for value_input2iciindex, value_input2ici in enumerate(value_input2):
                    if value_input2ici != "2":
                        evaluate["x_" + str(value_input2iciindex)] = bool(int(value_input2ici))

                exp_DNFici = simplify_logic(exp_DNF.subs(evaluate), form='dnf')
                if str(exp_DNFici) == "False":
                    table[enter][:] = "0"
                elif str(exp_DNFici) == "True":
                    table[enter][:] = str(unq2)
                else:
                    exp_CNFici = simplify_logic(exp_CNF.subs(evaluate), form='cnf')
                    exp_CNF3ici = self.get_exp_with_y(exp_DNFici, exp_CNFici)
                    exp_CNF3ici2 = simplify_logic(exp_CNF3ici, form='cnf')

                    if str(exp_CNF3ici2) == "y":
                        table[enter][:] = str(unq2)
                    elif str(exp_CNF3ici2) == "~y":
                        table[enter][:] = "0"
                    else:
                        exp_CNF3strlist = str(exp_CNF3ici2).replace(" ", "").split("&")
                        for cnficiindex, cnfici in enumerate(exp_CNF3strlist):
                            cnfici = cnfici.replace("(", "").replace(")", "").split("|")
                            strcnfici = ""
                            for cnficiici in cnfici:
                                strcnfici += self.reference3[cnficiici]
                            table[enter][cnficiindex] = str(strcnfici)
            return table



        def for_1_filter(self, unq2, resici):
            answer = resici == unq2
            dfres = pd.DataFrame(answer)
            dfres.columns = ["Filter_" + str(self.filterici) + "_Value_" + str(int(unq2))]
            df2 = pd.concat([self.df, dfres], axis=1)
            condtion_filter = df2["index"].values[answer].tolist()
            answer_cnf = (1.0 * answer) == 0.
            condtion_filter_cnf = df2["index"].values[answer_cnf].tolist()
            #if self.args.sympy:
            exp_DNF, exp_CNF = self.get_expresion_methode1(condtion_filter)
            exp_CNF3 = self.get_exp_with_y(exp_DNF, exp_CNF)
            #exp_CNF3 = simplify_logic(exp_CNF3str, form='cnf')
            return exp_DNF, exp_CNF, exp_CNF3




        def generate_all_input(self, n, c_a_ajouter=None):
            print(n)
            self.n = n
            l = [[int(y) for y in format(x, 'b').zfill(n)] for x in range(2 ** n)]
            df = pd.DataFrame(l)
            df = df.reset_index()
            x_input_f2 = None
            if n == 9:
                x_input_f2 = torch.Tensor(l).reshape(2 ** n, 1, 3, 3)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2
            elif n == 4:
                x_input_f2 = torch.Tensor(l).reshape(2 ** n, 1, 2,
                                                     2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2

            elif n == 8:
                x_input_f2 = torch.Tensor(l).reshape(2 ** n, 2, 2,
                                                     2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2

            return x_input_f2, df



        def get_expresion_methode1(self, condtion_filter):
            if self.n == 4:
                w1, x1, y1, v1 = symbols('x_0, x_1, x_2, x_3')
                exp_DNF = SOPform([w1, x1, y1, v1], minterms=condtion_filter)
                exp_CNF = POSform([w1, x1, y1, v1], minterms=condtion_filter)
            elif self.n == 8:
                w1, x1, y1, v1, w2, x2, y2, v2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7')
                exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter)
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter)
            else:
                w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
                exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
            return exp_DNF, exp_CNF



        def get_exp_with_y(self, exp_DNFstr, exp_CNFstr):
            exp_DNFstr, exp_CNFstr = str(exp_DNFstr).replace(" ", ""), str(exp_CNFstr).replace(" ", "")
            masks = exp_DNFstr.split("|")
            clausesnv = []
            for mask in masks:
                # print(mask)
                masknv = mask.replace("&", " | ")
                masknv = masknv.replace("x", "~x")
                masknv = masknv.replace("~~", "")
                masknv = masknv.replace(")", "").replace("(", "")
                masknv = "(" + masknv + ")"
                masknv = masknv.replace("(", "(y | ")
                clausesnv.append(masknv)
                # print(masknv)
            clauses = exp_CNFstr.split("&")
            for clause in clauses:
                # print(clause)
                clausenv = clause.replace("|", " | ")
                clausenv = clausenv.replace(")", "").replace("(", "")
                clausenv = "(" + clausenv + ")"
                clausenv = clausenv.replace(")", " | ~y)")
                clausesnv.append(clausenv)
            exp_CNF3 = " & ".join(clausesnv)

            return exp_CNF3

        def filter_exp(self, exp_DNF, exp_CNF):
            exp_DNFstrv2 = []
            try:

                exp_DNFstr, exp_CNFstr = str(exp_DNF).replace(" ", ""), str(exp_CNF).replace(" ", "")
                exp_DNFstrv2 = []
                thr = self.n - round(self.args.proportion*self.n)
                masks = exp_DNFstr.split("|")
                for mask in masks:
                    if mask.count("&")+1 < thr:
                        exp_DNFstrv2.append(mask)
                exp_DNFstrv3 = " | ".join(exp_DNFstrv2)

                print("1", len(exp_DNFstrv2), len(masks))
                exp_DNFvf0 = simplify_logic(exp_DNFstrv3, form='dnf')
                exp_CNFvf0 = simplify_logic(exp_DNFstrv3, form='cnf')
                print(exp_DNFvf0)


            except:
                exp_DNFvf0 = exp_DNF
                exp_CNFvf0 = exp_CNF

            try:

                if len(exp_DNFstrv2)>0:
                    exp_DNFstrv2bis = []
                    maxlen = max([x.count("&") for x in exp_DNFstrv2])
                    for mask in exp_DNFstrv2:
                        if mask.count("&") == maxlen:
                            if np.random.binomial(size=1, n=1, p= self.args.proba)[0]==0:
                                exp_DNFstrv2bis.append(mask)
                        else:
                            exp_DNFstrv2bis.append(mask)

                    exp_DNFstrv3 = " | ".join(exp_DNFstrv2bis)

                    print("2", len(exp_DNFstrv2bis), len(masks))

                exp_DNFvf = simplify_logic(exp_DNFstrv3, form='dnf')
                exp_CNFvf = simplify_logic(exp_DNFstrv3, form='cnf')


            except:
                exp_DNFvf = exp_DNFvf0
                exp_CNFvf = exp_CNFvf0

            return exp_DNFvf, exp_CNFvf


        def generate_all_inputv2(self, blocknum):

            if blocknum == 0:
                nbrefilter = self.args.nchannel
            else:
                nbrefilter = self.args.filters[blocknum-1]

            if self.args.kernelsizes[blocknum] == 3:
                self.n = 9
                c_a_ajouter = nbrefilter - 1
            elif self.args.kernelsizes[blocknum] == 2 and nbrefilter == 2 * self.args.groups[blocknum]:
                self.n = 8
                c_a_ajouter = int((nbrefilter - 2)/2)
            elif self.args.kernelsizes[blocknum] == 2 and nbrefilter==  self.args.groups[blocknum]:
                self.n = 4
                c_a_ajouter = nbrefilter - 1
            else:
                raise "PB"

            #print(self.n, c_a_ajouter)



            l = [[int(y) for y in format(x, 'b').zfill(self.n)] for x in range(2 ** self.n)]
            df = pd.DataFrame(l)
            df = df.reset_index()
            x_input_f2 = None
            if self.n == 9:
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, 1, 3, 3)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2
            elif self.n == 4:
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, 1, 2,
                                                     2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2

            elif self.n == 8:
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, 2, 2,
                                                     2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
                #print(x_input_f2.shape, c_a_ajouter)
                if c_a_ajouter is not None:
                    y = x_input_f2.detach().clone()
                    padding = torch.autograd.Variable(y)
                    for itera in range(c_a_ajouter):
                        x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                    del padding
                x_input_f2 = x_input_f2

            return x_input_f2, df

