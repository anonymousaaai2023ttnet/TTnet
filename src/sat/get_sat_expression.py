import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sympy import symbols, SOPform, POSform, simplify_logic

from src.utils.utils import get_reference1_2_3


class Sat_expression:

    def __init__(self, args, model, path_save_exp):
        self.args = args
        self.model = model
        self.path_save_exp = path_save_exp
        self.reference1, self.reference2, self.reference3 = get_reference1_2_3()



    def iterate_over_block_real(self):
        with torch.no_grad():
            for layer, (name, module) in enumerate(self.model._modules.items()):
                if name == "layer":
                    for layer2, (name2, module2) in enumerate(module._modules.items()):
                        self.blockici = layer2
                        print()
                        print("Block ", layer2)
                        print()
                        if layer2==0:
                            self.x_input_f2, self.df = self.generate_all_input(
                                self.args.channels[layer2] * self.args.kernelsizes[layer2]**2, self.model.nchannel-1)
                        else:
                            self.x_input_f2, self.df = self.generate_all_input(
                                self.args.channels[layer2] * self.args.kernelsizes[layer2]**2,
                                module2.output_channel-1)

                        res_numpy = self.get_res_numpy(module2)
                        self.iterate_over_filter(res_numpy)





    def iterate_over_filter(self, res_numpy):
        nkici = self.args.channels[self.blockici] * self.args.kernelsizes[self.blockici]**2
        for filterici in tqdm(range(0, self.args.filters[self.blockici])):
            self.filterici = filterici
            if self.args.filter_occurence == filterici:
                resici = res_numpy[:, filterici]
                unique = np.unique(resici)
                if len(unique)==1:
                    exp_DNF, exp_CNF = "True", "True"
                    coef = unique[0]
                    exp_CNF3 = str(coef)
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
                        table = self.for_1_filter_table(nkici, exp_DNF, unq2, exp_CNF)
                        np.save(self.path_save_exp + 'table_outputblock_' +
                                str(self.blockici) + '_filter_' + str(self.filterici) +
                                '_value_' + str(unq2) + '_coefdefault_' +
                                str(coef_default) + '.npy', table)

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
            exp_DNFici = simplify_logic(exp_DNF.subs(evaluate),  form='dnf')
            if str(exp_DNFici) == "False":
                table[enter][:] = "0"
            elif str(exp_DNFici) == "True":
                table[enter][:] = str(unq2)
            else:
                exp_CNFici = simplify_logic(exp_CNF.subs(evaluate),  form='cnf')
                exp_CNF3ici = self.get_exp_with_y(exp_DNFici, exp_CNFici)
                exp_CNF3ici2 = simplify_logic(exp_CNF3ici, form='cnf')
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
        if self.args.sympy:
            exp_DNF, exp_CNF = self.get_expresion_methode1(condtion_filter)
        else:
            exp_DNF, exp_CNF = self.get_expresion_methode2(condtion_filter, condtion_filter_cnf)
        exp_CNF3str = self.get_exp_with_y(exp_DNF, exp_CNF)
        exp_CNF3 = simplify_logic(exp_CNF3str, form='cnf')
        return exp_DNF, exp_CNF, exp_CNF3




    def generate_all_input(self, n, c_a_ajouter=None):
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
            x_input_f2 = 2 * x_input_f2 - 1
        elif n == 4:
            x_input_f2 = torch.Tensor(l).reshape(2 ** n, 1, 2,
                                                 2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
            if c_a_ajouter is not None:
                y = x_input_f2.detach().clone()
                padding = torch.autograd.Variable(y)
                for itera in range(c_a_ajouter):
                    x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                del padding
            x_input_f2 = 2 * x_input_f2 - 1

        elif n == 12:
            x_input_f2 = torch.Tensor(l).reshape(2 ** n, 3, 2,
                                                 2)  # .type(torch.ByteTensor)  # .unsqueeze(0).unsqueeze(0)
            if c_a_ajouter is not None:
                y = x_input_f2.detach().clone()
                padding = torch.autograd.Variable(y)
                for itera in range(c_a_ajouter):
                    x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
                del padding
            x_input_f2 = 2 * x_input_f2 - 1

        return x_input_f2, df

    def get_res_numpy(self, module2):
        _ = module2(self.x_input_f2)
        if not module2.last:
            res = (module2.outputblock + 1) / 2
        else:
            res = module2.outputblock
        res_numpy = res.squeeze(-1).squeeze(-1).numpy()
        return res_numpy

    def get_expresion_methode1(self, condtion_filter):
        if self.blockici == 0:
            w1, x1, y1, v1 = symbols('x_0, x_1, x_2, x_3')
            exp_DNF = SOPform([w1, x1, y1, v1], minterms=condtion_filter)
            exp_CNF = POSform([w1, x1, y1, v1], minterms=condtion_filter)
        else:
            w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
            exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter)
        #exp_DNF, exp_CNF = str(exp_DNF), str(exp_CNF)
        return exp_DNF, exp_CNF

    def get_expresion_methode2(self, condtion_filter, condtion_filter_cnf):
        condtion_filterbin = [list(str("{0:" + str(16) + "b}").format(x)) for x in condtion_filter]
        condtion_filterbin = [x for x in condtion_filterbin]
        condtion_filterbin_letter = ["(" + " & ".join(
                ["~x_" + str(ix) if i == ' ' or i == '0' else "x_" + str(ix) for ix, i in enumerate(x)]) + ")" for x in
                                         condtion_filterbin]
        exp_DNF = " | ".join(condtion_filterbin_letter)
        condtion_filterbincnf = [list(str("{0:" + str(16) + "b}").format(x)) for x in condtion_filter_cnf]
        condtion_filterbincnf = [x for x in condtion_filterbincnf]
        condtion_filterbin_lettercnf = ["(" + " | ".join(
                ["~x_" + str(ix) if i == ' ' or i == '0' else "x_" + str(ix) for ix, i in enumerate(x)]) + ")" for x in
                                            condtion_filterbincnf]
        exp_CNF = " & ".join(condtion_filterbin_lettercnf)

        return exp_DNF.replace(" ", ""), exp_CNF.replace(" ", "")

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

