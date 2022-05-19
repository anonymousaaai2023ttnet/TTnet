import numpy as np
import torch
import os
import copy
#import polytope.polytope as alg
#from pypoman import compute_polytope_vertices
from pysat.formula import CNF
from pysat.solvers import Lingeling, Glucose3, Glucose4, Minisat22, Cadical,MapleChrono, MapleCM, Maplesat, Solver, Minicard, MinisatGH
import math
from functools import reduce
from tqdm import tqdm


def find_gcd(list):
    x = reduce(math.gcd, list)
    return x


from threading import Timer
import time

from eevbnn.utils import ModelHelper


class concat(object):

    def __call__(self, img):
        image = torch.cat((img, img), axis=0)
        return image


def get_reference1_2_3():
    reference1 = {}
    reference2 = {}
    reference3 = {}
    for kkkk in range(1,18):
        reference1[str(np.base_repr(kkkk,35))] = "x_"+str(kkkk-1)
        reference2[str(np.base_repr(kkkk,35))] = (kkkk-1, 1)
        reference3["x_" + str(kkkk - 1)] = np.base_repr(kkkk, 35)
        if kkkk == 17:
            reference1[str(np.base_repr(kkkk,35))] = "y"
            reference2[str(np.base_repr(kkkk,35))] = (-1, 1)
            reference3["y"] = np.base_repr(kkkk, 35)
    for kkkk in range(18,35):
        reference1[str(np.base_repr(kkkk,35))] = "~x_" + str(kkkk-18)
        reference2[str(np.base_repr(kkkk,35))] = (kkkk-18, -1)
        reference3["~x_" + str(kkkk - 18)] = np.base_repr(kkkk, 35)
        if kkkk == 34:
            reference1[str(np.base_repr(kkkk,35))] = "~y"
            reference2[str(np.base_repr(kkkk,35))] = (-1, -1)
            reference3["~y"] = np.base_repr(kkkk, 35)
    return reference1, reference2, reference3


def get_res_numpybloc(args, path, num=0):
    res_numpybloc0 = {}
    for filterici in tqdm(range(args.filters[num])):
        res_numpybloc0[filterici] = {}
        for coef in [0.0, 1.0, 2.0, 3.0]:
            pathfile0 = path + 'table_outputblock_' + str(num)+ '_filter_' + str(filterici) + '_value_' + str(coef) + '_coefdefault_' + str(0.0) + '.npy'
            if os.path.isfile(pathfile0):
                if coef==0.0:
                    res_numpybloc0[filterici][coef] = np.load(pathfile0)
                else:
                    #print(pathfile0)
                    #test = np.load(pathfile0)
                    #print(test.shape)
                    #print(res_numpybloc0[filterici])
                    res_numpybloc0[filterici][coef] = np.load(pathfile0)

            pathfile0 = path + 'table_outputblock_' + str(num) + '_filter_' + str(filterici) + '_value_' + str(
                coef) + '_coefdefault_' + str(1.0) + '.npy'
            if os.path.isfile(pathfile0):
                if coef == 1.0:
                    res_numpybloc0[filterici][1.0] = np.load(pathfile0)
                else:
                    raise "PB"
    #print(num, res_numpybloc0.keys(), res_numpybloc0[0].keys())
    return res_numpybloc0



def eval_model_general_binary(model_train, img, block_tot):
    with torch.no_grad():
        end_layer = 15 + (block_tot-2) * 6
        all_block_acomplter = {}
        all_inputbshape = {}
        for layer, module in enumerate(model_train.features):
            if layer == 0:
                res = module(img)
            else:
                res = module(res)
                if layer == 2:
                    input_binary = res.detach().cpu().clone()
                elif layer == end_layer:
                    features_ref = res.detach().cpu().clone().numpy()
                for b in range(block_tot):
                    if layer == 8 + 6*b:
                        all_block_acomplter[b] = res.detach().cpu().clone().numpy()
                        all_inputbshape[b] = res.detach().cpu().clone().numpy().shape[-1]

    return res, input_binary, all_block_acomplter, features_ref, all_inputbshape


def eval_model_general_real(model_train, img, block_tot):
    with torch.no_grad():
        all_block_acomplter = {}
        all_inputbshape = {}
        res = model_train(img)

        for layer, (name, module) in enumerate(model_train._modules.items()):
            if name == "layer":
                for layer2, (name2, module2) in enumerate(module._modules.items()):
                    if layer2!=block_tot-1:
                        all_block_acomplter[layer2] = (1.0 + module2.outputblock.detach().cpu().clone())/2
                    else:
                        all_block_acomplter[layer2] = (module2.outputblock.detach().cpu().clone())
                    all_inputbshape[layer2] = module2.outputblock.detach().cpu().clone().numpy().shape[-1]

        features_ref = model_train.featuresf.detach().cpu().clone()
        input_binary = model_train.inputnim_post_process.detach().cpu().clone()



    return res, input_binary, all_block_acomplter, features_ref, all_inputbshape


def infer_1_block_sat(args, numblock, input_binary, outputbinary, batchici, unfoldblock,
                      input_output_shape, sat_exp_table):
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        #print(input_vu_par_cnn_avant_unfold.shape)
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        assert input_vu_par_cnn_et_sat.shape[-1] == input_output_shape * input_output_shape
        for filtreici in range(nombredefiltrequivoitcegroupe):
            filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
            coef_all = list(sat_exp_table[filtreicifin].keys())
            for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
                #flag_already_verify = False
                valueint_list = [0] * 3
                for index_coeffall in range(len(coef_all)):
                    all_value = sat_exp_table[filtreicifin][coef_all[index_coeffall]]
                    une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                    i_b0 = int(patches // input_output_shape)
                    j_b0 = int(patches % input_output_shape)
                    input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
                    input_ter = input_bin.replace("-1", "2")
                    index_val_ter = int(input_ter, 3)
                    value = all_value[index_val_ter]
                    if b'-1' not in value:
                        valueint = int(value[-1][:1])
                        valueint_list[valueint-1] = valueint
                assert np.sum(valueint_list) == outputbinary[batchici, filtreicifin, i_b0, j_b0]



def infer_and_replace_1_block_sat(args, numblock, input_binary, outputbinary,
                                  batchici, unfoldblock, sat_exp_table):
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        input_output_shape = int(np.sqrt(input_vu_par_cnn_et_sat.shape[-1]))
        for filtreici in range(nombredefiltrequivoitcegroupe):
            filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
            coef_all = list(sat_exp_table[filtreicifin].keys())
            for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
                #flag_already_verify = False
                valueint_list = [0] * len(coef_all)
                #print(len(coef_all))
                for index_coeffall in range(len(coef_all)):
                    all_value = sat_exp_table[filtreicifin][coef_all[index_coeffall]]
                    une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                    i_b0 = int(patches // input_output_shape)
                    j_b0 = int(patches % input_output_shape)
                    #print(i_b0, j_b0)
                    input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
                    input_ter = input_bin.replace("-1", "2")
                    index_val_ter = int(input_ter, 3)
                    value = all_value[index_val_ter]
                    if b'-1' not in value:
                        valueint = int(value[-1][:1])
                        valueint_list[index_coeffall] = valueint
                #print(i_b0, j_b0)
                outputbinary[batchici, filtreicifin, i_b0, j_b0] = np.sum(valueint_list)

    return outputbinary


def infer_and_replace_1_block_sat_vitesse(args, numblock, input_binary, outputbinary,
                                  batchici, unfoldblock, sat_exp_table, input_binary_duplicate, outputbinary_duplicate):
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        #input_vu_par_cnn_avant_unfold_duplicate = input_binary_duplicate[:, groupici * nombredefiltredansgroupe
        #                                                : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
        #                                :, :]
        #input_vu_par_cnn_et_sat_duplicate = unfoldblock(input_vu_par_cnn_avant_unfold_duplicate)
        input_output_shape = int(np.sqrt(input_vu_par_cnn_et_sat.shape[-1]))
        for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
            une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
            #une_entree_de_sat_duplicate = input_vu_par_cnn_et_sat_duplicate[:, :, patches].detach().cpu().clone().numpy()[batchici]
            #if -1 in une_entree_de_sat_duplicate:
            i_b0 = int(patches // input_output_shape)
            j_b0 = int(patches % input_output_shape)
            input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
            input_ter = input_bin.replace("-1", "2")
            index_val_ter = int(input_ter, 3)
            for filtreici in range(nombredefiltrequivoitcegroupe):
                filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
                coef_all = list(sat_exp_table[filtreicifin].keys())
                for index_coeffall in range(len(coef_all)):
                    all_value = sat_exp_table[filtreicifin][coef_all[index_coeffall]]
                    value = all_value[index_val_ter]
                    if b'-1' not in value:
                        valueint = int(value[-1][:1])
                        outputbinary[batchici, filtreicifin, i_b0, j_b0] = valueint
                        outputbinary_duplicate[batchici, filtreicifin, i_b0, j_b0] = valueint
    return outputbinary, outputbinary_duplicate


def infer_and_replace_1_block_sat_robuste(args, numblock, input_binary, outputbinary,
                                  batchici, unfoldblock, sat_exp_table):
    #print(100*np.sum(input_binary.numpy()==-1)/(input_binary.shape[-1]*input_binary.shape[-1]*input_binary.shape[1]))
    cpt = 0
    cpt_tot = 0
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        input_output_shape = int(np.sqrt(input_vu_par_cnn_et_sat.shape[-1]))
        for filtreici in range(nombredefiltrequivoitcegroupe):
            filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
            coef_all = list(sat_exp_table[filtreicifin].keys())
            all_value = sat_exp_table[filtreicifin][coef_all[0]]
            for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
                une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                i_b0 = int(patches // input_output_shape)
                j_b0 = int(patches % input_output_shape)
                input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
                input_ter = input_bin.replace("-1", "2")
                index_val_ter = int(input_ter, 3)
                value = all_value[index_val_ter]
                if b'-1' in value:
                    outputbinary[batchici, filtreicifin, i_b0, j_b0] = np.random.randint(1)
                    cpt +=1
                cpt_tot += 1
    #print(cpt, cpt_tot)
    return outputbinary


def infer_1_block_sat_verify(args, numblock, input_binary, outputbinary, batchici, unfoldblock,
                      input_output_shape, sat_exp_table, dictionnaire_ref, outputbinary_ref, cnf_general, reference2):
    outputbinary_refv2 = copy.deepcopy(outputbinary_ref)
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        inputref_vu_par_cnn_et_sat = dictionnaire_ref[numblock][groupici]
        assert input_vu_par_cnn_et_sat.shape[-1] == input_output_shape * input_output_shape
        for filtreici in range(nombredefiltrequivoitcegroupe):
            filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
            coef_all = list(sat_exp_table[filtreicifin].keys())
            assert len(coef_all) > 0
            for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
                #flag_already_verify = False
                valueint_list = [0] * 3
                valueint_list_table = [0] * 3
                for index_coeffall in range(len(coef_all)):
                    all_value = sat_exp_table[filtreicifin][coef_all[index_coeffall]]
                    une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                    une_entreeref_de_sat = inputref_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                    i_b0 = int(patches // input_output_shape)
                    j_b0 = int(patches % input_output_shape)
                    input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
                    input_ter = input_bin.replace("-1", "2")
                    index_val_ter = int(input_ter, 3)
                    value = all_value[index_val_ter]

                    if b'-1' not in value:
                        valueint = int(value[-1][:1])
                        valueint_list[valueint-1] = valueint
                    else:
                        valueint = int(coef_all[index_coeffall])
                        valueint_list[valueint-1] = -1
                        valueint_list_table[valueint-1] = value

                if -1 not in valueint_list:
                    assert np.sum(valueint_list) == outputbinary[batchici, filtreicifin, i_b0, j_b0]
                elif -1 in valueint_list and (np.sum(np.array(valueint_list)>0)>0):
                    assert np.sum(valueint_list)+np.sum(np.array(valueint_list)==-1) == outputbinary[batchici, filtreicifin, i_b0, j_b0]
                else:
                    outputbinary[batchici, filtreicifin, i_b0, j_b0] = -1
                    for index_coeffall in range(len(valueint_list)):
                        if valueint_list[index_coeffall] ==-1:
                            une_sortieref_de_sat = outputbinary_ref[
                                batchici, filtreicifin, i_b0, j_b0, index_coeffall]
                            cnf_general = incremente_clause(valueint_list_table[index_coeffall],
                                                            reference2,
                                                            une_entreeref_de_sat,
                                                            une_sortieref_de_sat, cnf_general)
                            if outputbinary_refv2.shape[-1] == 3:
                                outputbinary_refv2[batchici, filtreicifin, i_b0, j_b0, index_coeffall] = -1



    return cnf_general, outputbinary, outputbinary_refv2



def infer_1_block_sat_verify_vitesse(args, numblock, input_binary, outputbinary, batchici, unfoldblock,
                      input_output_shape, sat_exp_table, dictionnaire_ref, outputbinary_ref, cnf_general, reference2):
    outputbinary_refv2 = copy.deepcopy(outputbinary_ref)
    if numblock == 0:
        nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
    else:
        nombredefiltredansgroupe = int(args.filters[numblock-1] / args.groups[numblock])
    nombredefiltrequivoitcegroupe = int(args.filters[numblock] / args.groups[numblock])
    for groupici in range(args.groups[numblock]):
        input_vu_par_cnn_avant_unfold = input_binary[:, groupici * nombredefiltredansgroupe
                                                        : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                        :, :]
        input_vu_par_cnn_et_sat = unfoldblock(input_vu_par_cnn_avant_unfold)
        for patches in range(input_vu_par_cnn_et_sat.shape[-1]):
            une_entree_de_sat = input_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
            if -1 in une_entree_de_sat:
                inputref_vu_par_cnn_et_sat = dictionnaire_ref[numblock][groupici]
                une_entreeref_de_sat = inputref_vu_par_cnn_et_sat[:, :, patches].detach().cpu().clone().numpy()[batchici]
                i_b0 = int(patches // input_output_shape)
                j_b0 = int(patches % input_output_shape)
                input_bin = "".join(une_entree_de_sat.astype('int').astype('str').tolist())
                input_ter = input_bin.replace("-1", "2")
                index_val_ter = int(input_ter, 3)
                for filtreici in range(nombredefiltrequivoitcegroupe):
                    filtreicifin = groupici * nombredefiltrequivoitcegroupe + filtreici
                    coef_all = list(sat_exp_table[filtreicifin].keys())
                    valueint_list = [0] * 3
                    valueint_list_table = [0] * 3
                    for index_coeffall in range(len(coef_all)):
                        value = sat_exp_table[filtreicifin][coef_all[index_coeffall]][index_val_ter]
                        une_sortieref_de_sat = outputbinary_ref[batchici, filtreicifin, i_b0, j_b0]

                        if b'-1' not in value:
                            valueint = int(value[-1][:1])
                            valueint_list[valueint - 1] = valueint
                        else:
                            valueint = int(coef_all[index_coeffall])
                            valueint_list[valueint - 1] = -1
                            valueint_list_table[valueint - 1] = value

                    if -1 in valueint_list and (np.sum(np.array(valueint_list)>0)==0):
                        outputbinary[batchici, filtreicifin, i_b0, j_b0] = -1
                        flagcontinue = True

                        if numblock == 0:
                            if np.random.randint(0,101)<args.ratio_distrib_maxsat:
                                outputbinary[batchici, filtreicifin, i_b0, j_b0] = np.random.randint(0,2)
                                flagcontinue = False

                        if flagcontinue:

                            for index_coeffall in range(len(valueint_list)):
                                if valueint_list[index_coeffall] == -1:
                                    une_sortieref_de_sat = outputbinary_ref[
                                        batchici, filtreicifin, i_b0, j_b0, index_coeffall]
                                    cnf_general = incremente_clause(valueint_list_table[index_coeffall],
                                                                    reference2,
                                                                    une_entreeref_de_sat,
                                                                    une_sortieref_de_sat, cnf_general)
                                    if outputbinary_refv2.shape[-1] == 3:
                                        outputbinary_refv2[batchici, filtreicifin, i_b0, j_b0, index_coeffall] = -1

    return cnf_general, outputbinary, outputbinary_refv2

def get_input_noise(images, args, quantized_model_train, batchici):
    imagesPLUS = images.clone() + float(args.attack_eps_ici)
    imagesMOINS = images.clone() - float(args.attack_eps_ici)
    if args.type == "binary":
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(imagesPLUS)
            elif layer == 1 or layer == 2:
                res = module(res)
        outp = res.detach().cpu().clone()
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(imagesMOINS)
            elif layer == 1 or layer == 2:
                res = module(res)
        outm = res.detach().cpu().clone()
    elif args.type == "real":
        _ = quantized_model_train(imagesPLUS)
        outp = quantized_model_train.inputnim_post_process.detach().cpu().clone()
        _ = quantized_model_train(imagesMOINS)
        outm = quantized_model_train.inputnim_post_process.detach().cpu().clone()




    input_acomplter = (outm * (outm == outp).float() + -1 * (outm != outp))




    return input_acomplter, imagesPLUS, imagesMOINS


def get_input_noise_maxsat(images, args, quantized_model_train, batchici):
    imagesPLUS = images.clone() + float(args.attack_eps_ici)
    imagesMOINS = images.clone() - float(args.attack_eps_ici)
    images2PLUS = images.clone() + 2*float(args.attack_eps_ici)
    images2MOINS = images.clone() - 2*float(args.attack_eps_ici)
    if args.type == "binary":
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(imagesPLUS)
            elif layer == 1 or layer == 2:
                res = module(res)
        outp = res.detach().cpu().clone()
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(imagesMOINS)
            elif layer == 1 or layer == 2:
                res = module(res)
        outm = res.detach().cpu().clone()
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(images2PLUS)
            elif layer == 1 or layer == 2:
                res = module(res)
        out2p = res.detach().cpu().clone()
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(images2MOINS)
            elif layer == 1 or layer == 2:
                res = module(res)
        out2m = res.detach().cpu().clone()
        for layer, module in enumerate(quantized_model_train.features):
            if layer == 0:
                res = module(images)
            elif layer == 1 or layer == 2:
                res = module(res)
        outnorm = res.detach().cpu().clone()
    elif args.type == "real":
        _ = quantized_model_train(imagesPLUS)
        outp = quantized_model_train.inputnim_post_process.detach().cpu().clone()
        _ = quantized_model_train(imagesMOINS)
        outm = quantized_model_train.inputnim_post_process.detach().cpu().clone()



    input_acomplter = (outm * (outm == outp).float() + -1 * (outm != outp))
    input_acomplter2m = (out2m * (out2m == outnorm).float() + -1 * (out2m != outnorm))
    input_acomplter2p = (outnorm * (outnorm == out2p).float() + -1 * (outnorm != out2p))

    moins1maskv0 = (input_acomplter==-1).float() + (input_acomplter2m==-1).float() + (input_acomplter2p==-1).float()
    moins1maskv1a = (moins1maskv0>0).float()
    moins1maskv1b = (moins1maskv0 <= 0).float()

    input_acompltervf = moins1maskv1a*-1 + moins1maskv1b*images.clone().cpu()

    return input_acompltervf, imagesPLUS, imagesMOINS

def get_input_noise2(images, quantized_model_train, noise):
    imagesPLUS = images.clone() + float(noise)
    for layer, module in enumerate(quantized_model_train.features):
        if layer == 0:
            res = module(imagesPLUS)
        elif layer == 1 or layer == 2:
            res = module(res)
    outp = res.detach().cpu().clone()

    imagesMOINS = images.clone() - float(noise)
    for layer, module in enumerate(quantized_model_train.features):
        if layer == 0:
            res = module(imagesMOINS)
        elif layer == 1 or layer == 2:
            res = module(res)
    outm = res.detach().cpu().clone()

    input_acomplter = (outm * (outm == outp).float() + -1 * (outm != outp))

    input_acomplterv2 = 1.0 * (input_acomplter == -1)

    return input_acomplterv2, imagesPLUS, imagesMOINS


def get_refs_all(args, img_refsizex, all_inputbshape):
    if args.dataset == "CIFAR10":
        img_ref = np.zeros((args.nchannel, img_refsizex, img_refsizex))
    else:
        img_ref = np.zeros((1, img_refsizex, img_refsizex))
    block_refcall = {}
    for numblockici in range(len(args.filters)):
        if numblockici != len(args.filters)-1:
            block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici], 1))
        else:
            if args.type == "binary":
                block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici], 1))
            else:
                block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici], 3))

    img_ref_inverse = {}
    cpt = 0
    for k in range(img_ref.shape[0]):
        for i in range(img_ref.shape[1]):
            for j in range(img_ref.shape[2]):
                cpt += 1
                img_ref[k][i][j] = cpt
                img_ref_inverse[cpt] = [k, i, j]

    cpttot = 0
    for numblockici in range(len(args.filters)):
        if numblockici != len(args.filters)-1:
            for k in range(block_refcall[numblockici].shape[0]):
                for i in range(block_refcall[numblockici].shape[1]):
                    for j in range(block_refcall[numblockici].shape[2]):
                        for v in range(block_refcall[numblockici].shape[3]):
                            cpttot += 1
                            block_refcall[numblockici][k][i][j][v] = cpttot + cpt
        else:
            for v in range(block_refcall[numblockici].shape[0]):
                for k in range(block_refcall[numblockici].shape[1]):
                    for i in range(block_refcall[numblockici].shape[2]):
                        for j in range(block_refcall[numblockici].shape[3]):
                            cpttot += 1
                            block_refcall[numblockici][v][k][i][j] = cpttot + cpt

    img_ref = torch.Tensor(img_ref).unsqueeze(0)
    for numblockici in range(len(args.filters)):
        block_refcall[numblockici] = torch.Tensor(block_refcall[numblockici]).unsqueeze(0)



    return img_ref, block_refcall, img_ref_inverse



def get_refs_all_maxsat(args, img_refsizex, all_inputbshape):
    if args.dataset == "CIFAR10":
        img_ref = np.zeros((args.nchannel, img_refsizex, img_refsizex))
    else:
        img_ref = np.zeros((1, img_refsizex, img_refsizex))
    block_refcall = {}
    for numblockici in range(len(args.filters)):
        if numblockici != len(args.filters)-1:
            block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici]))
        else:
            if args.type == "binary":
                block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici], 1))
            else:
                block_refcall[numblockici] = np.zeros((args.filters[numblockici], all_inputbshape[numblockici], all_inputbshape[numblockici], 3))


    cpt = 0
    for k in range(img_ref.shape[0]):
        for i in range(img_ref.shape[1]):
            for j in range(img_ref.shape[2]):
                #cpt += 1
                img_ref[k][i][j] = 0

    cpttot = 0
    for numblockici in range(len(args.filters)):
        if numblockici != len(args.filters)-1:
            for k in range(block_refcall[numblockici].shape[0]):
                for i in range(block_refcall[numblockici].shape[1]):
                    for j in range(block_refcall[numblockici].shape[2]):
                        cpttot += 1
                        block_refcall[numblockici][k][i][j] = cpttot
        else:
            for v in range(block_refcall[numblockici].shape[0]):
                for k in range(block_refcall[numblockici].shape[1]):
                    for i in range(block_refcall[numblockici].shape[2]):
                        for j in range(block_refcall[numblockici].shape[3]):
                            cpttot += 1
                            block_refcall[numblockici][v][k][i][j] = cpttot

    img_ref = torch.Tensor(img_ref).unsqueeze(0)
    for numblockici in range(len(args.filters)):
        block_refcall[numblockici] = torch.Tensor(block_refcall[numblockici]).unsqueeze(0)



    return img_ref, block_refcall

def incremente_clause(value, reference2, une_entreeref_de_sat, une_sortieref_de_sat, cnf_general):
    indexclausewhile = 0
    clause = value[indexclausewhile]
    while clause != b'-1':
        cnfici = []
        literalall = str(clause)[2:-1]
        for literal in literalall:
            valuetoadd0 = reference2[literal]
            if valuetoadd0[0] >= 0:
                valuetoadd = valuetoadd0[1] * une_entreeref_de_sat[valuetoadd0[0]]
            else:
                valuetoadd = valuetoadd0[1] * une_sortieref_de_sat
            cnfici.append(int(valuetoadd))
        cnf_general.append(cnfici)
        indexclausewhile += 1
        clause = value[indexclausewhile]
    return cnf_general

def interrupt(s):
    s.interrupt()

def solve_cnf(args, cnf_general, indexicicici):
    if args.sat_solver == "Minicard":
        l = Minicard()
    elif args.sat_solver == "Glucose3":
        l = Glucose3()
    elif args.sat_solver == "Glucose4":
        l = Glucose4()
    elif args.sat_solver == "Minisat22":
        l = Minisat22()
    elif args.sat_solver == "Lingeling":
        l = Lingeling()
    elif args.sat_solver == "CaDiCaL":
        l = Cadical()
    elif args.sat_solver == "MapleChrono":
        l = MapleChrono()
    elif args.sat_solver == "MapleCM":
        l = MapleCM()
    elif args.sat_solver == "Maplesat":
        l = Maplesat()
    elif args.sat_solver == "Mergesat3":
        l = Solver("mergesat3")
    elif args.sat_solver == "MinisatGH":
        l = MinisatGH()



    else:
        raise "PB"
    #print(cnf_general)
    l.append_formula(cnf_general)
    timer = Timer(args.time_out, interrupt, [l])
    timer.start()
    start = time.time()
    flag2 = l.solve_limited(expect_interrupt=True)
    end0 = time.time()
    sol = None
    #print(flag2)
    if flag2:
        print("ATTACK", indexicicici)
        sol = l.get_model()
    del l

    return flag2, sol, end0-start

def attackornotattack(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,
                      indexicicici, outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)
    nbre_clausesall, nbre_varsall = [], []
    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)

        """cnf_general_condition = []
        for findex, fvalue in enumerate(features1_ref):
            literal1ici = features1_ref[findex]
            literal2ici = features2_ref[findex]
            literal3ici = features3_ref[findex]
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
        cnf_general += cnf_general_condition"""

    #print(features_replace.shape)

    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    #print(V_ref.shape, V_ref)


    litsici2 = features1_ref[UNKNOWN1].tolist()
    litsici2 = [int(x) for x in litsici2]
    weigth_l_1 = W[labels_np[batchici], UNKNOWN1]
    Wf_l = 1.0 * weigth_l_1
    if args.type == "real":
        litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
            UNKNOWN3].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = 1.0 * W[labels_np[batchici], UNKNOWN1]
        weigth_l_2 = 2.0 * W[labels_np[batchici], UNKNOWN2]
        weigth_l_3 = 3.0 * W[labels_np[batchici], UNKNOWN3]
        Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

    flag_attack = False
    cnf_general2 = copy.copy(cnf_general)
    flag2, sol, timesatsolve = None, None, None
    for aconcurant in range(10):
        if not flag_attack:
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != labels_np[batchici]:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, labels_np[batchici]]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]

                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])

                if len(litsici3) > 0:
                    gscdw = max(find_gcd(weightsici2), 1)
                    Vfinal = np.floor((V-1) / gscdw)
                    weightsici3 = [int(x / gscdw) for x in weightsici2]
                    from pysat.pb import PBEnc

                    cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                                      encoding=args.encoding_type).clauses
                    save_flag = True
                else:
                    cnflr = [[]]
                    save_flag = False
                if save_flag:


                    cnflrfinal = CNF()
                    cnflrfinal.extend(cnflr)
                    cnflrfinal.extend(cnf_general)
                    cnf_general3 = cnflr + cnf_general2

                    nbre_clauses = len(cnf_general3)
                    cnf_general3_var = []
                    for y in cnf_general3:
                        for x in y:
                            cnf_general3_var.append(abs(x))

                    nbre_vars = len(np.unique(cnf_general3_var))

                    # print(nbre_clauses, nbre_vars)

                    nbre_clausesall.append(nbre_clauses)
                    nbre_varsall.append(nbre_vars)

                    flag2, sol, timesatsolve = solve_cnf(args, cnf_general3, indexicicici)
                    if flag2:
                        flag_attack = True
                        print("attack en ", aconcurant)
                    del cnflrfinal, cnflr, PBEnc

    return flag_attack, flag2, sol, timesatsolve, nbre_clausesall, nbre_varsall


def attackornotattackvmaxsat_v2(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,
                      indexicicici, outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)
    nbre_clausesall, nbre_varsall = [], []
    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)

        """cnf_general_condition = []
        for findex, fvalue in enumerate(features1_ref):
            literal1ici = features1_ref[findex]
            literal2ici = features2_ref[findex]
            literal3ici = features3_ref[findex]
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
        cnf_general += cnf_general_condition"""

    #print(features_replace.shape)

    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    #print(V_ref.shape, V_ref)


    litsici2 = features1_ref[UNKNOWN1].tolist()
    litsici2 = [int(x) for x in litsici2]
    #print(labels_np)

    weigth_l_1 = W[labels_np, UNKNOWN1]
    Wf_l = 1.0 * weigth_l_1
    if args.type == "real":
        litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
            UNKNOWN3].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = 1.0 * W[labels_np, UNKNOWN1]
        weigth_l_2 = 2.0 * W[labels_np, UNKNOWN2]
        weigth_l_3 = 3.0 * W[labels_np, UNKNOWN3]
        Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

    flag_attack = False
    cnf_general2 = copy.copy(cnf_general)
    flag2, sol, timesatsolve = None, None, None
    for aconcurant in range(10):
        if not flag_attack:
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != labels_np:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, labels_np]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]

                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])

                if len(litsici3) > 0:
                    gscdw = max(find_gcd(weightsici2), 1)
                    Vfinal = np.floor((V-1) / gscdw)
                    weightsici3 = [int(x / gscdw) for x in weightsici2]
                    from pysat.pb import PBEnc



                    cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                                      encoding=args.encoding_type).clauses
                    save_flag = True
                else:
                    cnflr = [[]]
                    save_flag = False
                if save_flag:


                    cnflrfinal = CNF()
                    cnflrfinal.extend(cnflr)
                    cnflrfinal.extend(cnf_general)
                    cnf_general3 = cnflr + cnf_general2

                    #print(len(cnf_general3))
                    #print("Numbre de cluase attaque", len(cnflrfinal.clauses))

                    nbre_clauses = len(cnf_general3)
                    cnf_general3_var = []
                    for y in cnf_general3:
                        for x in y:
                            cnf_general3_var.append(abs(x))

                    nbre_vars = len(np.unique(cnf_general3_var))


                    #print(nbre_clauses, nbre_vars)

                    nbre_clausesall.append(nbre_clauses)
                    nbre_varsall.append(nbre_vars)

                    flag2, sol, timesatsolve = solve_cnf(args, cnf_general3, indexicicici)
                    if flag2:
                        flag_attack = True
                        print("attack en ", aconcurant)










                    del cnflrfinal, cnflr, PBEnc



    return flag_attack, flag2, sol, timesatsolve, nbre_clausesall, nbre_varsall



def attackornotattackvmaxsat(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,
                      indexicicici, outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)

    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)

        """cnf_general_condition = []
        for findex, fvalue in enumerate(features1_ref):
            literal1ici = features1_ref[findex]
            literal2ici = features2_ref[findex]
            literal3ici = features3_ref[findex]
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
            cnf_general_condition.append([int(-1*literal1ici), int(-1*literal2ici), int(-1*literal3ici)])
        cnf_general += cnf_general_condition"""

    #print(features_replace.shape)

    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    #print(V_ref.shape, V_ref)


    litsici2 = features1_ref[UNKNOWN1].tolist()
    litsici2 = [int(x) for x in litsici2]
    #print(labels_np)

    weigth_l_1 = W[labels_np, UNKNOWN1]
    Wf_l = 1.0 * weigth_l_1
    if args.type == "real":
        litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
            UNKNOWN3].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = 1.0 * W[labels_np, UNKNOWN1]
        weigth_l_2 = 2.0 * W[labels_np, UNKNOWN2]
        weigth_l_3 = 3.0 * W[labels_np, UNKNOWN3]
        Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

    flag_attack = False
    cnf_general2 = copy.copy(cnf_general)
    flag2, sol, timesatsolve = None, None, None
    for aconcurant in range(10):
        if not flag_attack:
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != labels_np:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, labels_np]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]
                from pysat.pb import PBEnc
                if len(litsici2) > 0:
                    cnflr = PBEnc.leq(lits=litsici2, weights=weightsici, bound=int(V) - 1,
                                      encoding=args.encoding_type).clauses
                    save_flag = True
                else:
                    cnflr = [[]]
                    save_flag = False
                if save_flag:


                    cnflrfinal = CNF()
                    cnflrfinal.extend(cnflr)
                    cnflrfinal.extend(cnf_general)
                    cnf_general3 = cnflr + cnf_general2
                    flag2, sol, timesatsolve = solve_cnf(args, cnf_general3, indexicicici)
                    if flag2:
                        flag_attack = True
                        print("attack en ", aconcurant)
                    del cnflrfinal, cnflr, PBEnc

    return flag_attack, flag2, sol, timesatsolve


def verification_vitess(quantized_model_train, images, args, batchici, labels_np, correct,timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                        indexicicici, device, img_ref_inverse):

    correctflag, all_block_acomplter, features_ref, outputs = evaluatenormalement_vitessev2(quantized_model_train, images,
                                                                                     args, batchici, labels_np, correct,
                                                                                     unfoldball,
                                                                                     all_inputbshape,
                                                                                     res_numpyblocall, W, b, device)

    mean_clause = 0
    mean_var = 0
    if correctflag:
        correct += 1
        input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train, batchici)
        if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
            noise_original = args.attack_eps_ici
            all_noise = np.linspace(0.0, noise_original, num=args.quant_noise)[1:]
            all_bruit = torch.zeros_like(input_acomplter)
            for noise in all_noise:
                input_acomplterv2, _, _ = get_input_noise2(images, quantized_model_train, noise)
                all_bruitmask = 1.0*(all_bruit ==0)
                all_bruit = noise*all_bruitmask*input_acomplterv2 + all_bruit



        cnf_general = []
        for numblock in range(len(args.filters)):
            if numblock == 0:
                cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                              input_acomplter,
                                                                                              all_block_acomplter[
                                                                                                  numblock], batchici,
                                                                                              unfoldball[numblock],
                                                                                              all_inputbshape[numblock],
                                                                                              res_numpyblocall[
                                                                                                  numblock],
                                                                                              dictionnaire_ref,
                                                                                              block_refcall[numblock],
                                                                                              cnf_general, reference2)

                if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":

                    litteralsall_l1l2 = []
                    for clauseici in cnf_general:
                        for litterals_ici in clauseici:
                            if abs(litterals_ici)<= input_acomplter.shape[1]*input_acomplter.shape[2]*input_acomplter.shape[3]:
                                litteralsall_l1l2.append(int(abs(litterals_ici)))
                    litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()
                    weigthall_l1l2 = []
                    for litteral_l12 in litteralsall_l1l2_unique:
                        [channel, x_pos, y_pos] = img_ref_inverse[litteral_l12]
                        weigthall_l1l2.append(all_bruit[batchici,channel, x_pos, y_pos].item())

                    if args.type_norm_noise == "l2":
                        noise_original = noise_original**2

                    weigthall_l1l2 = [ int(round(args.coef_multiply_equation_noise * x - 0.49999)) for x in weigthall_l1l2]
                    noisetot = int(args.coef_multiply_equation_noise * noise_original)



                    if len(litteralsall_l1l2_unique) > 0:
                        from pysat.pb import PBEnc
                        cnflr = PBEnc.leq(lits=litteralsall_l1l2_unique, weights=weigthall_l1l2, bound=noisetot,
                                          encoding=args.encoding_type).clauses
                        cnf_general = cnflr + cnf_general
                        del PBEnc, cnflr


            elif numblock == len(args.filters) - 1:
                cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                              torch.Tensor(
                                                                                                  all_block_acomplter[
                                                                                                      numblock - 1]),
                                                                                              all_block_acomplter[
                                                                                                  numblock], batchici,
                                                                                              unfoldball[numblock],
                                                                                              all_inputbshape[numblock],
                                                                                              res_numpyblocall[
                                                                                                  numblock],
                                                                                              dictionnaire_ref,
                                                                                              block_refcall[numblock],
                                                                                                                  cnf_general,
                                                                                              reference2)
            else:
                cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                              torch.Tensor(
                                                                                                  all_block_acomplter[
                                                                                                      numblock - 1]),
                                                                                              all_block_acomplter[
                                                                                                  numblock],
                                                                                              batchici,
                                                                                              unfoldball[
                                                                                                  numblock],
                                                                                              all_inputbshape[
                                                                                                  numblock],
                                                                                              res_numpyblocall[
                                                                                                  numblock],
                                                                                              dictionnaire_ref,
                                                                                              block_refcall[
                                                                                                  numblock],
                                                                                              cnf_general,
                                                                                              reference2)

        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(all_block_acomplter[numblock].shape[0],
                                                                            -1).numpy()

        flag_attack, flag2, sol, timesatsolve, nbre_clausesall, nbre_varsall = attackornotattack(features_replace, batchici,
                                                                  features1_ref, labels_np, W, b, cnf_general,
                                                                  args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock])
        if timesatsolve is None:
            timesatsolve = 0

        if flag_attack:
            correct -= 1
        if timesatsolve > args.time_out:
            timeout+=1

        if len(nbre_clausesall) > 0:
            #print(int(np.mean(nbre_clausesall)), int(np.mean(nbre_varsall)))
            mean_clause = int(np.mean(nbre_clausesall))
            mean_var = int(np.mean(nbre_varsall))

    return correct, timeout,mean_clause, mean_var


def verification_vitessmaxsat(quantized_model_train, images, args, batchici, labels_np, correct, timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                        indexicicici, device, img_ref_inverse):

    correctflag, all_block_acomplter, features_ref, res = evaluatenormalement_vitessev2(quantized_model_train, images,
                                                                                     args, batchici, labels_np, correct,
                                                                                     unfoldball,
                                                                                     all_inputbshape,
                                                                                     res_numpyblocall, W, b, device)


    #if correctflag:
    if args.modeltoeval == "normal" or args.modeltoeval == "prune":
        prediction_ici = np.argmax(res[batchici].detach().cpu().clone().numpy())
    else:
        prediction_ici = np.argmax(res[batchici])

    #print(prediction_ici)

    #input_acomplter, imagesPLUS, imagesMOINS = get_input_noise_maxsat(images, args, quantized_model_train, batchici)
    input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train,
                                                               batchici)
    if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
        noise_original = args.attack_eps_ici
        all_noise = np.linspace(0.0, noise_original, num=args.quant_noise)[1:]
        all_bruit = torch.zeros_like(input_acomplter)
        for noise in all_noise:
            input_acomplterv2, _, _ = get_input_noise2(images, quantized_model_train, noise)
            all_bruitmask = 1.0*(all_bruit ==0)
            all_bruit = noise*all_bruitmask*input_acomplterv2 + all_bruit



    cnf_general = []
    for numblock in range(len(args.filters)):
        if numblock == 0:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          input_acomplter,
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                          cnf_general, reference2)

            cnf_generalorigine = copy.deepcopy(cnf_general)
            litteralsall_l1l2 = []
            for clauseici in cnf_general:
                for litterals_ici in clauseici:
                    if abs(litterals_ici) <= input_acomplter.shape[1] * input_acomplter.shape[2] * \
                            input_acomplter.shape[3]:
                        litteralsall_l1l2.append(int(abs(litterals_ici)))
            litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()



            if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":

                litteralsall_l1l2 = []
                for clauseici in cnf_general:
                    for litterals_ici in clauseici:
                        if abs(litterals_ici)<= input_acomplter.shape[1]*input_acomplter.shape[2]*input_acomplter.shape[3]:
                            litteralsall_l1l2.append(int(abs(litterals_ici)))
                litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()
                weigthall_l1l2 = []
                for litteral_l12 in litteralsall_l1l2_unique:
                    [channel, x_pos, y_pos] = img_ref_inverse[litteral_l12]
                    weigthall_l1l2.append(all_bruit[batchici,channel, x_pos, y_pos].item())

                if args.type_norm_noise == "l2":
                    noise_original = noise_original**2

                weigthall_l1l2 = [ int(round(args.coef_multiply_equation_noise * x - 0.49999)) for x in weigthall_l1l2]
                noisetot = int(args.coef_multiply_equation_noise * noise_original)



                if len(litteralsall_l1l2_unique) > 0:
                    from pysat.pb import PBEnc
                    cnflr = PBEnc.leq(lits=litteralsall_l1l2_unique, weights=weigthall_l1l2, bound=noisetot,
                                      encoding=args.encoding_type).clauses
                    cnf_general = cnflr + cnf_general
                    del PBEnc, cnflr

            #cnf_general = []
        elif numblock == len(args.filters) - 1:
            cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                                              cnf_general,
                                                                                          reference2)
        else:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock],
                                                                                          batchici,
                                                                                          unfoldball[
                                                                                              numblock],
                                                                                          all_inputbshape[
                                                                                              numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[
                                                                                              numblock],
                                                                                          cnf_general,
                                                                                          reference2)

    features_replace = torch.Tensor(all_block_acomplter[numblock]).view(all_block_acomplter[numblock].shape[0],
                                                                        -1).numpy()

    flag_attack, flag2, sol, timesatsolve,nbre_clausesall, nbre_varsall  = attackornotattackvmaxsat(features_replace, batchici,
                                                              features1_ref, prediction_ici, W, b, cnf_general,
                                                              args, indexicicici, outputbinary_refv2,
                                                              block_refcall[
                                                                  numblock])

    if (not flag_attack) and correctflag:
        correct += 1
    elif flag_attack:

        if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
            cnf_generalvf = copy.deepcopy(cnf_general)
        else:
            cnf_generalvf = [x for x in cnf_general if x not in cnf_generalorigine]


        resall, resallmemomry, predict = attackornotattack_maxsat(features_replace, batchici,
                                                                  features1_ref, labels_np, W, b,
                                                                  cnf_generalvf,
                                                                  args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock])
        correctflag2 = predict == labels_np[batchici]
        if correctflag2 and (resallmemomry[predict] > 0):
            correct += 1



    return correct, timeout



def verification_vitesstheo(quantized_model_train, images, args, batchici, labels_np, correct, timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                        indexicicici, device, img_ref_inverse):

    correctflag, all_block_acomplter, features_ref, res = evaluatenormalement_vitessev2(quantized_model_train, images,
                                                                                     args, batchici, labels_np, correct,
                                                                                     unfoldball,
                                                                                     all_inputbshape,
                                                                                     res_numpyblocall, W, b, device)


    #if correctflag:
    if args.modeltoeval == "normal" or args.modeltoeval == "prune":
        prediction_ici = np.argmax(res[batchici].detach().cpu().clone().numpy())
    else:
        prediction_ici = np.argmax(res[batchici])

    p_adv = None
    mean_clause = 0
    mean_var = 0

    input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train, batchici)
    if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
        noise_original = args.attack_eps_ici
        all_noise = np.linspace(0.0, noise_original, num=args.quant_noise)[1:]
        all_bruit = torch.zeros_like(input_acomplter)
        for noise in all_noise:
            input_acomplterv2, _, _ = get_input_noise2(images, quantized_model_train, noise)
            all_bruitmask = 1.0*(all_bruit ==0)
            all_bruit = noise*all_bruitmask*input_acomplterv2 + all_bruit



    cnf_general = []
    for numblock in range(len(args.filters)):
        if numblock == 0:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          input_acomplter,
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                          cnf_general, reference2)

            cnf_generalorigine = copy.deepcopy(cnf_general)



            if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":

                litteralsall_l1l2 = []
                for clauseici in cnf_general:
                    for litterals_ici in clauseici:
                        if abs(litterals_ici)<= input_acomplter.shape[1]*input_acomplter.shape[2]*input_acomplter.shape[3]:
                            litteralsall_l1l2.append(int(abs(litterals_ici)))
                litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()
                weigthall_l1l2 = []
                for litteral_l12 in litteralsall_l1l2_unique:
                    [channel, x_pos, y_pos] = img_ref_inverse[litteral_l12]
                    weigthall_l1l2.append(all_bruit[batchici,channel, x_pos, y_pos].item())

                if args.type_norm_noise == "l2":
                    noise_original = noise_original**2

                weigthall_l1l2 = [ int(round(args.coef_multiply_equation_noise * x - 0.49999)) for x in weigthall_l1l2]
                noisetot = int(args.coef_multiply_equation_noise * noise_original)



                if len(litteralsall_l1l2_unique) > 0:
                    from pysat.pb import PBEnc
                    cnflr = PBEnc.leq(lits=litteralsall_l1l2_unique, weights=weigthall_l1l2, bound=noisetot,
                                      encoding=args.encoding_type).clauses
                    cnf_general = cnflr + cnf_general
                    del PBEnc, cnflr


        elif numblock == len(args.filters) - 1:
            cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                                              cnf_general,
                                                                                          reference2)
        else:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock],
                                                                                          batchici,
                                                                                          unfoldball[
                                                                                              numblock],
                                                                                          all_inputbshape[
                                                                                              numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[
                                                                                              numblock],
                                                                                          cnf_general,
                                                                                          reference2)

    features_replace = torch.Tensor(all_block_acomplter[numblock]).view(all_block_acomplter[numblock].shape[0],
                                                                        -1).numpy()

    flag_attack, flag2, sol, timesatsolve, nbre_clausesall, nbre_varsall = attackornotattackvmaxsat_v2(features_replace, batchici,
                                                              features1_ref, prediction_ici, W, b, cnf_general,
                                                              args, indexicicici, outputbinary_refv2,
                                                              block_refcall[
                                                                  numblock])

    if (not flag_attack) and correctflag:
        correct += 1
    elif flag_attack:

        if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
            cnf_generalvf = copy.deepcopy(cnf_general)
        else:
            cnf_generalvf = [x for x in cnf_general if x not in cnf_generalorigine]

        resall, resallmemomry, predict = attackornotattack_maxsat(features_replace, batchici,
                                                                  features1_ref, labels_np, W, b,
                                                                  cnf_generalvf,
                                                                  args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock])

        flagTimeout = False
        cpt_timeout = 0
        for iterpreci in range(10):
            if resall[iterpreci] == 2 or resallmemomry[iterpreci] == 2:
                flagTimeout = True
            if resall[iterpreci] == 0 and resallmemomry[iterpreci] == 0:
                cpt_timeout += 1
        if cpt_timeout == 10:
            flagTimeout = True
        if flagTimeout:
            timeout += 1

        correctflag2 = predict == labels_np[batchici]

        if correctflag2 and (resallmemomry[predict]>0):

            numerateur = resallmemomry[predict] * 10**resall[predict]
            denumerateur = 0
            for iterpreci in range(10):
                denumerateur += resallmemomry[iterpreci] * 10 ** resall[iterpreci]

            p_adv = 1- numerateur/denumerateur

            print(p_adv)

            correct += 1
    if len(nbre_clausesall)>0:
        print(int(np.mean(nbre_clausesall)), int(np.mean(nbre_varsall)))
        mean_clause = int(np.mean(nbre_clausesall))
        mean_var = int(np.mean(nbre_varsall))




    return correct, timeout, p_adv, mean_clause, mean_var


def verification_vitesstheo2(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                        indexicicici, device, img_ref_inverse):

    correctflag, all_block_acomplter, features_ref, res = evaluatenormalement_vitessev2(quantized_model_train, images,
                                                                                     args, batchici, labels_np, correct,
                                                                                     unfoldball,
                                                                                     all_inputbshape,
                                                                                     res_numpyblocall, W, b, device)


    #if correctflag:
    if args.modeltoeval == "normal" or args.modeltoeval == "prune":
        prediction_ici = np.argmax(res[batchici].detach().cpu().clone().numpy())
    else:
        prediction_ici = np.argmax(res[batchici])

    #print(prediction_ici)

    input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train, batchici)
    if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
        noise_original = args.attack_eps_ici
        all_noise = np.linspace(0.0, noise_original, num=args.quant_noise)[1:]
        all_bruit = torch.zeros_like(input_acomplter)
        for noise in all_noise:
            input_acomplterv2, _, _ = get_input_noise2(images, quantized_model_train, noise)
            all_bruitmask = 1.0*(all_bruit ==0)
            all_bruit = noise*all_bruitmask*input_acomplterv2 + all_bruit



    cnf_general = []
    for numblock in range(len(args.filters)):
        if numblock == 0:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          input_acomplter,
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                          cnf_general, reference2)

            cnf_generalorigine = copy.deepcopy(cnf_general)



            if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":

                litteralsall_l1l2 = []
                for clauseici in cnf_general:
                    for litterals_ici in clauseici:
                        if abs(litterals_ici)<= input_acomplter.shape[1]*input_acomplter.shape[2]*input_acomplter.shape[3]:
                            litteralsall_l1l2.append(int(abs(litterals_ici)))
                litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()
                weigthall_l1l2 = []
                for litteral_l12 in litteralsall_l1l2_unique:
                    [channel, x_pos, y_pos] = img_ref_inverse[litteral_l12]
                    weigthall_l1l2.append(all_bruit[batchici,channel, x_pos, y_pos].item())

                if args.type_norm_noise == "l2":
                    noise_original = noise_original**2

                weigthall_l1l2 = [ int(round(args.coef_multiply_equation_noise * x - 0.49999)) for x in weigthall_l1l2]
                noisetot = int(args.coef_multiply_equation_noise * noise_original)



                if len(litteralsall_l1l2_unique) > 0:
                    from pysat.pb import PBEnc
                    cnflr = PBEnc.leq(lits=litteralsall_l1l2_unique, weights=weigthall_l1l2, bound=noisetot,
                                      encoding=args.encoding_type).clauses
                    cnf_general = cnflr + cnf_general
                    del PBEnc, cnflr


        elif numblock == len(args.filters) - 1:
            cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock], batchici,
                                                                                          unfoldball[numblock],
                                                                                          all_inputbshape[numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[numblock],
                                                                                                              cnf_general,
                                                                                          reference2)
        else:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args, numblock,
                                                                                          torch.Tensor(
                                                                                              all_block_acomplter[
                                                                                                  numblock - 1]),
                                                                                          all_block_acomplter[
                                                                                              numblock],
                                                                                          batchici,
                                                                                          unfoldball[
                                                                                              numblock],
                                                                                          all_inputbshape[
                                                                                              numblock],
                                                                                          res_numpyblocall[
                                                                                              numblock],
                                                                                          dictionnaire_ref,
                                                                                          block_refcall[
                                                                                              numblock],
                                                                                          cnf_general,
                                                                                          reference2)

    features_replace = torch.Tensor(all_block_acomplter[numblock]).view(all_block_acomplter[numblock].shape[0],
                                                                        -1).numpy()

    flag_attack, flag2, sol, timesatsolve = attackornotattackvmaxsat(features_replace, batchici,
                                                              features1_ref, prediction_ici, W, b, cnf_general,
                                                              args, indexicicici, outputbinary_refv2,
                                                              block_refcall[
                                                                  numblock])

    if (not flag_attack) and correctflag:
        correct += 1
    elif flag_attack:

        if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
            cnf_generalvf = copy.deepcopy(cnf_general)
        else:
            cnf_generalvf = [x for x in cnf_general if x not in cnf_generalorigine]


        resall, resallmemomry, predict = attackornotattack_theo2(features_replace, batchici,
                                                                  features1_ref, labels_np, W, b,
                                                                  cnf_generalvf,
                                                                  args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock])
        correctflag2 = predict == labels_np[batchici]
        if correctflag2 and (resallmemomry[predict]>0):

            correct += 1


    return correct

def get_dictionnaire_ref(args, img_ref, unfoldball, block_refcall):
    dictionnaire_ref = {}
    for numblock in range(len(args.filters)):
        dictionnaire_ref[numblock] = {}
        if numblock == 0:
            nombredefiltredansgroupe = int(args.nchannel / args.groups[0])
            input_binary_ref = img_ref
            unfoldblock = unfoldball[numblock]
        else:
            nombredefiltredansgroupe = int(args.filters[numblock - 1] / args.groups[numblock])
            input_binary_ref = block_refcall[numblock - 1]
            unfoldblock = unfoldball[numblock]
        for groupici in range(args.groups[numblock]):
            #print(input_binary_ref.shape)
            if len(input_binary_ref.shape) == 4:
                inputref_vu_par_cnn_avant_unfold = input_binary_ref[:, groupici * nombredefiltredansgroupe
                                                                   : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                               :, :]
            else:
                inputref_vu_par_cnn_avant_unfold = input_binary_ref[:, groupici * nombredefiltredansgroupe
                                                                       : groupici * nombredefiltredansgroupe + nombredefiltredansgroupe,
                                                   :, :, 0]
            inputref_vu_par_cnn_et_sat = unfoldblock(inputref_vu_par_cnn_avant_unfold)

            dictionnaire_ref[numblock][groupici] = inputref_vu_par_cnn_et_sat
    return dictionnaire_ref


def verification_total(quantized_model_train, images, args, batchici, labels_np, correct, timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b, indexicicici,
                       device, img_ref, img_refsizex, img_ref_inverse):


    correctflag, all_block_acomplter, features_ref, outputs = evaluatenormalement_v2(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball,
                                  all_inputbshape,
                                  res_numpyblocall, W, b, device)

    #print(outputs, correctflag, labels_np)


    if correctflag:

        correct += 1
        #if args.type == "binary":
        #    _, _, _, _, _ = eval_model_general_binary(
        #    quantized_model_train.eval(), images, len(args.filters))
        #elif args.type == "real":
        #    _, _, _, _, _ = eval_model_general_real(
        #        quantized_model_train.eval(), images, len(args.filters))

        input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train,
                                                                   batchici)
        if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
            noise_original = args.attack_eps_ici
            all_noise = np.linspace(0.0, noise_original, num=args.quant_noise)[1:]
            all_bruit = torch.zeros_like(input_acomplter)
            for noise in all_noise:
                input_acomplterv2, _, _ = get_input_noise2(images, quantized_model_train, noise)
                all_bruitmask = 1.0*(all_bruit ==0)
                all_bruit = noise*all_bruitmask*input_acomplterv2 + all_bruit

        cnf_general = []



        for numblock in range(len(args.filters)):
            if numblock == 0:
                cnf_general, all_block_acomplter[numblock],_ = infer_1_block_sat_verify(args,
                                                                                      numblock,
                                                                                      input_acomplter,
                                                                                      all_block_acomplter[
                                                                                          numblock],
                                                                                      batchici,
                                                                                      unfoldball[
                                                                                          numblock],
                                                                                      all_inputbshape[
                                                                                          numblock],
                                                                                      res_numpyblocall[
                                                                                          numblock],
                                                                                      dictionnaire_ref,
                                                                                      block_refcall[
                                                                                          numblock],
                                                                                      cnf_general,
                                                                                      reference2)

                if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":

                    litteralsall_l1l2 = []
                    for clauseici in cnf_general:
                        for litterals_ici in clauseici:
                            if abs(litterals_ici)<= input_acomplter.shape[1]*input_acomplter.shape[2]*input_acomplter.shape[3]:
                                litteralsall_l1l2.append(int(abs(litterals_ici)))
                    litteralsall_l1l2_unique = np.unique(litteralsall_l1l2).tolist()
                    weigthall_l1l2_origine = []
                    for litteral_l12 in litteralsall_l1l2_unique:
                        [channel, x_pos, y_pos] = img_ref_inverse[litteral_l12]
                        weigthall_l1l2_origine.append(all_bruit[batchici,channel, x_pos, y_pos].item())

                    if args.type_norm_noise == "l2":
                        weigthall_l1l2_origine = [x**2 for x in weigthall_l1l2_origine]
                        noise_original = noise_original**2
                    #print(weigthall_l1l2_origine, noise_original)
                    weigthall_l1l2 = [ int(round(args.coef_multiply_equation_noise * x - 0.49999)) for x in weigthall_l1l2_origine]
                    noisetot = int(args.coef_multiply_equation_noise * noise_original)
                    #print(weigthall_l1l2, noisetot)



                    if len(litteralsall_l1l2_unique) > 0:
                        from pysat.pb import PBEnc
                        cnflr = PBEnc.leq(lits=litteralsall_l1l2_unique, weights=weigthall_l1l2, bound=noisetot,
                                          encoding=args.encoding_type).clauses
                        cnf_general = cnflr + cnf_general
                        del PBEnc, cnflr


            elif numblock == len(args.filters) - 1:
                cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify(args,
                                                                                      numblock,
                                                                                      torch.Tensor(
                                                                                          all_block_acomplter[
                                                                                              numblock - 1]),
                                                                                      all_block_acomplter[
                                                                                          numblock],
                                                                                      batchici,
                                                                                      unfoldball[
                                                                                          numblock],
                                                                                      all_inputbshape[
                                                                                          numblock],
                                                                                      res_numpyblocall[
                                                                                          numblock],
                                                                                      dictionnaire_ref,
                                                                                      block_refcall[
                                                                                          numblock],
                                                                                      cnf_general,
                                                                                      reference2)
            else:
                cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify(args,
                                                                                      numblock,
                                                                                      torch.Tensor(
                                                                                          all_block_acomplter[
                                                                                              numblock - 1]),
                                                                                      all_block_acomplter[
                                                                                          numblock],
                                                                                      batchici,
                                                                                      unfoldball[
                                                                                          numblock],
                                                                                      all_inputbshape[
                                                                                          numblock],
                                                                                      res_numpyblocall[
                                                                                          numblock],
                                                                                      dictionnaire_ref,
                                                                                      block_refcall[
                                                                                          numblock],
                                                                                      cnf_general,
                                                                                      reference2)

        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        KNOWN = (features_replace[batchici, :] >= 0)



        assert all(features_replace[batchici, KNOWN] == features_ref[batchici, KNOWN])

        flag_attack, flag2, sol, timesatsolve, nbre_clausesall, nbre_varsall = attackornotattack(features_replace, batchici,
                                                                  features1_ref, labels_np, W, b,
                                                                  cnf_general,
                                                                  args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock]
                                                                  )

        if flag_attack:



            input_acomplter_version_attack = input_acomplter.clone()
            for i in range(args.nchannel):
                for j in range(img_refsizex):
                    for k in range(img_refsizex):
                        value = input_acomplter[batchici, i, j, k]
                        value_ref = img_ref[batchici, i, j, k]
                        if value == -1:
                            if int(value_ref) in sol:
                                input_acomplter_version_attack[batchici, i, j, k] = 1
                            elif int(-1 * value_ref) in sol:
                                input_acomplter_version_attack[batchici, i, j, k] = 0
                            else:
                                raise "PB"


            if args.type_norm_noise == "l1" or args.type_norm_noise == "l2":
                cpttotverif_l1 =  0
                for index_value, value_ref in enumerate(litteralsall_l1l2_unique):
                    if int(value_ref) in sol:
                        #if args.type_norm_noise == "l1":
                        cpttotverif_l1+=weigthall_l1l2_origine[index_value]
                        #elif args.type_norm_noise == "l1":
                #print(noise_original, cpttotverif_l1)
                assert noise_original >= cpttotverif_l1

                imagesPLUS2 = images.clone() + all_bruit.float()
                imagesMOINS2 = images.clone() - all_bruit.float()
                input_float_version_attack = (input_acomplter_version_attack == 1).float() * imagesPLUS2.float() + (
                    input_acomplter_version_attack == 0).float() * imagesMOINS2.float()

            else:
                input_float_version_attack = (input_acomplter_version_attack == 1).float().cpu() * imagesPLUS.cpu() + (
                    input_acomplter_version_attack == 0).float().cpu() * imagesMOINS.cpu()

            if args.modeltoeval == "normal" or args.modeltoeval == "prune":

                if args.type == "binary":
                    res_attack, input_binary_version_attack, _, _, _ = eval_model_general_binary(
                        quantized_model_train.eval(), input_float_version_attack.to(device), len(args.filters))
                elif args.type == "real":
                    res_attack, input_binary_version_attack, _, _, _ = eval_model_general_real(
                        quantized_model_train.eval(), input_float_version_attack.to(device), len(args.filters))



                predictednpres_attack = np.argmax(res_attack.detach().cpu().clone().numpy())

                correctflag = predictednpres_attack != labels_np[batchici]

                assert correctflag
                correct -= 1

            elif args.modeltoeval == "filtered" or args.modeltoeval == "prune_filtered":

                correctflag, _, _, outputs_quan = evaluatenormalement_v2(quantized_model_train.eval(),
                                                                                                 input_float_version_attack.to(device), args, batchici,
                                                                                                 labels_np, correct,
                                                                                                 unfoldball,
                                                                                                 all_inputbshape,
                                                                                                 res_numpyblocall, W, b,
                                                                                                 device)

                #print(correctflag, outputs_quan)

                assert (correctflag == False)
                correct -= 1





    return correct, timeout



def evaluatenormalement(quantized_model_train, images, args, batchici, labels_np, correct, timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, W, b, device):
    if args.type == "binary":
        res, input_binary, all_block_acomplter, features_ref, _ = eval_model_general_binary(
        quantized_model_train.eval(), images, len(args.filters))
    elif args.type == "real":
        res, input_binary, all_block_acomplter, features_ref, _ = eval_model_general_real(
            quantized_model_train.eval(), images, len(args.filters))

    outputs = quantized_model_train(images.to(device))

    if args.modeltoeval == "normal" or args.modeltoeval == "prune":
        assert all(outputs[0].detach().cpu().clone().numpy() == res[0].detach().cpu().clone().numpy())
        for numblock in range(len(args.filters)):
            if numblock == 0:
                infer_1_block_sat(args, numblock, input_binary, all_block_acomplter[numblock], batchici,
                                  unfoldball[numblock], all_inputbshape[numblock], res_numpyblocall[numblock])
            else:
                infer_1_block_sat(args, numblock, torch.Tensor(all_block_acomplter[numblock - 1]),
                                  all_block_acomplter[numblock],
                                  batchici, unfoldball[numblock], all_inputbshape[numblock], res_numpyblocall[numblock])

        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b

        if args.type == "binary":
            assert all(outputs[0].detach().cpu().clone().numpy() == outputs_quan[0])
        elif args.type == "real":
            assert all(outputs[0].detach().cpu().clone().numpy() == outputs_quan)


    elif args.modeltoeval == "filtered" or args.modeltoeval == "prune_filtered":

        for numblock in range(len(args.filters)):
            if numblock == 0:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, input_binary,
                                                                              all_block_acomplter[numblock], batchici,
                                                                              unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
            else:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, torch.Tensor(
                    all_block_acomplter[numblock - 1]),
                                                                              all_block_acomplter[numblock],
                                                                              batchici, unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b

    predictednpres = np.argmax(outputs_quan)
    correctflag = predictednpres == labels_np[batchici]
    if correctflag:
        correct += 1

    return correct, timeout


def evaluatenormalement_v2(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball, all_inputbshape,
                        res_numpyblocall, W, b, device):


    if args.type == "binary":
        res, input_binary, all_block_acomplter, features_ref, _ = eval_model_general_binary(
        quantized_model_train.eval(), images, len(args.filters))
    elif args.type == "real":
        res, input_binary, all_block_acomplter, features_ref, _ = eval_model_general_real(
            quantized_model_train.eval(), images, len(args.filters))


    outputs = quantized_model_train(images.to(device))

    if args.modeltoeval == "normal" or args.modeltoeval == "prune":
        if args.type == "binary":
            assert all(outputs[0].detach().cpu().clone().numpy() == res[0].detach().cpu().clone().numpy())
        for numblock in range(len(args.filters)):
            if numblock == 0:
                infer_1_block_sat(args, numblock, input_binary, all_block_acomplter[numblock], batchici,
                                  unfoldball[numblock], all_inputbshape[numblock], res_numpyblocall[numblock])
            else:
                infer_1_block_sat(args, numblock, torch.Tensor(all_block_acomplter[numblock - 1]),
                                  all_block_acomplter[numblock],
                                  batchici, unfoldball[numblock], all_inputbshape[numblock], res_numpyblocall[numblock])

        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b
        #assert all(outputs[0].detach().cpu().clone().numpy() == outputs_quan[0])


        if args.type == "binary":
            assert all(outputs[0].detach().cpu().clone().numpy() == outputs_quan[0])
        elif args.type == "real":
            assert all(outputs[0].detach().cpu().clone().numpy() == outputs_quan)

    elif args.modeltoeval == "filtered" or args.modeltoeval == "prune_filtered":

        for numblock in range(len(args.filters)):
            if numblock == 0:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, input_binary,
                                                                              all_block_acomplter[numblock], batchici,
                                                                              unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
            else:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, torch.Tensor(
                    all_block_acomplter[numblock - 1]),
                                                                              all_block_acomplter[numblock],
                                                                              batchici, unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b

    predictednpres = np.argmax(outputs_quan)
    correctflag = predictednpres == labels_np[batchici]


    return correctflag, all_block_acomplter, features_replace, outputs_quan

def evaluatenormalement_vitesse(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball, all_inputbshape,
                        res_numpyblocall, W, b, device):
    res, input_binary, all_block_acomplter, features_ref, _ = eval_model_general_binary(
        quantized_model_train.eval(), images, len(args.filters))

    outputs_quan = quantized_model_train(images.to(device))



    if args.modeltoeval == "filtered" or args.modeltoeval == "prune_filtered":

        for numblock in range(len(args.filters)):
            if numblock == 0:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, input_binary,
                                                                              all_block_acomplter[numblock], batchici,
                                                                              unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
            else:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, torch.Tensor(
                    all_block_acomplter[numblock - 1]),
                                                                              all_block_acomplter[numblock],
                                                                              batchici, unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b

    predictednpres = np.argmax(outputs_quan)
    correctflag = predictednpres == labels_np[batchici]


    return correctflag, all_block_acomplter, features_replace, outputs_quan


def evaluatenormalement_vitessev2(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball, all_inputbshape,
                        res_numpyblocall, W, b, device):


    if args.type == "binary":
        res, input_binary, all_block_acomplter, features_replace, _ = eval_model_general_binary(
        quantized_model_train.eval(), images, len(args.filters))
        _, input_binary_duplicate, all_block_acomplter_duplicate, _, _ = eval_model_general_binary(
            quantized_model_train.eval(), images, len(args.filters))
    elif args.type == "real":
        res, input_binary, all_block_acomplter, features_replace, _ = eval_model_general_real(
            quantized_model_train.eval(), images, len(args.filters))
        _, input_binary_duplicate, all_block_acomplter_duplicate, _, _ = eval_model_general_real(
            quantized_model_train.eval(), images, len(args.filters))


    outputs_quan = quantized_model_train(images.to(device)).cpu()



    if args.modeltoeval == "filtered" or args.modeltoeval == "prune_filtered":

        for numblock in range(len(args.filters)):
            if numblock == 0:
                all_block_acomplter[numblock], all_block_acomplter_duplicate[numblock] = infer_and_replace_1_block_sat_vitesse(args, numblock, input_binary,
                                                                              all_block_acomplter[numblock], batchici,
                                                                              unfoldball[numblock],
                                                                              res_numpyblocall[numblock], input_binary,
                                                                                      all_block_acomplter_duplicate[numblock])
            else:
                all_block_acomplter[numblock], all_block_acomplter_duplicate[numblock] = infer_and_replace_1_block_sat_vitesse(args, numblock, torch.Tensor(
                    all_block_acomplter[numblock - 1]),
                                                                              all_block_acomplter[numblock],
                                                                              batchici, unfoldball[numblock],
                                                                              res_numpyblocall[numblock], torch.Tensor(all_block_acomplter[numblock - 1]),
                                                                                      all_block_acomplter_duplicate[numblock]
                                                                                                                               )
        """

        for numblock in range(len(args.filters)):
            if numblock == 0:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, input_binary,
                                                                              all_block_acomplter[numblock], batchici,
                                                                              unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
            else:
                all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, torch.Tensor(
                    all_block_acomplter[numblock - 1]),
                                                                              all_block_acomplter[numblock],
                                                                              batchici, unfoldball[numblock],
                                                                              res_numpyblocall[numblock])
        """
        features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
            all_block_acomplter[numblock].shape[0],
            -1).numpy()
        outputs_quan = np.dot(W, features_replace[batchici]) + b

    predictednpres = np.argmax(outputs_quan)
    correctflag = predictednpres == labels_np[batchici]


    return correctflag, all_block_acomplter, features_replace, outputs_quan


def evaluation_robuste(quantized_model_train, images, args, batchici, labels_np, correct, unfoldball,
                        res_numpyblocall, W, b, all_inputbshape, device):


    _, all_block_acomplter, _, _ = evaluatenormalement_v2(quantized_model_train, images,
                                                                                     args, batchici, labels_np, correct,
                                                                                     unfoldball,
                                                                                     all_inputbshape,
                                                                                     res_numpyblocall, W, b, device)

    input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train,
                                                               batchici)

    for numblock in range(len(args.filters)):
        if numblock == 0:

            all_block_acomplter[numblock] = infer_and_replace_1_block_sat_robuste(args, numblock, input_acomplter,
                                                                                  all_block_acomplter[numblock],
                                                                                  batchici, unfoldball[numblock],
                                                                                  res_numpyblocall[numblock])

        else:
            all_block_acomplter[numblock] = infer_and_replace_1_block_sat(args, numblock, torch.Tensor(
                all_block_acomplter[numblock - 1]),
                                                                          all_block_acomplter[numblock],
                                                                          batchici, unfoldball[numblock],
                                                                          res_numpyblocall[numblock])

    features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
        all_block_acomplter[numblock].shape[0],
        -1).numpy()
    outputs_quan = np.dot(W, features_replace[batchici]) + b
    predictednpres = np.argmax(outputs_quan)

    correctflag = predictednpres == labels_np[batchici]
    if correctflag:
        correct += 1
    return correct

def update_args(args, path_save_model, device):
    model = (ModelHelper.
             create_with_load(path_save_model + "/last.pth").
             to(device).
             eval())
    nbre_block = 2
    if len(model.features) > 20:
        nbre_block = 3
    args.nchannel = model.features[1].weight.shape[0]
    new_filters = []
    new_groups = []
    new_kernelsizes = []
    for nnnn in range(nbre_block):
        new_filters.append(model.features[7 + nnnn * 6].weight.shape[0])
        new_kernelsizes.append(model.features[3 + nnnn * 6].weight.shape[-1])
        if nnnn == 0:
            new_groups.append(int(args.nchannel / model.features[3 + nnnn * 6].weight.shape[1]))
        else:
            new_groups.append(int(new_filters[nnnn - 1] / model.features[3 + nnnn * 6].weight.shape[1]))
    args.filters = new_filters
    args.groups = new_groups
    args.kernelsizes = new_kernelsizes
    if len(args.strides) > nbre_block:
        args.strides = args.strides[:nbre_block]
    elif len(args.strides) < nbre_block:
        args.strides.append(1)
    return args


def verification_total_maxsat(quantized_model_train, images, args, batchici, labels_np, correct, timeout, unfoldball, all_inputbshape,
                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b, indexicicici,
                       device, img_ref, img_refsizex):




    if args.type == "binary":
        res, _, all_block_acomplter, features_ref, _ = eval_model_general_binary(
            quantized_model_train.eval(), images, len(args.filters))
    elif args.type == "real":
        res, _, all_block_acomplter, features_ref, _  = eval_model_general_real(
            quantized_model_train.eval(), images, len(args.filters))

    input_acomplter, imagesPLUS, imagesMOINS = get_input_noise(images, args, quantized_model_train,
                                                               batchici)
    cnf_general = []



    for numblock in range(len(args.filters)):
        if numblock == 0:
            _, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args,
                                                                                  numblock,
                                                                                  input_acomplter,
                                                                                  all_block_acomplter[
                                                                                      numblock],
                                                                                  batchici,
                                                                                  unfoldball[
                                                                                      numblock],
                                                                                  all_inputbshape[
                                                                                      numblock],
                                                                                  res_numpyblocall[
                                                                                      numblock],
                                                                                  dictionnaire_ref,
                                                                                  block_refcall[
                                                                                      numblock],
                                                                                  cnf_general,
                                                                                  reference2)
            cnf_general = []




        elif numblock == len(args.filters) - 1:
            cnf_general, all_block_acomplter[numblock], outputbinary_refv2 = infer_1_block_sat_verify_vitesse(args,
                                                                                  numblock,
                                                                                  torch.Tensor(
                                                                                      all_block_acomplter[
                                                                                          numblock - 1]),
                                                                                  all_block_acomplter[
                                                                                      numblock],
                                                                                  batchici,
                                                                                  unfoldball[
                                                                                      numblock],
                                                                                  all_inputbshape[
                                                                                      numblock],
                                                                                  res_numpyblocall[
                                                                                      numblock],
                                                                                  dictionnaire_ref,
                                                                                  block_refcall[
                                                                                      numblock],
                                                                                  cnf_general,
                                                                                  reference2)
        else:
            cnf_general, all_block_acomplter[numblock], _ = infer_1_block_sat_verify_vitesse(args,
                                                                                  numblock,
                                                                                  torch.Tensor(
                                                                                      all_block_acomplter[
                                                                                          numblock - 1]),
                                                                                  all_block_acomplter[
                                                                                      numblock],
                                                                                  batchici,
                                                                                  unfoldball[
                                                                                      numblock],
                                                                                  all_inputbshape[
                                                                                      numblock],
                                                                                  res_numpyblocall[
                                                                                      numblock],
                                                                                  dictionnaire_ref,
                                                                                  block_refcall[
                                                                                      numblock],
                                                                                  cnf_general,
                                                                                  reference2)


    features_replace = torch.Tensor(all_block_acomplter[numblock]).view(
        all_block_acomplter[numblock].shape[0],
        -1).numpy()

    KNOWN = (features_replace[batchici, :] >= 0)

    #assert all(features_replace[batchici, KNOWN] == features_ref[batchici, KNOWN])

    resall, resallmemomry, predict = attackornotattack_maxsat(features_replace, batchici,
                                                              features1_ref, labels_np, W, b,
                                                              cnf_general,
                                                              args, indexicicici, outputbinary_refv2,
                                                                  block_refcall[
                                                                      numblock])




    correctflag = predict == labels_np[batchici]
    if correctflag:
        correct += 1

    return correct,timeout

def attackornotattack_maxsat(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,indexicicici,
                             outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)

    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)


    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    print("LABEL REF", labels_np[batchici], len(cnf_general))

    resall = {}
    resallmemomry = {}
    predict = 0
    predict_value = 0
    for label in range(10):
        cnf_general2 = copy.copy(cnf_general)
        cnf_general3 = cnf_general2
        cnflrfinal = CNF()
        cnflrfinal.extend(cnf_general)
        litsici2 = features1_ref[UNKNOWN1].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = W[label, UNKNOWN1]
        Wf_l = 1.0 * weigth_l_1
        if args.type == "real":
            litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
                UNKNOWN3].tolist()
            litsici2 = [int(x) for x in litsici2]
            weigth_l_1 = 1.0 * W[labels_np[batchici], UNKNOWN1]
            weigth_l_2 = 2.0 * W[labels_np[batchici], UNKNOWN2]
            weigth_l_3 = 3.0 * W[labels_np[batchici], UNKNOWN3]
            Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

        for aconcurant in range(10):
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != label:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, label]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]

                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])


                if len(litsici3) > 0:
                    gscdw = max(find_gcd(weightsici2),1)
                    #print(gscdw, weightsici2)
                    Vfinal = int(np.floor((V+1) / gscdw))
                    weightsici3 = [int(x/gscdw) for x in weightsici2]
                    from pysat.pb import PBEnc

                    cnflr = PBEnc.geq(lits=litsici3, weights=weightsici3, bound=Vfinal,
                                      encoding=args.encoding_type).clauses
                    cnf_general3 += cnflr
                    cnflrfinal.extend(cnflr)
                    del cnflr, PBEnc
        print("Numbre de cluase", len(cnflrfinal.clauses))
        cnflrfinal.to_file("./ganak/scripts/" + args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf")
        #time.sleep(1)
        #print(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf > "+args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log ; cd ./../../")
        starttime = time.time()
        os.system(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf > "+args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log ; cd ./../../")
        #os.system(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf ; cd ./../../")
        endtime = time.time()

        if endtime - starttime < args.time_out:

            resall[label] = 0
            resallmemomry[label] = 0


            with open("./ganak/scripts/" +args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log", 'r') as fh:
                for line in fh:
                    if "s mc" in line:
                        lineclean = line.replace(" ", "").replace("smc", "").replace('/n', "")
                        resall[label] = len(lineclean)
                        if resall[label] ==2:
                            #print("./ganak/scripts/" +args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log")
                            resall[label] = int(lineclean[:-1])
                        last_index = min(10, len(lineclean))-1
                        if '' != lineclean[:last_index].replace(" ", "").replace("\n", ""):
                            resallmemomry[label] = int(lineclean[:last_index].replace(" ", "").replace("\n", ""))
                        else:
                            resallmemomry[label] = 0
                        if len(lineclean) > resall[predict]:
                            predict = label
                        elif len(lineclean) == resall[predict]:
                            if int(resallmemomry[label]) > resallmemomry[predict]:
                                predict = label

        else:
            resall[label] = 2
            resallmemomry[label] = 0






        del cnflrfinal, cnf_general3


    print("Labels proposed : ", resall, predict)
    print(resallmemomry)

    return resall, resallmemomry, predict

def calcultae_theo(p,b,n):
    cpt = 0
    b2 = round(b + 0.5 + 1e-5) - 1
    for l in range(b2 + 1):
        for k in range(n + 1):
            cpt += math.comb(p, l + k) * math.comb(n, k)
    return cpt


def attackornotattack_theo(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,indexicicici,
                             outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)

    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)


    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    print("LABEL REF", labels_np[batchici], len(cnf_general))

    resall = {}
    resallmemomry = {}
    predict = 0
    predict_value = 0
    for label in range(10):
        cnf_general2 = copy.copy(cnf_general)
        cnf_general3 = cnf_general2
        cnflrfinal = CNF()
        #cnflrfinal.extend(cnf_general)
        litsici2 = features1_ref[UNKNOWN1].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = W[label, UNKNOWN1]
        Wf_l = 1.0 * weigth_l_1
        if args.type == "real":
            litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
                UNKNOWN3].tolist()
            litsici2 = [int(x) for x in litsici2]
            weigth_l_1 = 1.0 * W[labels_np[batchici], UNKNOWN1]
            weigth_l_2 = 2.0 * W[labels_np[batchici], UNKNOWN2]
            weigth_l_3 = 3.0 * W[labels_np[batchici], UNKNOWN3]
            Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

        unique_var = []
        unique_wei = {}
        vf = 0
        for aconcurant in range(10):
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != label:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, label]
                #vf += (V)
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]

                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] !=0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])


                """for index_litteral, litteral in enumerate(litsici3):
                    if weightsici2[index_litteral] !=0:
                        if litteral not in unique_var:
                            unique_var.append(litteral)
                            unique_wei[litteral] = weightsici2[index_litteral]
                        else:
                            unique_wei[litteral] += weightsici2[index_litteral]
                print(litsici3, weightsici2, V)"""
                #wigth_finale = [unique_wei[uniquevarici] for uniquevarici in unique_var]
                #print(litsici3, weightsici2, V-1)
                if len(litsici3) > 0:
                    gscdw = max(find_gcd(weightsici2),1)
                    #print(gscdw, weightsici2)
                    Vfinal = int(np.floor((V+1) / gscdw))
                    weightsici3 = [int(x / gscdw) for x in weightsici2]

                    p = np.sum([x>0 for x in weightsici3])
                    n = np.sum([x < 0 for x in weightsici3])
                    calcultae_theo(p, Vfinal, n)

                    #from pysat.pb import PBEnc
                    #cnflr = PBEnc.geq(lits=litsici3, weights=weightsici3, bound=Vfinal,
                    #                  encoding=args.encoding_type).clauses
                    #cnf_general3 += cnflr
                    #cnflrfinal.extend(cnflr)
                    #del cnflr, PBEnc
        #print(unique_wei, wigth_finale, vf)
        #print(ok)
        print("Numbre de cluase", len(cnflrfinal.clauses))
        cnflrfinal.to_file("./ganak/scripts/" + args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf")
        #time.sleep(1)
        #print(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf > "+args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log ; cd ./../../")
        os.system(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf > "+args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log ; cd ./../../")
        #os.system(" cd ./ganak/scripts/ ; ./run_ganak.sh "+args.path_exp.replace("/", "_").replace(".", "") + "test"+str(label)+".cnf ; cd ./../../")

        resall[label] = 0
        resallmemomry[label] = 0

        with open("./ganak/scripts/" +args.path_exp.replace("/", "_").replace(".", "") + "res"+str(label)+".log", 'r') as fh:
            for line in fh:
                if "s mc" in line:
                    lineclean = line.replace(" ", "").replace("smc", "")
                    resall[label] = len(lineclean)
                    last_index = min(10, len(lineclean))-1
                    if '' != lineclean[:last_index].replace(" ", "").replace("\n", ""):
                        resallmemomry[label] = int(lineclean[:last_index].replace(" ", "").replace("\n", ""))
                    else:
                        resallmemomry[label] = 0
                    if len(lineclean) > resall[predict]:
                        predict = label
                    elif len(lineclean) == resall[predict]:
                        if int(resallmemomry[label]) > resallmemomry[predict]:
                            predict = label




        del cnflrfinal, cnf_general3


    print("Labels proposed : ", resall, predict)
    print(resallmemomry)

    return resall, resallmemomry, predict

"""def attackornotattack_theo2(features_replace, batchici, features1_ref,labels_np,W,b,cnf_general,args,indexicicici,
                             outputbinary_refv2, outputbinary_refv3):
    KNOWN = (features_replace[batchici, :] >= 0)
    UNKNOWN1 = (features_replace[batchici, :] < 0)

    if args.type == "real":
        UNKNOWN1 = (outputbinary_refv2[:, :, :, :, 0].reshape(-1) < 0)
        UNKNOWN2 = (outputbinary_refv2[:, :, :, :, 1].reshape(-1) < 0)
        UNKNOWN3 = (outputbinary_refv2[:, :, :, :, 2].reshape(-1) < 0)

        features2_ref = outputbinary_refv3[:, :, :, :, 1].reshape(-1)
        features3_ref = outputbinary_refv3[:, :, :, :, 2].reshape(-1)


    V_ref = np.dot(W[:, KNOWN], features_replace[batchici, KNOWN]) + b

    if args.type == "real":
        V_ref = np.expand_dims(V_ref, axis = 0)

    print("LABEL REF", labels_np[batchici], len(cnf_general))

    resall = {}
    resallmemomry = {}
    predict = 0
    predict_value = 0
    for label in range(10):
        cnf_general2 = copy.copy(cnf_general)
        cnf_general3 = cnf_general2
        cnflrfinal = CNF()
        #cnflrfinal.extend(cnf_general)
        litsici2 = features1_ref[UNKNOWN1].tolist()
        litsici2 = [int(x) for x in litsici2]
        weigth_l_1 = W[label, UNKNOWN1]
        Wf_l = 1.0 * weigth_l_1
        if args.type == "real":
            litsici2 = features1_ref[UNKNOWN1].tolist() + features2_ref[UNKNOWN2].tolist() + features3_ref[
                UNKNOWN3].tolist()
            litsici2 = [int(x) for x in litsici2]
            weigth_l_1 = 1.0 * W[labels_np[batchici], UNKNOWN1]
            weigth_l_2 = 2.0 * W[labels_np[batchici], UNKNOWN2]
            weigth_l_3 = 3.0 * W[labels_np[batchici], UNKNOWN3]
            Wf_l = 1.0 * np.concatenate((weigth_l_1, weigth_l_2, weigth_l_3))

        unique_equation = []
        unique_var = []
        weightsunique = {}
        vf = 0
        for aconcurant in range(10):
            weigth_a_1 = W[aconcurant, UNKNOWN1]
            Wf_a = 1.0 * weigth_a_1
            if args.type == "real":
                weigth_a_1 = 1.0 * W[aconcurant, UNKNOWN1]
                weigth_a_2 = 2.0 * W[aconcurant, UNKNOWN2]
                weigth_a_3 = 3.0 * W[aconcurant, UNKNOWN3]
                Wf_a = 1.0 * np.concatenate((weigth_a_1, weigth_a_2, weigth_a_3))
            if aconcurant != label:
                V = V_ref[batchici, aconcurant] - V_ref[batchici, label]
                #vf += (V)
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.tolist()
                weightsici = [int(x) for x in weightsici]

                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] !=0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])
                if len(litsici3) >0:
                    unique_equation.append([litsici3, weightsici2, int(V)-1])

                for index_litteral, litteral in enumerate(litsici3):
                    if litteral not in unique_var:
                        unique_var.append(litteral)
                        #weightsunique[litteral] = [weightsici2[index_litteral]]
                    #else:
                        #weightsunique[litteral].append(weightsici2[index_litteral])

        m = len(unique_equation)
        n = len(unique_var)
        A0 = np.zeros((n,m))
        b0 = np.zeros(m)
        for interm in range(m):
            equation = unique_equation[interm]
            b0[interm] = equation[2]
            for iter_n in range(n):
                literaloccurenceici = unique_var[iter_n]
                if literaloccurenceici in equation[0]:
                    A0[iter_n, interm] = equation[1][equation[0].index(literaloccurenceici)]



        A1 = np.identity(n)
        A2 = -1*np.identity(n)
        print(A0.shape, A1.shape, A2.shape)
        A = np.concatenate((A0, A1, A2), axis = 1).transpose()
        print(A, A.shape)
        b1 = np.ones(n)
        b2 = np.zeros(n)
        b = np.concatenate((b0, b1, b2), axis=0)
        print(b, b.shape)

        vertices = compute_polytope_vertices(A, b)
        print(vertices)
        if len(vertices)>0:
            hull = alg.qhull(np.array(vertices))
            #p = Polyhedron(vertices=vertices)
            #nbre_total = p.integral_points_count()
            integral_points = alg.enumerate_integral_points(hull).shape[1]
            print(integral_points)
            print(ok)



        #resall[label] = nbre_total
        #resallmemomry[label] = nbre_total
        #predict = 0





    print("Labels proposed : ", resall, predict)
    print(resallmemomry)

    return resall, resallmemomry, predict"""
