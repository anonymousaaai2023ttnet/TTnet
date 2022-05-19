import copy
import warnings
import numpy as np
import torch
from tqdm import tqdm
from src.nn.evaluateur import Evaluateur
from src.nn.load_models import load_models_binary, load_models_real, load_data, load_models_binary_prunnig
from src.utils.utils import concat, get_res_numpybloc, eval_model_general_binary, infer_1_block_sat, \
    infer_and_replace_1_block_sat, get_refs_all, get_input_noise, infer_1_block_sat_verify_vitesse, get_reference1_2_3, \
    attackornotattack, verification_vitess, get_dictionnaire_ref, infer_1_block_sat_verify, verification_total, \
    evaluatenormalement, infer_and_replace_1_block_sat_robuste, evaluation_robuste, update_args, \
    eval_model_general_real, verification_total_maxsat, verification_vitessmaxsat, verification_vitesstheo, verification_vitesstheo2
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
from src.utils.config import Config
from src.utils.config import str2bool, two_args_str_int, two_args_str_float, str2list, \
    transform_input_filters, transform_input_lr, transform_input_eps



config_general = Config(path="config/")
config_general.dataset="CIFAR10"
config = Config(path="config/cifar10/")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config_general.dataset)
parser.add_argument("--type", default=config_general.type)

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cuda", "cpu"])
parser.add_argument("--device_ids", default=config.general.device_ids, type=str2list)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--num_workers", default=config.general.num_workers, type=int)

parser.add_argument("--filters", default=config.model.filters, type=transform_input_filters)
parser.add_argument("--kernelsizes", default=config.model.kernelsizes, type=transform_input_filters)
parser.add_argument("--amplifications", default=config.model.amplifications, type=transform_input_filters)
parser.add_argument("--strides", default=config.model.strides, type=transform_input_filters)
parser.add_argument("--fc", default=config.model.fc, type=int)
parser.add_argument("--nchannel", default=config.model.nchannel, type=int)
parser.add_argument("--groups", default=config.model.groups, type=transform_input_filters)
parser.add_argument("--g_remove_last_bn", default=config.model.g_remove_last_bn)
parser.add_argument("--step_quantization", default=config.model.step_quantization, type=transform_input_eps)

parser.add_argument("--adv_epsilon", default=config.train.adv_epsilon)
parser.add_argument("--batch_size_train", default=config.train.batch_size_train, type=int)
parser.add_argument("--n_epoch", default=config.train.n_epoch, type=two_args_str_int)
parser.add_argument("--loss_type", default=config.train.loss_type, type=two_args_str_int)
parser.add_argument("--optimizer_type", default=config.train.optimizer_type, type=two_args_str_int)
parser.add_argument("--weight_decay", default=config.train.weight_decay, type=two_args_str_float)
parser.add_argument("--lr", default=config.train.lr, type=transform_input_lr)
parser.add_argument("--epochs_lr", default=config.train.epochs_lr, type=transform_input_lr)
parser.add_argument("--clip_grad_norm", default=config.train.clip_grad_norm, type=two_args_str_float)
parser.add_argument("--a_bit_final", default=config.train.a_bit_final, type=two_args_str_float)
parser.add_argument("--l1_coef", default=config.train.l1_coef, type=two_args_str_float)
parser.add_argument("--l1_reg", default=config.train.l1_reg, type=str2bool)

parser.add_argument("--batch_size_test", default=config.eval.batch_size_test, type=two_args_str_int)
parser.add_argument("--pruning", default=config.eval.pruning, type=str2bool)
parser.add_argument("--coef_mul", default=config.eval.coef_mul, type=two_args_str_int)

parser.add_argument("--path_exp", default=config.get_exp.path_exp)
parser.add_argument("--filter_occurence", default=config.get_exp.filter_occurence, type=two_args_str_int)
parser.add_argument("--proportion", default=config.get_exp.proportion, type=two_args_str_float)
parser.add_argument("--proba", default=config.get_exp.proba, type=two_args_str_float)
parser.add_argument("--filtre_exp", default=config.get_exp.filtre_exp, type=str2bool)

parser.add_argument("--modeltoeval", default=config.eval_with_sat.modeltoeval)
parser.add_argument("--mode_eval", default=config.eval_with_sat.mode_eval)
parser.add_argument("--attack_eps_ici", default=config.eval_with_sat.attack_eps_ici, type=two_args_str_float)
parser.add_argument("--number_ici", default=config.eval_with_sat.number_ici, type=two_args_str_int)
parser.add_argument("--encoding_type", default=config.eval_with_sat.encoding_type, type=two_args_str_int)
parser.add_argument("--coef", default=config.eval_with_sat.coef, type=two_args_str_int)
parser.add_argument("--quant_noise", default=config.eval_with_sat.quant_noise, type=two_args_str_int)
parser.add_argument("--coef_multiply_equation_noise", default=config.eval_with_sat.coef_multiply_equation_noise, type=two_args_str_int)
parser.add_argument("--type_norm_noise", default=config.eval_with_sat.type_norm_noise)
parser.add_argument("--ratio_distrib_maxsat", default=config.eval_with_sat.ratio_distrib_maxsat,type=two_args_str_int)

parser.add_argument("--sat_solver", default=config.solve_sat_formulas_per_img.sat_solver)
parser.add_argument("--time_out", default=config.solve_sat_formulas_per_img.time_out, type=two_args_str_int)
#

args = parser.parse_args()

path_save_model = args.path_exp
path_save_model_prunning = args.path_exp + "prune/"

if config_general.dataset=="CIFAR10":
    args.attack_eps_ici = args.attack_eps_ici / 255

if "cifar10_high_noise" in args.path_exp:
    args.strides = [3,2]


device = torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")


if args.type == "binary":
    args = update_args(args, path_save_model, device)


print("-"*100)
print()
print("START EVLUATION WITH INT WEIGTHS")
print()

dataloaders, testset = load_data(args)
evaluateur = Evaluateur(dataloaders, args, device)

if args.type == "binary":
    quantized_model_train, feature_pos = load_models_binary(args, path_save_model, device)
    totwfin = 0
    for layer, module in enumerate(quantized_model_train.features):
        try:
            totw = 1
            for x in module.weight.shape:
                totw = totw * x
        except:
            pass
        totwfin += totw
    print(totwfin)
    if args.modeltoeval == "prune" or args.modeltoeval == "prune_filtered":
        netprunned = load_models_binary_prunnig(args, quantized_model_train, path_save_model_prunning+"last.pth", device, feature_pos, evaluateur)
    else:
        acc = evaluateur.eval(quantized_model_train, ["val"])
elif args.type == "real":
    quantized_model_train = load_models_real(args, path_save_model, device)
    feature_pos = None
    if args.modeltoeval == "prune" or args.modeltoeval == "prune_filtered":

        netprunned = load_models_binary_prunnig(args, quantized_model_train, path_save_model_prunning + "last.pth", device,
                                            feature_pos, evaluateur)


print(quantized_model_train)

quantized_model_train.eval()

reference1, reference2, reference3 = get_reference1_2_3()


if args.modeltoeval == "normal":
    path = path_save_model
elif args.modeltoeval == "filtered":
    path = path_save_model + "filtered/"
elif args.modeltoeval == "prune":
    path = path_save_model_prunning
    del quantized_model_train
    quantized_model_train = netprunned
elif args.modeltoeval == "prune_filtered":
    path = path_save_model + "prune_filtered/"
    del quantized_model_train
    quantized_model_train = netprunned

if args.type == "binary":
    W = 1.0*quantized_model_train.features[feature_pos - 1].weight.detach().cpu().clone().numpy()
    # print(model_train.features[feature_pos - 1].bias.data.shape)
    b = 1.0*quantized_model_train.features[feature_pos - 1].bias.detach().cpu().clone().numpy()
elif args.type == "real":
    W = quantized_model_train.fc.weight.detach().cpu().clone().numpy()
    W1 = 1.0 * W
    W2 = 2.0 * W
    W3 = 3.0 * W
    b = 0


res_numpyblocall = {}
for numblockici in range(len(args.filters)):
    res_numpyblocall[numblockici] = get_res_numpybloc(args, path, num = numblockici)




with torch.no_grad():
    img = testset[0][0].unsqueeze(0)
    img_refsizex = testset[0][0].shape[-1]
    if args.type == "binary":
        res, input_binary, all_block_acomplter, features_ref, all_inputbshape = eval_model_general_binary(quantized_model_train.eval(), img.to(device), len(args.filters))
    elif args.type == "real":
        res, input_binary, all_block_acomplter, features_ref, all_inputbshape = eval_model_general_real(quantized_model_train.eval(), img.to(device), len(args.filters))


    img_ref, block_refcall, img_ref_inverse = get_refs_all(args, img_refsizex, all_inputbshape)
    features1_ref = block_refcall[len(args.filters)-1][:, :, :, :, 0].reshape(-1)



unfoldball = {}
for numblockici in range(len(args.filters)):
    unfoldball[numblockici] = torch.nn.Unfold(kernel_size=args.kernelsizes[numblockici], stride=args.strides[numblockici])




dictionnaire_ref = get_dictionnaire_ref(args, img_ref, unfoldball, block_refcall)








del dataloaders
args.batch_size_test = 1
dataloaders, testset = load_data(args)

quantized_model_train.eval()
total = 0
timeout = 0
correct = 0
batchici = 0
nbre_clausetot, nbre_varstot, padvtot = [], [], []
coefintchannel = int(args.filters[0]/args.nchannel)
with torch.no_grad():
    print("START INFER FORMULA")
    tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))
    for indexicicici, (images, labels) in enumerate(tk0):
        if int(args.coef_mul)*(int(args.number_ici)+1)> indexicicici >=int(args.coef_mul)*int(args.number_ici):
            total += 1
            labels_np = labels.numpy()

            if args.mode_eval == "evaluation":
                correct, timeout = evaluatenormalement(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                    all_inputbshape,
                                    res_numpyblocall, W, b, device)

            elif args.mode_eval == "evaluation_maxsat":
                correct, timeout = verification_total_maxsat(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                   all_inputbshape,
                                   res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                                   indexicicici,
                                   device, img_ref, img_refsizex)

            elif args.mode_eval =="verification_vitesse":
                correct, timeout, nbre_clause, nbre_vars = verification_vitess(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                all_inputbshape,
                                res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                                indexicicici, device, img_ref_inverse)
                if nbre_clause != 0 and nbre_vars != 0:
                    nbre_clausetot.append(nbre_clause)
                    nbre_varstot.append(nbre_vars)
                #print(nbre_clause, nbre_vars)

            elif args.mode_eval =="verification":
                correct, timeout = verification_total(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                   all_inputbshape,
                                   res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                                   indexicicici,
                                   device, img_ref, img_refsizex, img_ref_inverse)

            elif args.mode_eval =="verification_maxsat":
                if args.ratio_distrib_maxsat >0:
                    correctinit = copy.copy(correct)
                    for k in range(3):
                        correct, timeout = verification_vitessmaxsat(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                        all_inputbshape,
                                        res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                                        indexicicici, device, img_ref_inverse)
                    if correct - correctinit >int(3/2):
                        correct = correctinit+1
                else:
                    correct, timeout = verification_vitessmaxsat(quantized_model_train, images.to(device), args,
                                                                 batchici, labels_np, correct, timeout, unfoldball,
                                                                 all_inputbshape,
                                                                 res_numpyblocall, dictionnaire_ref, block_refcall,
                                                                 reference2, features1_ref, W, b,
                                                                 indexicicici, device, img_ref_inverse)


            elif args.mode_eval =="verification_maxsat2":
                correct, timeout, padv, nbre_clause, nbre_vars = verification_vitesstheo(quantized_model_train, images.to(device), args, batchici, labels_np, correct, timeout, unfoldball,
                                all_inputbshape,
                                res_numpyblocall, dictionnaire_ref, block_refcall, reference2, features1_ref, W, b,
                                indexicicici, device, img_ref_inverse)
                if nbre_clause!=0 and nbre_vars!=0:
                    nbre_clausetot.append(nbre_clause)
                    nbre_varstot.append(nbre_vars)
                if padv is not None:
                    padvtot.append(padv)
                #print(padv, nbre_clause, nbre_vars)

        elif int(args.coef_mul) * (int(args.number_ici) + 1) < indexicicici:
            print("Number of sample verified: ", correct, "Number of time-out: ", timeout, "Number of clauses per sample: ", np.mean(nbre_clausetot),  "Number of variables per sample: ",np.mean(nbre_varstot))
            if len(padvtot)>0:
                print("P(adv)", np.mean(padvtot))
            break
#print(correct, timeout)

print()
print()

namelog = args.path_exp + "ACC_"+str(args.number_ici) + '_'+str(args.coef_mul) +  '_'+str(args.type)+'_'+str(args.attack_eps_ici) + '_'+str(args.modeltoeval)+'_'+str(args.mode_eval) + '_' + str(args.ratio_distrib_maxsat) + '_accuracy_final.txt'

f = open(str(namelog), "w")
f.write(str(correct))
f.write('\n')
f.write(str(timeout))
f.write('\n')
f.write(str(int(np.floor(np.mean(nbre_clausetot)))))
f.write('\n')
f.write(str(int(np.floor(np.mean(nbre_varstot)))))
f.write('\n')
if len(padvtot) > 0:
    f.write(str((np.mean(padvtot))))
f.close
#break
ok




