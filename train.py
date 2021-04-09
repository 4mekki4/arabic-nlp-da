import argparse
import os
import os.path as osp


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd 

import network
import loss

import random
from Coral import CORAL
from mmd import  MMD_loss
import Dataset
from sklearn.metrics import classification_report
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sentence_classification_test(loader, model):
    start_test = True
    dataset = loader
    with torch.no_grad():
        iter_test = iter(dataset)
        for i in range(len(dataset)):
            tinput_ids, tattention_mask, labels = iter_test.next()
            tinput_ids, tattention_mask = tinput_ids.to(device), tattention_mask.to(device)
            feature, outputs = model(input_ids=tinput_ids, attention_mask=tattention_mask)
            outputs = nn.Softmax(dim=1)(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    report = classification_report(all_label.numpy(), torch.squeeze(predict).float().numpy(), target_names=["Negative", "Positive"])
    return accuracy, report, torch.squeeze(predict).float().numpy()



def train(config, source_train_loader, target_train_loader, target_test_loader, source_test_loader):

    class_num = config["class_num"]

    ## set base network

    base_network = network.BertClassifier(pretrained_path=config['pretrained_path'])
    base_network = base_network.to(device)

    parameter_list = [{"params": base_network.parameters()}]
    ## add additional network for some methods
    if "ALDA" in args.method:
        ad_net = network.Multi_AdversarialNetwork(base_network.output_num(), 768, class_num, max_iter=config["num_iterations"])
        ad_net = ad_net.to(device)
        parameter_list.extend([{"params": ad_net.parameters(), 'lr': config["lr"] * config["lr_mult"]}])
    elif "DANN" in  args.method:
        ad_net = network.AdversarialNetwork(base_network.output_num(), 768, max_iter=config["num_iterations"])
        ad_net = ad_net.to(device)
        parameter_list.extend([{"params": ad_net.parameters(), 'lr': config["lr"] * config["lr_mult"]}])
    elif "MMD" in args.method:
        ad_net = MMD_loss()
        ad_net = ad_net.to(device)
        parameter_list.extend([{"params": ad_net.parameters()}])

    ## set optimizer

    optimizer = AdamW(parameter_list, lr=config["lr"], weight_decay=0.005, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=config["num_iterations"])

    loss_params = config["loss"]
    high = loss_params["trade_off"]
    begin_label = False

    ## train
    len_train_source = len(source_train_loader)
    len_train_target = len(target_train_loader)
    #transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    loss_value = 0
    loss_adv_value = 0
    loss_correct_value = 0
    best_labels = None
    for i in tqdm(range(config["num_iterations"])):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            if "indomain" in config["method"]:
                temp_acc, report, labels_p = sentence_classification_test(source_test_loader, base_network)
            else:
                temp_acc, report, labels_p = sentence_classification_test(target_test_loader, base_network)

            if temp_acc > best_acc:
                best_labels = labels_p
                best_step = i
                best_acc = temp_acc
                torch.save(base_network.state_dict(), "./ckpts/best_model.pth")

                print("\n##########     save the best model.    #############\n")
                log_str = "iter: {:05d}, accuracy: {:.5f}".format(best_step, best_acc)
                print(log_str)
            log_str = "iter: {:05d}, accuracy: {:.5f}".format(i, temp_acc)
            print(log_str)
            loss_value = 0
            loss_adv_value = 0
            loss_correct_value = 0


        if i > config["stop_step"]:
            break

        ## train one iter
        base_network.train(True)
        if args.method in["ALDA","DANN", "MMD"]:
            ad_net.train(True)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(source_train_loader)
        if i % len_train_target == 0:
            iter_target = iter(target_train_loader)
        sinput_ids, sattention_mask, labels_source = iter_source.next()
        sinput_ids, sattention_mask, labels_source = sinput_ids.to(device), sattention_mask.to(device), labels_source.to(device)
        tinput_ids, tattention_mask, label_target = iter_target.next()
        tinput_ids, tattention_mask = tinput_ids.to(device), tattention_mask.to(device)

        if sinput_ids.size(0) != tinput_ids.size(0):
            continue

        features_source, outputs_source = base_network(input_ids=sinput_ids, attention_mask=sattention_mask)
        if args.source_detach:
            features_source = features_source.detach()
        features_target, outputs_target = base_network(input_ids=tinput_ids, attention_mask=tattention_mask)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        loss_params["trade_off"] = network.calc_coeff(i, high=high, max_iter=config["num_iterations"])  # if i > 500 else 0.0
        transfer_loss = 0.0
        if 'DANN' in config['method']:
            transfer_loss = loss.DANN(features, ad_net)
        elif "MMD" in config['method']:
            transfer_loss = ad_net(features_source, features_target)
        elif "CORAL" in config['method'] :
            transfer_loss = CORAL(features_source, features_target)
        elif "ALDA" in config['method']:
            ad_out = ad_net(features)
            adv_loss, reg_loss, correct_loss = loss.ALDA_loss(ad_out, labels_source, softmax_out,
                                                              weight_type=config['args'].weight_type,
                                                              threshold=config['threshold'])
            # whether add the corrected self-training loss
            if "nocorrect" in config['args'].loss_type:
                transfer_loss = adv_loss
            else:
                transfer_loss = config['args'].adv_weight * adv_loss + config['args'].adv_weight * loss_params[
                    "trade_off"] * correct_loss
            # reg_loss is only backward to the discriminator
            if "noreg" not in config['args'].loss_type:
                for param in base_network.parameters():
                    param.requires_grad = False
                reg_loss.backward(retain_graph=True)
                for param in base_network.parameters():
                    param.requires_grad = True
        # on-line self-training
        elif 'SelfTraining' in config['method']:
            transfer_loss += loss_params["trade_off"] * loss.SelfTraining_loss(outputs, softmax_out,
                                                                               config['threshold'])
        # off-line self-training
        elif 'PseudoLabel' in config['method']:
            labels_target = labels_target.to(device)
            if begin_label:
                transfer_loss += loss_params["trade_off"] * nn.CrossEntropyLoss(ignore_index=-1)(outputs_target,
                                                                                                 labels_target)
            else:
                transfer_loss += 0.0 * nn.CrossEntropyLoss(ignore_index=-1)(outputs_target, labels_target)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        loss_value += classifier_loss.item() / config["test_interval"]
        if "ALDA" in config['method']:
            loss_adv_value += adv_loss.item() / config["test_interval"]
            loss_correct_value += correct_loss.item() / config["test_interval"]
        if config['method'] not in ["indomain", "zeroshot"]:
            total_loss = classifier_loss + transfer_loss
        else:
            total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    return best_acc, best_labels


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='ALDA')
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--restore_dir', type=str, default=None,
                        help="restore directory of our model (in ../snapshot directory)")
    parser.add_argument('--lm_pretrained', type=str, default='arabert',
                        help=" path of pretrained transformer")
    parser.add_argument('--source', type=str, default='TEAD_MSA',
                        help=" path of pretrained transformer")
    parser.add_argument('--target', type=str, default='BRAD_LEV',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_mult', type=float, default=10, help="dicriminator learning rate multiplier")
    parser.add_argument('--trade_off', type=float, default=1.0,
                        help="trade off between supervised loss and self-training loss")
    parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
    parser.add_argument('--cos_dist', type=str2bool, default=False, help="the classifier uses cosine similarity.")
    parser.add_argument('--threshold', default=0.9, type=float, help="threshold of pseudo labels")
    parser.add_argument('--label_interval', type=int, default=200, help="interval of two continuous pseudo label phase")
    parser.add_argument('--stop_step', type=int, default=0, help="stop steps")
    parser.add_argument('--final_log', type=str, default=None, help="final_log file")
    parser.add_argument('--weight_type', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--test_10crop', type=str2bool, default=True)
    parser.add_argument('--adv_weight', type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument('--source_detach', default=False, type=str2bool,
                        help="detach source feature from the adversarial learning")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    # train config
    config = {}
    config['args'] = args
    config['method'] = args.method
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["class_num"] = 2
    config["lr"] = args.lr
    config["lr_mult"] = args.lr_mult
    dosegmentation = False

    if args.lm_pretrained == 'gigabert':
        config['pretrained_path'] = '../GigaBERT-v4'

    elif args.lm_pretrained == 'dialbert':
        config['pretrained_path'] = 'bashar-talafha/multi-dialect-bert-base-arabic'
    elif args.lm_pretrained == 'arabert':
        config['pretrained_path'] = 'aubmindlab/bert-base-arabert'
        dosegmentation = True
    else:
        config['pretrained_path'] = 'bert-base-multilingual-cased'

    config["loss"] = {"trade_off": args.trade_off}
    config["threshold"] = args.threshold


    if args.stop_step == 0:
        config["stop_step"] = 100000
    else:
        config["stop_step"] = args.stop_step
    
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sourceTask = args.source
    targetTask = args.target

    source_train_loader, source_test_loader = Dataset.loadData(sourceTask, args.batch_size, 1,
                                                          dosegmentation, config['pretrained_path'],split=1000,ttype='source')

    target_train_loader, target_test_loader = Dataset.loadData(targetTask, args.batch_size, 1,
                                          dosegmentation, config['pretrained_path'], split=1000,ttype='target')
    config["test_interval"] = max(len(target_train_loader), len(target_train_loader))
    config["num_iterations"] = max(len(target_train_loader), len(source_train_loader)) * args.epochs

    acc, labels_p = train(config, source_train_loader, target_train_loader, target_test_loader, source_test_loader)
    print("Task : {}--{} accuracy : {}".format(sourceTask, targetTask, acc))