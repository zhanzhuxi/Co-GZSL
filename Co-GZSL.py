from __future__ import print_function
import argparse
import sys

from logger import logger

sys.path.append("..")
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier

import model as model
import losses
from default import _C as cfg
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument(
        "--config-file",
        default="./Co_para/awa2_4w_2s.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

opt = parser.parse_args()

cfg.merge_from_file(opt.config_file)
opt = cfg
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            h, output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


for s in range(1):
    print(opt)
    data = util.DATA_LOADER(opt)
    logger.info("# of training samples: " + str(data.ntrain))

    netG = model.MLP_G(opt)
    netMap = model.Embedding_Net(opt)
    netD = model.MLP_CRITIC(opt)
    netDec = model.AttDec(opt)

    model_path = './models/' + opt.dataset
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if len(opt.gpus.split(',')) > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        netMap = nn.DataParallel(netMap)
        netDec = nn.DataParallel(netDec)

    contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
    input_label = torch.LongTensor(opt.batch_size)

    if opt.cuda:
        netG.cuda()
        netD.cuda()
        netMap.cuda()
        netDec.cuda()
        input_res = input_res.cuda()
        noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
        input_label = input_label.cuda()

    import itertools

    optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), netDec.parameters()), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    best_performance = [0, 0, 0, 0]
    best_epoch = -1
    output_dir = opt.OUTPUT_DIR + "/" + opt.dataset
    model_file_name = opt.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    for epoch in range(opt.nepoch):
        FP = 0
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):
            for p in netD.parameters():
                p.requires_grad = True
            for p in netMap.parameters():
                p.requires_grad = True
            for p in netDec.parameters():
                p.requires_grad = True

            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                netMap.zero_grad()

                sparse_real = opt.resSize - input_res[1].gt(0).sum()
                embed_real, outz_real = netMap(input_res)
                criticD_real = netD(input_res, input_att)
                criticD_real = criticD_real.mean()

                mu_real, var_real, att_dec_real = netDec(embed_real)
                real_att_contras_loss = contras_criterion(att_dec_real, input_label)
                cos_dist = torch.einsum('bd,nd->bn', att_dec_real, input_att)
                CLS_loss = nn.CrossEntropyLoss()
                Latt_real = CLS_loss(cos_dist, input_label)
                noise_gen.normal_(0, 1)
                fake_gem, fake = netG(noise_gen, input_att)
                fake_norm = fake.data[0].norm()
                sparse_fake = fake.data[0].eq(0).sum()
                criticD_fake = netD(fake.detach(), input_att)
                criticD_fake = criticD_fake.mean()
                embed_fake, outz_fake = netMap(fake)
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty + real_att_contras_loss + Latt_real
                D_cost.backward()
                optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False
            for p in netMap.parameters():
                p.requires_grad = False
            for p in netDec.parameters():
                p.requires_grad = False

            netG.zero_grad()
            noise_gen.normal_(0, 1)
            fake_gem, fake = netG(noise_gen, input_att)
            embed_fake, outz_fake = netMap(fake)
            criticG_fake = netD(fake, input_att)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            embed_real, outz_real = netMap(input_res)
            all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)
            all_embedz = torch.cat((embed_fake, embed_real.detach()), dim=0)
            mu_fake, var_fake, att_dec_fake = netDec(all_embedz)
            fake_att_contras_loss = contras_criterion(att_dec_fake, torch.cat((input_label, input_label), dim=0))
            cos_dist = torch.einsum('bd,nd->bn', att_dec_fake, torch.cat((input_att, input_att), dim=0))
            CLS_loss = nn.CrossEntropyLoss()
            Latt_fake = CLS_loss(cos_dist, torch.cat((input_label, input_label), dim=0))
            errG = G_cost + opt.cls_weight * (fake_att_contras_loss + Latt_fake)
            errG.backward()
            optimizerG.step()

        netDec.zero_grad()
        if (epoch + 1) % opt.lr_decay_epoch == 0:
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

        mean_lossG /= data.ntrain / opt.batch_size
        mean_lossD /= data.ntrain / opt.batch_size

        logger.info('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_att_contras_loss: %.4f, '
                    'fake_att_contras_loss: %.4f, Latt_real: %.4f, Latt_fake: %.4f' % (epoch, opt.nepoch, D_cost,
                                                                                       G_cost, Wasserstein_D,
                                                                                       real_att_contras_loss,
                                                                                       fake_att_contras_loss,
                                                                                       Latt_real, Latt_fake))

        netG.eval()

        for p in netMap.parameters():
            p.requires_grad = False
        for p in netDec.parameters():
            p.requires_grad = False

        if opt.gzsl:  # Generalized zero-shot learning
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)

            nclass = opt.nclass_all

            cls = classifier.CLASSIFIER(train_X, train_Y, netMap, opt.embedSize, data, nclass, opt.cuda,
                                        opt.classifier_lr, 0.5, 25, opt.syn_num,
                                        True)
            logger.info('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            if cls.H > best_performance[-1]:
                best_epoch = epoch + 1
                best_performance[1:] = [best_epoch, cls.acc_seen, cls.acc_unseen, cls.H]
                txt_file = './models/' + opt.dataset + '/' + str(s)
                file = open(txt_file, 'w')
                file.write(str(best_performance))
                torch.save(netMap.state_dict(), './models/' + opt.dataset + '/' + '{}_netMap_best.pth'.format(opt.dataset))
                torch.save(netG.state_dict(), './models/' + opt.dataset + '/' + '{}_netG_best.pth'.format(opt.dataset))
                torch.save({
                    'epoch': epoch,
                    'netMap_state_dict': netMap.state_dict(),
                    'netG_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict()
                }, './models/' + opt.dataset + '/' + '{}_model_best.pth'.format(opt.dataset))

        else:  # conventional zero-shot learning
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                        netMap,
                                        opt.embedSize, data,
                                        data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 100,
                                        opt.syn_num,
                                        False)
            acc = cls.acc
            logger.info('unseen class accuracy=%.4f ' % acc)

        netG.train()
        for p in netMap.parameters():
            p.requires_grad = True
        for p in netDec.parameters():
            p.requires_grad = True

    logger.info("full process finished! Everything is OK!")

