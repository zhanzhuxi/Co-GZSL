import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def reparameter(mu, sigma):
    return (torch.randn_like(mu) * sigma) + mu


class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        # self.fc2 = nn.Linear(opt.embedSize, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding, out_z


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        n = h.shape[0]
        g = h.view(n, 128, 4, 4)
        return g, h


class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.attSize+opt.attSize, opt.nhF)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h


class Dis_Embed_Att_(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att_, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h


class MLP_CRITIC_DEC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC_DEC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class AttDec_(nn.Module):
    def __init__(self, opt):
        super(AttDec_, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.attSize)
        self.hidden = None
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, feat):
        h = self.lrelu(self.fc1(feat))
        self.hidden = h
        h = self.fc2(h)
        h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return h

    def getLayersOutDet(self):
        return self.hidden.detach()


class AttDec(nn.Module):
    def __init__(self, opt):
        super(AttDec, self).__init__()
        self.hidden = None
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.embedSize, opt.nghA)
        self.fc3 = nn.Linear(opt.nghA, opt.attSize * 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, feat):
        h = feat
        h = self.lrelu(self.fc1(h))
        self.hidden = h
        h = self.fc3(h)
        mus, stds = h[:, :self.attSize], h[:, self.attSize:]
        stds = self.sigmoid(stds)
        h = reparameter(mus, stds)
        mus = F.normalize(mus, dim=1)
        h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return mus, stds, h

    def getLayersOutDet(self):
        return self.hidden.detach()

