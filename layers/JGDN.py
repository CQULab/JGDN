import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .JGDN_Modules import *
from .loss import AlignmentContrastiveLoss
from .text import EncoderText
import copy


class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt=opt)
        self.img_feature = EncoderImage()

        # vsa feature
        self.mvsa = VSA_Module(opt=opt)

        # text feature
        # self.text_feature = Skipthoughts_Embedding_Module(
        #     vocab=vocab_words,
        #     opt=opt
        # )
        #
        self.text_feature = EncoderText(opt)
        self.fc = nn.Linear(768, 512)


        self.cross_attention_s = CrossAttention(opt=opt)
        self.cross_attention_i = CrossAttentionImg(opt=opt)

        self.SAF_modlule = AttentionFiltration(sim_dim=512)

        self.vgmf_gate = VGMF_Fusion(opt=opt)

        self.Eiters = 0


        self.alignment_criterion = AlignmentContrastiveLoss(margin=0.2,
                                                            measure='alignment',
                                                            max_violation=True,
                                                            return_similarity_mat=True)


    def forward(self, img, text, text_lens):
        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)
        # _, img_embs = self.img_feature(img)

        # mvsa featrues
        mvsa_feature = self.mvsa(lower_feature, higher_feature, solo_feature)

        # text features
        # text_feature = self.text_feature(text)
        text = text[:, :max(text_lens)]
        text_feature, cap_embs = self.text_feature(text, text_lens)
        cap_embs = self.fc(cap_embs)

        # VGMF
        Ft = self.cross_attention_s(mvsa_feature, cap_embs)
        # 80 * 80 * 27 * 512
        # SAF
        Ft = self.SAF_modlule(Ft)

        Fi = self.cross_attention_i(lower_feature, higher_feature, solo_feature, text_feature)
        Fi = Fi.permute(1, 0, 2)
        # Fi = Fi.unsqueeze(dim=0).expand(Ft.shape[0], -1, -1)
        # print(Ft.shape)
        # print(Fi.shape)

        # sim dual path
        # mvsa_feature = mvsa_feature.unsqueeze(dim=1).expand(-1, Ft.shape[1], -1)
        dual_sim = cosine_similarity(Fi, Ft)

        # regions and words similarity
        # img_lengths = [cap_embs.size(1) for _ in range(cap_embs.size(0))]


        # normalize even every vector of the set
        # img_embs = F.normalize(img_embs, p=2, dim=2)
        # cap_embs = F.normalize(cap_embs, p=2, dim=2)


        # alignment_sim = self.alignment_criterion(img_embs, cap_embs, img_lengths, text_lens)

        # sim = dual_sim + alignment_sim

        return dual_sim


def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
