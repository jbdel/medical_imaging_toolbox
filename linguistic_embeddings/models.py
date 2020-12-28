import sent2vec
import os
import nltk
import torch
import gensim

from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModel
from gensim.models import Doc2Vec
from dataloaders import *
from classifier import get_model
import torch.nn as nn


def get_foward_function(args):
    """
    Defined a forward function (a lambda function) that receives a 'x' from dataloader.
     'x' is a dict of the following form
    {'idx': idx,
    'key': key,
    'report': report,
    'img': img,
    'label': label}
    The lambda function must return a vector representation of input
    """

    if args.model == "Bio_ClinicalBERT":
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", return_dict=True)
        return lambda x: model(
            **tokenizer(x['report'], return_tensors="pt")
        ).pooler_output.cpu().data.numpy().squeeze(0)

    elif args.model == "BlueBERT":
        tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16")
        model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16", return_dict=True)
        return lambda x: model(
            **tokenizer(x['report'], return_tensors="pt")
        ).pooler_output.cpu().data.numpy().squeeze(0)

    elif args.model == "BioSentVec":
        # Uncertain ?
        model = sent2vec.Sent2vecModel()
        ckpt = os.path.join(args.model_dir, 'BioSentVec', 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
        assert os.path.exists(ckpt), 'Please download BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
        model.load_model(ckpt)
        return lambda x: model.embed_sentence(' '.join(word_tokenize(x['report']))).squeeze(0)

    elif args.model == "Doc2Vec":
        model = Doc2Vec.load(
            os.path.join(args.model_dir, 'Doc2Vec', args.doc2vec_model))
        return lambda x: model.infer_vector(gensim.utils.simple_preprocess(x['report']))

    elif args.model == "CNN":
        ckpt = torch.load(args.cnn_model)
        args = ckpt['args']
        net = get_model(args, pretrained=False)
        network_name = type(net.net).__name__
        if 'densenet' in network_name.lower():
            fc_name = 'classifier'
        elif 'resnet' in network_name.lower():
            fc_name = 'fc'
        else:
            raise NotImplementedError

        net = nn.DataParallel(net)
        net.load_state_dict(ckpt["state_dict"])
        net.cuda()
        net.eval()
        fc = getattr(net.module.net, fc_name)
        in_features = fc.in_features

        def ret_repr(x):
            repr = torch.zeros(in_features)
            def hook(m, i, o): repr.copy_(i[0].squeeze().data)
            fc.register_forward_hook(hook)
            x['img'] = x['img'].unsqueeze(0)
            _ = net(x)
            return repr.cpu().data.numpy()

        return ret_repr

    else:
        raise NotImplementedError
