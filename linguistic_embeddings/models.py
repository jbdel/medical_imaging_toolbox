import sent2vec
import os
from transformers import AutoTokenizer, AutoModel
from nltk import sent_tokenize, word_tokenize
import gensim
from gensim.models import Doc2Vec


# get_model returns model function and tokenizer function that receive:
# report = tokenizer(sample['report'])
# vector = model(report)
def get_model(args):
    if args.model == "Bio_ClinicalBERT":
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", return_dict=True)
        return lambda x: model(**x).pooler_output.cpu().data.numpy().squeeze(0), \
               lambda x: tokenizer(x, return_tensors="pt")

    elif args.model == "BlueBERT":
        tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16")
        model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16", return_dict=True)
        return lambda x: model(**x).pooler_output.cpu().data.numpy().squeeze(0), \
               lambda x: tokenizer(x, return_tensors="pt")

    elif args.model == "BioSentVec":
        ## uncertain ?
        model = sent2vec.Sent2vecModel()
        model.load_model(os.path.join(args.output, 'BioSentVec', 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'))
        return lambda x: model.embed_sentence(x).squeeze(0), \
               lambda x: ' '.join(word_tokenize(x))

    elif args.model == "Doc2Vec":
        model = Doc2Vec.load(
            os.path.join(args.output, 'Doc2Vec', args.doc2vec_model))
        return lambda x: model.infer_vector(x), \
               lambda x: gensim.utils.simple_preprocess(x)
    else:
        raise NotImplementedError
