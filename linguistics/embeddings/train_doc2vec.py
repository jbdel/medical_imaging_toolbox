import pickle
from tqdm import tqdm
import argparse, os, random
import numpy as np
from dataloaders.MimicDataset import OldMimicDataset
from torch.utils.data import DataLoader
import gensim
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print("\rEpoch #{} end".format(self.epoch),
              end='          ')
        self.epoch += 1

    def on_training_end(self, model):
        print('\n')


class Doc2Vec:
    def __init__(self, args):
        self.args = args
        self.model = gensim.models.doc2vec.Doc2Vec(dm=0,
                                                   dbow_words=0,
                                                   vector_size=args.vector_size,
                                                   window=8,
                                                   min_count=15,
                                                   epochs=args.epochs,
                                                   workers=multiprocessing.cpu_count(),
                                                   callbacks=[EpochLogger()])
        print('Using model: ', self.model)


def train_doc2vec(net, train_loader, eval_loader, test_loader, args):
    doc2vec = net.model

    # Build corpus
    train_corpus = []
    print("Building corpus")
    os.makedirs(args.output, exist_ok=True)
    doc2vec_corps = os.path.join(args.output, 'Doc2Vec', "doc2vec_corps.pkl")
    study_id_dict = set()
    if os.path.exists(doc2vec_corps):
        train_corpus = pickle.load(open(doc2vec_corps, 'rb'))
    else:
        for i, sample in enumerate(tqdm(train_loader)):
            _, study_id, _ = sample['key']
            if int(study_id) in study_id_dict:
                continue
            report = gensim.utils.simple_preprocess(sample['report'][0])
            train_corpus.append(gensim.models.doc2vec.TaggedDocument(report, [i]))
            study_id_dict.add(int(study_id))
        pickle.dump(train_corpus, open(doc2vec_corps, 'wb'))

    # Build vocab
    doc2vec.build_vocab(train_corpus)
    print("Corpus contains " + str(len(train_corpus)) + " reports \n" +
          "Vocabulary count : " + str(len(doc2vec.wv.vocab)) + ' words \n' +
          "Corpus total words : " + str(doc2vec.corpus_total_words) + " words \n" +
          "Corpus count : " + str(doc2vec.corpus_count))
    print(len(doc2vec.docvecs))

    # Train the model
    print("Training model")
    doc2vec.train(train_corpus, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    # Save the model
    doc2vec.save(os.path.join(args.output, 'DBOW_vector' +
                              str(doc2vec.vector_size) +
                              '_window' +
                              str(doc2vec.window) +
                              '_count' +
                              str(doc2vec.vocabulary.min_count) +
                              '_epoch' +
                              str(doc2vec.epochs) +
                              '_mimic.doc2vec'))
    print("Model saved")

    # inference on train and val
    # print('Saving Doc2Vec vectors')
    # out_dir = os.path.join(args.data_root, args.vector_folder, args.name)
    # os.makedirs(out_dir, exist_ok=True)
    # for i, sample in enumerate(tqdm(chain(train_loader, eval_loader, test_loader), total=len(train_loader)
    #                                                                                      + len(eval_loader)
    #                                                                                      + len(test_loader))):
    #     report = gensim.utils.simple_preprocess(sample['report'][0])
    #     vector = doc2vec.infer_vector(report)
    #     np.save(os.path.join(
    #             out_dir,
    #             str(sample['key'][0].item()) + '-' + str(sample['key'][1].item())
    #             ), np.array(vector))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default="MimicDataset")

    # Training
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--vector_size', type=int, default=300)

    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--output', type=str, default='ckpt/')
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)

    # DataLoader
    train_dset = eval(args.dataset)('train', return_report=True)
    eval_dset = eval(args.dataset)('val', return_report=True)
    test_dset = eval(args.dataset)('test', return_report=True)

    train_loader = DataLoader(train_dset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(eval_dset,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True)

    test_loader = DataLoader(test_dset,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True)


    net = Doc2Vec(args)
    train_doc2vec(net, train_loader, eval_loader, test_loader, args)
