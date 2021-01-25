from .VisualBertModel import VisualBertModel
from transformers import BertTokenizer, BasicTokenizer
from .MMFTokenizer import MMFTokenizer
from omegaconf import OmegaConf
from transformers.models.bert.modeling_bert import BertConfig
import torch

# Creating vocabulary from mimic reports, putting into file
vocabulary = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '.'}
reports = {}

# df_reports = pd.read_csv('mimic-cxr-reports.csv')
# for index, row in tqdm(df_reports.iterrows(), total=df_reports.shape[0]):
#   impression = str(row['impression'])
#   findings = str(row['findings'])
#   study = row['study']
#   # save report
#   report = (impression if impression != "nan" else '') + ' ' + (findings if findings != "nan" else '')
#   reports[study] = report

report_tokenizer = BasicTokenizer(do_lower_case=True)
report = 'No evidence of consolidation to suggest pneumonia is seen. ' \
         'There is some retrocardiac atelectasis. A small left pleural ' \
         'effusion may be present. No pneumothorax is seen. No pulmonary edema. ' \
         'A right granuloma is unchanged. The heart is mildly enlarged, unchanged. There is tortuosity of the aorta.'

# Add word to vocabulary
words = report_tokenizer.tokenize(report)
vocabulary.update(set(words))

print("Vocabulary size is ", len(vocabulary))
with open("vocabulary.txt", "w+") as f:
    for w in vocabulary:
        f.write(str(w) + '\n')

import matplotlib.pyplot as plt

params = {'tokenizer_config': {'type': 'bert-base-uncased',
                               'params': {'do_lower_case': True}},
          'mask_probability': 0,
          'max_seq_length': 128}
mmf_tok = MMFTokenizer(OmegaConf.create(params))
mmf_tok._tokenizer = BertTokenizer(vocab_file="vocabulary.txt")

config = BertConfig.from_pretrained('bert-large-uncased',
                                    num_labels=2,
                                    vocab_size=len(vocabulary),
                                    num_hidden_layers=3)
net = VisualBertModel(config, visual_embedding_dim=2048).cuda()

out_txt = mmf_tok({'text': report})

input_ids = torch.tensor(out_txt['input_ids']).unsqueeze(0)
input_mask = torch.tensor(out_txt['input_mask']).unsqueeze(0)
img = torch.zeros(1, 14, 2048)

out = net(
    input_ids=input_ids.cuda(),
    text_mask=input_mask.cuda(),
    visual_embeddings=img.cuda()
)

print(net.config.add_cross_attention)
print(out.keys())
print(len(out['attentionMap']))
print(out['attentionMap'][0].shape)
