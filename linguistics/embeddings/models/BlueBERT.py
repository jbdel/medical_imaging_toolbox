import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from linguistics.embeddings.utils import get_report


class BlueBERT(nn.Module):
    def __init__(self, cfg):
        super(BlueBERT, self).__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16")
        self.model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16", return_dict=True)

    def forward(self, sample):
        inp = get_report(sample['report'], policy=self.cfg.report.report_policy)
        inp = self.tokenizer(inp, return_tensors="pt")
        return self.model(**inp).pooler_output.cpu().data.numpy().squeeze(0)
