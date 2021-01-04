import torch
import json
from argparse import Namespace
from models.r2gen import R2GenModel
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from tqdm import tqdm
from collections import defaultdict

args = Namespace(amsgrad=True, ann_path='data/mimic_cxr/annotation.json',
                 batch_size=16,
                 beam_size=3,
                 block_trigrams=1, bos_idx=0, d_ff=512, d_model=512, d_vf=2048, dataset_name='mimic_cxr',
                 decoding_constraint=0, drop_prob_lm=0.5, dropout=0.1, early_stop=50, eos_idx=0, epochs=30, gamma=0.8,
                 group_size=1, image_dir='data/mimic_cxr/images/', logit_layers=1, lr_ed=0.0001, lr_scheduler='StepLR',
                 lr_ve=5e-05, max_seq_length=100, monitor_metric='BLEU_4', monitor_mode='max', n_gpu=1, num_heads=8,
                 num_layers=3, num_workers=2, optim='Adam', output_logsoftmax=1, pad_idx=0, record_dir='records/',
                 resume=None, rm_d_model=512, rm_num_heads=8, rm_num_slots=3, sample_method='beam_search', sample_n=1,
                 save_dir='results/mimic_cxr', save_period=1, seed=456789, step_size=1, temperature=1.0, threshold=10,
                 use_bn=0, visual_extractor='resnet101', visual_extractor_pretrained=True, weight_decay=5e-05)

ckpt = torch.load('model_mimic_cxr.pth')
tokenizer = Tokenizer(args)
model = R2GenModel(args, tokenizer)
model.load_state_dict(ckpt['state_dict'])
model.cuda()

train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, user_inference=True)
val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False, user_inference=True)
test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False, user_inference=True)

json_data = defaultdict(list)

for mode in ['val', 'test', 'train']:
    model.eval()
    loader = eval(mode + '_dataloader')

    with torch.no_grad():
        val_gts, val_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks, study_id, subject_id, image_path) in enumerate(
                tqdm(loader)):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = model(images, mode='sample')
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())


            for r, g, i_id, study, subject, i_p in zip(reports, ground_truths, images_id, study_id, subject_id,
                                                       image_path):
                json_data[mode].append(
                    {"id": str(i_id), "study_id": int(study), "subject_id": int(subject),
                     "report": str(r),
                     "ground_truth": str(g),
                     "image_path": i_p,
                     "split": mode})

json.dump(json_data, open("r2gen_mimic.json", "w"))
