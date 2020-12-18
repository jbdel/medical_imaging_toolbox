### Dataloaders
<b>```MimicDataset.py```</b> <br/>
JB version. Returns 
- a `key` that is a triple `(subject_id, study_id, dicom_id)`
- an image, provided `return_image` is True
- a label, provided `return_label` is True
- a report, provided `return_report` is True

Report consists of the `findings` section if any, otherwise consists of the `impression` section. If both findings and impression
are non-existent, the full example is discarded. 