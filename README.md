# RSNA Launches 2026 Knee Abnormality Detection AI Challenge

RSNA has launched the 2026 RSNA Knee Abnormality Detection AI Challenge. RSNA organizes AI challenges to spur the creation of AI tools for radiology and improve patient care.

The 2026 RSNA Knee Abnormality Detection AI Challenge will engage researchers to develop machine learning models to detect and classify abnormalities. The challenge will be the first to use both images and the text of radiology reports to train and test AI models.

> “What makes this year’s Knee Abnormality Detection AI Challenge unique is that it is the most real-world challenge yet,” said Po-Hao “Howard” Chen, MD, MBA, challenge co-leader, member of the RSNA Artificial Intelligence Committee and vice chair for Artificial Intelligence at the Cleveland Clinic Diagnostics Institute. “Participants must learn from real-world diagnostic radiology reports, where findings are complex and answers are not neatly organized in a table. This brings the challenge closer to how clinical AI must actually be developed: by confronting the nuance and variability of real radiologic interpretation."

MRI is essential to evaluating knee injuries because it reveals damage to joint structures, including ligaments, menisci and cartilage loss, providing a comprehensive whole-joint assessment. MRI can also detect abnormalities such as bone marrow lesions, effusion, synovitis and Baker cysts that cause symptoms and require management.

However, MRI interpretation varies across readers and practice settings and access to specialty-trained musculoskeletal radiologists is limited. By training AI models on a large dataset of knee MRIs paired with the radiologist’s report, this challenge aims to develop tools that can reliably identify abnormalities and support more consistent, timely care.

The models developed in the RSNA Knee Abnormality Detection AI Challenge will be evaluated on how accurately they detect and classify knee abnormalities on MRI. To construct these models, AI researchers need access to substantial volumes of imaging data annotated by expert radiologists. RSNA’s AI challenges engage the radiology community to develop such datasets, which provide the standard of truth in training AI systems to perform tasks relevant to diagnostic imaging.

> “The 2026 RSNA AI Challenge introduces several exciting firsts,” said Naveen Subhas, MD, MPH, challenge co-leader and vice chair of Clinical Operations for Cleveland Clinic Enterprise Imaging. “It is the first challenge to focus on musculoskeletal MRI, with knee MRI—the gold standard for diagnosing internal derangement of the knee—as its centerpiece. Reflecting the global reach of medical imaging, the dataset includes cases from countries across five continents. The challenge is also the first to combine medical images with multilingual radiology reports, creating a unique multimodal dataset. These new elements raise the bar for participants and are expected to inspire innovative approaches to developing the next generation of AI models.”

The training dataset includes more than 5,000 knee MRI exams and the associated radiology reports in a dozen different languages from 16 sites worldwide. The dataset used to assess model performance was annotated by expert radiologists.

In a challenge, researchers compete on how well their AI models perform specific tasks such as detection, localization and categorization of abnormal features according to defined performance measures. Each AI challenge explores and demonstrates the ways AI can benefit radiology and improve patient care.

The 2026 RSNA Knee Abnormality Detection AI Challenge is open to all researchers. The competition will be conducted on a platform provided by Kaggle, Inc. The top-performing teams will share in a total of $77,000 in prize money, including for the first time awards for the most efficient models.

Previous RSNA AI challenges have drawn thousands of expert AI researchers from around the world. The winning models are made available under open licenses to encourage additional research.

The competition will run through October 22, 2026. Winners will be announced in November, and winning teams will be recognized in the AI Theater during RSNA’s 112th Scientific Assembly and Annual Meeting (RSNA 2026), held Nov. 29–Dec. 3 at McCormick Place in Chicago.

## For More Information

Learn more about the 2026 RSNA Knee Abnormality Detection AI Challenge or contact informatics@rsna.org.

Read previous RSNA News stories on AI Challenge winners:

- Intracranial Aneurysm Detection AI Challenge Results Announced
- RSNA Announces Lumbar Spine Degenerative Classification AI Challenge Results
- RSNA Announces Abdominal Trauma Detection AI Challenge Results

## Win-Focused Execution Scaffold

This repository now includes a lightweight Python scaffold under:

- `/home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/src/rsna_knee`
- `/home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/tests`

It is designed to accelerate the first-prize workflow with reproducible building blocks:

- deterministic patient/site/language-aware CV split generation (`build-splits`)
- strict patient leakage validation
- macro-AUC + per-label AUC scoring for the 12 competition targets (`score-oof`)
- standardized Kaggle submission file assembly (`make-submission`)
- experiment tracker logging (`run_id`, split, model, macro AUC, public LB, train time, cost)

### Quickstart

Run all tests:

```bash
cd /home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge
python -m unittest discover -s tests -p "test_*.py"
```

Create CV splits:

```bash
cd /home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge
PYTHONPATH=src python -m rsna_knee.cli build-splits \
  --train-csv data/train.csv \
  --output-csv artifacts/splits/folds.csv \
  --n-folds 5 \
  --seed 2026
```

Score OOF predictions and log run metrics:

```bash
cd /home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge
PYTHONPATH=src python -m rsna_knee.cli score-oof \
  --oof-csv artifacts/oof/model_oof.csv \
  --tracker-csv artifacts/experiments/tracker.csv \
  --run-id baseline_img_text_v1 \
  --split-name cv5_patient_site_lang \
  --model-name multimodal_baseline
```

Build Kaggle submission:

```bash
cd /home/runner/work/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge/RSNA-Launches-Knee-Abnormality-Detection-AI-Challenge
PYTHONPATH=src python -m rsna_knee.cli make-submission \
  --ids-csv data/test_ids.csv \
  --preds-csv artifacts/preds/test_preds.csv \
  --output-csv submissions/submission.csv
```

### Competition target columns

`StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`
