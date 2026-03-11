<!--
README Template for a Research / Benchmark / Code Release Repository
Replace placeholders like <PROJECT_NAME>, <PAPER_TITLE>, <YEAR>, <HOMEPAGE_URL>, etc.
Remove any sections you do not need.
-->

<h1 align="center">TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions</h1>
<!-- <h3 align="center">VENUE_AND_YEAR</h3> -->

<div align="center">
  <a href="https://github.com/nereagallego" target="_blank">Nerea Gallego</a><sup>1,*</sup>,
  <a href="https://github.com/fsgaspar" target="_blank">Fernando Salanova</a><sup>1,*</sup>,
  <a href="https://github.com/claudiomann" target="_blank">Claudio Mannarano</a><sup>1,2</sup>,
  <a href="https://github.com/cmahulea" target="_blank">Cristian Mahulea</a><sup>1</sup>
  <a href="https://github.com/emontijano" target="_blank">Eduardo Montijano</a><sup>1</sup>
</div>

<div align="center">
  <sup>1</sup>University of Zaragoza, <sup>2</sup>Università di Torino
</div>

<div align="center">
  <sup>*</sup><i>Equal contribution</i>
</div>

<div align="center">
  <a href="https://ropertunizar.github.io/TIMID/"><strong>🌍 Homepage</strong></a> |
  <a href="https://huggingface.co/datasets/nereagallego/TIMID-data"><strong>🤗 Benchmarks </strong></a> |
  <a href="https://arxiv.org/abs/2603.09782"><strong>📝 Paper</strong></a> |
  <!-- <a href="DEMO_URL"><strong>🎬 Demo</strong></a> | -->
  <a href="https://huggingface.co/datasets/nereagallego/TIMID-data/tree/main/ckpt"><strong>🧠 Models</strong></a>
</div>

---

## 🔔 News
- 🆕 03/2026: Code released.
- ⭐ 03/2026: Benchmark / dataset released.
<!-- - 🥳 MM/YYYY: Paper accepted at VENUE.
- 🔧 MM/YYYY: Added support for MODEL/FEATURE.
- ⚠️ MM/YYYY: Breaking change: WHAT_CHANGED. -->

---

## 📖 Description
**TIMID** is a framework designed to identify and localize temporal mistakes in robotic tasks. Unlike standard action recognition, TIMID focuses on the timing and sequencing of robot executions, detecting when a robot deviates from a "correct" execution path in video streams.

**Key Features:**
- **Time-Sensitive Analysis:** Detects mistakes that are only evident when considering the duration and order of actions.

- **Benchmark Suite:** Includes a comprehensive set of "correct" vs "failed" robot execution videos for evaluation.


<!-- ```
https://ropertunizar.github.io/baserepo/
```

To customize the webpage:
1. Edit `docs/index.html` and replace the placeholder content (Author1, Author2, Project Title, etc.) with your actual project information
2. Add your figures and images to the `docs/img/` folder
3. Update the CSS in `docs/css/main.css` if you want to customize the styling
4. Make the repository public
5. Go to Settings/Pages, select branch -> main and folder -> /docs
6. Your project webpage will have this link: `https://ropertunizar.github.io/your_repo_name/` -->

## 🛠️ Requirements
The code is tested on Ubuntu 22.04 with Python 3.10 and CUDA 12.1.

```bash
# Clone the repository
git clone https://github.com/ropertunizar/TIMID.git
cd TIMID

# Create a virtual environment
python -m venv timid_env
source timid_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## 🚀 Usage
1. Data preparation
Data and pretrained models are alloceted in [Huggingface](https://huggingface.co/datasets/nereagallego/TIMID-data). You can download using the command line:
```
hf download nereagallego/TIMID-data --repo-type=dataset --local-dir .

```
To use the [Bridge dataset](https://rail-berkeley.github.io/bridgedata/), please download the first 1,000 episodes. This repository provides the necessary annotations.

2. Inference
To run a pre-trained model on one of the datasets:
```
python main.py --mode infer --model_mode 1 --ckpt_path ckpt/mutex/mutex__7683.pkl --dataset mutex #dataset:[mutex, ordering, bridge, mutex_real, ordering_real] mode:[train, infer] model_mode[1, 2, 3, 4]
```

3. Training
To train the model on the benchmark:
```
python main.py --mode train --model_mode 1 --dataset mutex #dataset:[mutex, ordering, bridge, mutex_real, ordering_real] mode:[train, infer] model_mode[1, 2, 3, 4]
```
Training/inference Mode 2, 3 and 4 correspond to the "Semantic Only", "Temporal Only" and ["PEL4VAD"](https://github.com/yujiangpu20/PEL4VAD/tree/master) configurations in the ablation study and baseline comparison, respectively.

## Results and Models

|Dataset | AP | AR | F1 | ckpt |
| -------- | -------- | -------- | -------- | -------- |
| Bridge | 49.72 | 33.77 | 40.22 | [link](https://huggingface.co/datasets/nereagallego/TIMID-data/blob/main/ckpt/bridge/bridge__4972.pkl) |
| Mutex | 76.83 | 35.89 | 40.1 | [link](https://huggingface.co/datasets/nereagallego/TIMID-data/blob/main/ckpt/mutex/mutex__7683.pkl) |
| Ordering | 48.71 | 36.89 | 33.45 | [link](https://huggingface.co/datasets/nereagallego/TIMID-data/blob/main/ckpt/ordering/ordering__4871.pkl) |
| Mutex Real | 72.01 | 23.64 | 23.91 | [link](https://huggingface.co/datasets/nereagallego/TIMID-data/blob/main/ckpt/mutex/mutex__7683.pkl) |
| Ordering Real | 19.87 | 12.12 | 7.92 | [link](https://huggingface.co/datasets/nereagallego/TIMID-data/blob/main/ckpt/ordering/ordering__4871.pkl) |

<div align="center">
  <h2>Results Overview</h2>
  <table style="width:100%; text-align:center; border:none;">
    <tr>
      <td width="33%">
        <img src="docs/img/CompBridgeGood364.gif" alt="Bridge Prediction" width="100%">
        <br>
        <sub><b>Bridge Prediction (Green is ours)</b></sub>
      </td>
      <td width="33%">
        <img src="docs/img/CompProximityBad2.gif" alt="Proximity Real Videos Prediction" width="100%">
        <br>
        <sub><b>Proximity Real Videos Prediction (Green is ours)</b></sub>
      </td>
      <td width="33%">
        <img src="docs/img/CompOrderingBad1.gif" alt="Ordering Real Videos Prediction" width="100%">
        <br>
        <sub><b>Ordering Real Videos Prediction (Green is ours)</b></sub>
      </td>
      
    </tr>
  </table>
</div>

## 📜 License
 This work is under AGPL-3.0 license.
 
## 📝 Citation
```bibtex
@inproceedings{gallego2026timid,
  title={TIMID: Time-Dependent Mistake Detection in Videos of Robot Executions},
  author={Gallego, Nerea and Salanova, Fernando and Mannarano, Claudio and Mahulea, Cristian and Montijano, Eduardo},
  year={2026}
}
```

## 🙏 Acknowledgements
This work was partially supported by grants AIA2025-163563-C31, PID2024-159284NB-I00, funded by MCIN/AEI/10.13039/501100011033 and ERDF, the Office of Naval Research Global grant N62909-24-1-2081 and DGA project T45\_23R, the work was also supperted by a 2024 DGA scholarship.
