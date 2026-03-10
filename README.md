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
  <a target="_blank">Cristian Mahulea</a><sup>1</sup>
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
  <a href="DATASET_OR_BENCHMARK_URL"><strong>🤗 Benchmark / Dataset</strong></a> |
  <a href="ARXIV_OR_PAPER_URL"><strong>📝 Paper</strong></a> |
  <a href="DEMO_URL"><strong>🎬 Demo</strong></a> |
  <a href="MODEL_CARD_URL"><strong>🧠 Models</strong></a>
</div>

---

## 🔔 News
- 🆕 03/2026: Code released.
- ⭐ MM/YYYY: Benchmark / dataset released.
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

## 📜 License
 Discuss with your supervisor the license that you want to set and include the licenses of any previous repo in which your code was inspired.
 
## 📝 Citation
```bibtex
@inproceedings{CITATION_KEY,
  title={PAPER_TITLE},
  author={AUTHORS},
  booktitle={VENUE},
  year={YEAR}
}
```

## 🙏 Acknowledgements
Ask to your supervisor.
