# ReadBench Benchmark

A straightforward benchmark to measure how much performance degrades when using multimodal inputs rather than purely textual inputs: can your VLM read and reason about text?

## Quick Start

### Setup

#### Step-by-step

1. **Install dependencies:**

```bash
  [uv] pip install -r requirements.txt
```

2. **Set up API keys:**

If you're planning on using any API-based models, make sure you define your relevant API keys in the .env.

3. **Prepare raw ReadBench**

The images and text are stored on the HuggingFace hub, as a .zip. You may download it directly from there, using `huggingface-cli` (recommended):

```bash
huggingface-cli download answerdotai/ReadBench readbench.zip --type dataset
```

Alternatively, if you are unable to use `huggingface-cli`, you may use the direct download URL, as provided by HuggingFace:

```bash
wget https://huggingface.co/answerdotai/ReadBench/resolve/main/readbench.zip?download=true -O readbench.zip
```

You will then want to unzip the downloaded folder:

```bash
unzip readbench.zip
```

4. **Prepare GPQA**

The authors of GPQA have requested that the dataset should not be reshared as-is, to minimise model contamination. We follow their wishes, which means you need to generate the GPQA images yourself, absed on the original GPQA dataset. You can do so by running the following commands:
```bash
python data_prep.py --datasets gpqa
```

5. **Prepare the benchmark**

You may now run the following command to prepare the metadata file which will be used to run the benchmark:

```bash
python downsampler.py --root rendered_images_ft12 --split standard
```

#### tl;dr

Running the commands below will download and prepare the full ReadBench benchmark, as used in the paper:

```bash
huggingface-cli download answerdotai/ReadBench readbench.zip --type dataset
unzip readbench.zip
python data_prep.py --datasets gpqa
python downsampler.py --root rendered_images_ft12 --split standard
```



### Running ReadBench

  Evaluating a model on the benchmark then requires using the `run_eval.py` script with the newly created metadata path, for example:

```bash
python run_eval.py readbench_meta/readbench_8k-rendered_images_ft12-nano_metadata.json \
  --model gemini-2.0-flash \
  --mode all \
  --workers 16
```


## Supported Models Out of the Box

- **Gemini**: All models compatible with the Gemini API as of its May 2025 specs.
- **Claude**: All models compatible with the Anthropic and Vertex API as of its May 2025 specs.
- **OpenAI**: All models compatible with the OpenAI API as of its May 2025 specs.
- **Mistral**: Pixtral models, using the May 2025 API. Will need updated if the API supports longer images inputs (currently capped at 8)
- **Local VLLM-Served Qwen**: You'll need to spin up a VLLM instance. This function can be modified to support any VLLM hosted model.

## Results Structure

```
readbench_results/
├── {model}/{ppi}/{split}/cotdefault/
│   ├── text_{model}_cotdefault.json           # Full text results
│   ├── multimodal_{model}_cotdefault.json     # Full multimodal results  
│   ├── dataset-{dataset}_{model}_cotdefault.json  # Per-dataset summaries
│   └── overview_{model}.json                  # Overall summary
```

Example: `readbench_results/gemini-2.0-flash/93/nano/cotdefault/`

## Going further

Documentation TBC, but the repository supports:
- Three sizes of the data, the standard one, a **nano** size, and an **extended** size. These can be generated with different `data_prep.py` arguemnts.
- CoT experiments: you may run in standard mode (by default), or in inverted mode, where datasets that normally use CoT have it disabled and those that do not have it enabled, to check for CoT impact. You may look at the different `run_eval.py` cli parameters to understand more about this.
- Each run archives previous results to `run_{x}/` folders, for variance analysis purposes.