# GLAD

## Setup

```bash
git clone [REPO_LINK]
cd glad/glad
```

Download the pretrained model of choice from [Zenodo](https://zenodo.org/record/5398393) and place it in `./models/weights/`.

### With Docker
Then:
```bash
docker build -t [IMAGE_NAME] .
```

### Without Docker
Prerequisites:
 * CUDA 10.2
 * python3.9, pip3.9
 * Java 8
 * [Defects4J](https://github.com/rjust/defects4j)

Finally, run:
```bash
pip install -r requirements.txt
mkdir -p /home/java-pred-synth/data/defects4j-buggy-projects
mkdir -p /home/java-pred-synth/data/finetune
```

## Execution
With setup done, first get the buggy defects4j project with:
```bash
sh defects4j-loader.sh [PROJ] [BUG_ID]
```

Then repair using:
```bash
python repair.py --project [PROJ] --bug_id [BUG_ID] \
                 --beam_width 10000 --search_len 12 \
                 --model_path [MODEL_PATH]
```

Or with docker:
```bash
docker run --rm --gpus all --name glad-eval minus21/glad \
    bash -c "sh defects4j-loader.sh [PROJ] [BUG_ID] && \
             python3.9 repair.py --project [PROJ] --bug_id [BUG_ID] \
                                 --beam_width 10000 --search_len 12 \
                                 --model_path [MODEL_PATH]"
```

A log will print, showing the repair process.

## Acknowledgements
 * `./etc_data/defects4j_bugs.json` is from [defects4j dissection](https://github.com/program-repair/defects4j-dissection).
 * `java_analyzer/sane_jdb/sane_jdb.jar` is a modification of the [JDB](https://docs.oracle.com/javase/7/docs/technotes/tools/windows/jdb.html) tool. Some modifications are listed below. The source code of `sane_jdb` will also be published upon acceptance.
   * Implementing boolean operators such as `&&`, `||`, and `!`
   * Fixing the buggy `==` operator (would crash when the lhs term was null)
