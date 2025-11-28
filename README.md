# Apparent Age Estimation: Challenges and Outcomes

> A Case Study on Deep EXpectation with the IMDB-WIKI Dataset

* This is the IMDB-WIKI case study and final project repository.
* It includes a replication implementation and an exploration on potential
  refinements to the IMDB-WIKI DEX architecture.
* We will be exploring using different loss functions over cross entropy (e.g.
  mean-variance loss, adaptive mean-residue loss) as well as further
  fine-tuning the model on the APPA-REAL and FairFace datasets and seeing if
  these modifications to the original architecture predict one's apparent age
  better compared to the original architecture.
* We will also try to measure the accuracy of these different architectures
  across each demographic group, split by race and sex, by using an
  off-the-shelf classifier.

## Getting the performance results:

1. Download `models.zip` from [here](https://drive.google.com/drive/folders/1dFLn0RZ2FlvU8WjqwJ-YrOMor_SwTPfP?usp=sharing) and unzip to the `checkpoints` directory. It should look like this:

```
checkpoints
├── Caffe_IMDBWIKI_CHALEARN.pth
├── Caffe_IMDBWIKI.pth
├── CEL_CHALEARN_epoch=05_val_loss=3.69.ckpt 
├── CEL_IMDBWIKI_APPAREAL_epoch=09_val_loss=2.96.ckpt
├── CEL_IMDBWIKI_CHALEARN_epoch=13_val_loss=2.81.ckpt
├── CEL_IMDBWIKI_epoch=26_val_loss=3.36.ckpt
├── checkpoint_captions.csv
├── MVL_IMDBWIKI_APPAREAL_epoch=06_val_loss=5.95.ckpt
├── MVL_IMDBWIKI_CHALEARN_epoch=04_val_loss=5.94.ckpt
└── MVL_IMDBWIKI_epoch=10_val_loss=10.92.ckpt
```

2. Get the datasets. Check the respective `.py` scripts in `src/datasets` for instructions, e.g.:

```
data
├── appa-real-release/
├── chalearn_2015/
├── fairface/
└── imdb-wiki/
```

3. Run `src/scripts.test_mae.sh`. This should create a `pkl` directory containing the raw results for each model as well as `embeddings` containing image embeddings at the last FC layers.

4. Run `python3 -m src.model.test.mean_absolute_error`. This will show you the performance results per race and gender

From here you can...

5. Run `python3 -m src.model.test.cosine_sim`. This will create a `cosine_sim.pkl` file at the root directory which will have cosine similarities for each image vs the mean embeddings for that image's apparent age, per model.

6. Run `python3 -m src.model.test.mae_per_demographic`. This will show you performance stats per demographic.

## How do I add models

1. Put the `.ckpt` in your `checkpoints/` dir
2. Add it to `checkpoints/checkpoint_captions.csv`.
