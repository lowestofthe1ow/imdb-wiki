# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN
# --method can be DEX, MVL, AMRL
# time uv run \
#     -m src.model.train.train_dex \
#     --batch-size 32 \
#     --workers 4 \
#     --min-epochs 20 \
#     --max-epochs 100 \
#     --method "AMRL" \
#     --dataset "IMDBWIKI" \
#     --data-path "/kaggle/input/imdb-wiki-cleaned" \
#     --outfile-prefix "AMRL_IMDBWIKI" \
    # --checkpoint-path "checkpoints/DEX_IMDBWIKI_epoch=26_val_loss=3.36.ckpt" \
    # --use-weighted-ce \

time uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "AMRL" \
    --dataset "APPAREAL" \
    --dataset-path "data/appa-real-release" \
    --outfile-prefix "AMRL_IMDBWIKI_APPAREAL" \
    --checkpoint-path "checkpoints/AMRL_IMDBWIKI_epoch=07_val_loss=10.19.ckpt"


time uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "AMRL" \
    --dataset "FAIRFACE" \
    --dataset-path "data/fairface" \
    --outfile-prefix "AMRL_IMDBWIKI_FAIRFACE" \
    --checkpoint-path "checkpoints/AMRL_IMDBWIKI_epoch=07_val_loss=10.19.ckpt"

time uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "AMRL" \
    --dataset "APPAREAL" \
    --dataset-path "data/appa-real-release" \
    --outfile-prefix "AMRL_IMDBWIKI_FAIRFACE_APPAREAL" \
    --checkpoint-path "checkpoints/AMRL_IMDBWIKI_FAIRFACE_epoch=06_val_loss=6.36.ckpt"

time uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "AMRL" \
    --dataset "CHALEARN" \
    --dataset-path "data/chalearn_2015" \
    --outfile-prefix "AMRL_IMDBWIKI_CHALEARN" \
    --checkpoint-path "checkpoints/AMRL_IMDBWIKI_epoch=07_val_loss=10.19.ckpt"
