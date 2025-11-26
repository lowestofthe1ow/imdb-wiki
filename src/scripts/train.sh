# --dataset can be IMDBWIKI, APPAREAL, or CHALEARN

uv run \
    -m src.model.train.train_dex \
    --batch-size 32 \
    --workers 4 \
    --min-epochs 20 \
    --max-epochs 100 \
    --method "DEX" \
    --dataset "APPAREAL" \
    --dataset-path "data/appa-real-release/" \
    --outfile-prefix "CEL_IMDBWIKI_FAIRFACE_APPAREAL" \
    --checkpoint-path "checkpoints/CEL_IMDBWIKI_FAIRFACE_epoch=23_val_loss=1.05.ckpt" \
    # --use-weighted-ce \

