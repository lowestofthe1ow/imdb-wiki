mkdir results

echo test dex
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt \
    > results/dex.txt

echo test mvl
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt \
    > results/mvl.txt

echo test dex appareal
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt-FINETUNED-ON_APPAREAL_DEX_epoch=05_val_loss=2.52.ckpt \
    > results/dex-appareal.txt

echo test mvl appareal
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt-FINETUNED-ON_APPAREAL_MVL_epoch=10_val_loss=17.03.ckpt \
    > results/mvl-appareal.txt
