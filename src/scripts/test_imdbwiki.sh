mkdir results

echo test dex
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt \
    > results/dex.txt

echo test dex appareal
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt-FINETUNED-ON_APPAREAL_DEX_epoch=05_val_loss=2.52.ckpt \
    > results/dex-appareal.txt

echo test dex fairface
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt-FINETUNED-ON_FAIRFACE_DEX_epoch=18_val_loss=0.85.ckpt \
    > results/dex-fairface.txt

echo test dex appareal fairface
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt-FINETUNED-ON_APPAREAL_DEX_epoch=05_val_loss=2.52.ckpt-FINETUNED-ON_FAIRFACE_DEX_epoch=13_val_loss=0.90.ckpt \
    > results/dex-appareal-fairface.txt

echo test mvl
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt \
    > results/mvl.txt

echo test mvl appareal
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt-FINETUNED-ON_APPAREAL_MVL_epoch=10_val_loss=17.03.ckpt \
    > results/mvl-appareal.txt

echo test mvl fairface
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt-FINETUNED-ON_FAIRFACE_MVL_epoch=13_val_loss=14.70.ckpt \
    > results/mvl-fairface.txt


echo test mvl appareal fairface
time uv run \
    -m src.model.test.mean_absolute_error \
    --model-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt-FINETUNED-ON_APPAREAL_MVL_epoch=10_val_loss=17.03.ckpt-FINETUNED-ON_FAIRFACE_MVL_epoch=16_val_loss=14.39.ckpt \
    > results/mvl-appareal-fairface.txt


