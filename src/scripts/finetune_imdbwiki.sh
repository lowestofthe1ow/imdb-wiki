# echo Finetune DEX on AppaReal
# time uv run \
#     -m src.model.finetune.appareal \
#     --batch-size 64 \
#     --workers 4 \
#     --min-epochs 0 \
#     --max-epochs 100 \
#     --method DEX \
#     --csv-path data/appa-real-release \
#     --checkpoint-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt

# echo Finetune MVL on AppaReal
# time uv run \
#     -m src.model.finetune.appareal \
#     --batch-size 64 \
#     --workers 4 \
#     --min-epochs 0 \
#     --max-epochs 100 \
#     --method MVL \
#     --csv-path data/appa-real-release \
#     --checkpoint-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt

echo Finetune DEX on FairFace
time uv run \
    -m src.model.finetune.fairface \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method DEX \
    --csv-path data/fairface \
    --checkpoint-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt

echo Finetune MVL on FairFace
time uv run \
    -m src.model.finetune.fairface \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method MVL \
    --csv-path data/fairface \
    --checkpoint-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt

echo Finetune DEX-APPAREAL on FairFace
time uv run \
    -m src.model.finetune.fairface \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method DEX \
    --csv-path data/fairface \
    --checkpoint-path checkpoints/IMDBWIKI_DEX_epoch=09_val_loss=3.81.ckpt-FINETUNED-ON_APPAREAL_DEX_epoch=05_val_loss=2.52.ckpt

echo Finetune MVL-APPAREAL on FairFace
time uv run \
    -m src.model.finetune.fairface \
    --batch-size 64 \
    --workers 4 \
    --min-epochs 0 \
    --max-epochs 100 \
    --method MVL \
    --csv-path data/fairface \
    --checkpoint-path checkpoints/IMDBWIKI_MVL_epoch=10_val_loss=11.68.ckpt-FINETUNED-ON_APPAREAL_MVL_epoch=10_val_loss=17.03.ckpt