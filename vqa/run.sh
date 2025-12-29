if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then    
    OUTPUT_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output"
    IMAGE_ROOT="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    OUTPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
    IMAGE_ROOT="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv"
fi

RUN_NAME="run_11_general"
QUANTITY="10K"

python convert_tsv.py ${OUTPUT_PATH}/${RUN_NAME}/test_${RUN_NAME}_${QUANTITY}.json ./${RUN_NAME}_${QUANTITY}.tsv \
    --output-dir ${OUTPUT_PATH}/${RUN_NAME} \
    --image-root ${IMAGE_ROOT}
