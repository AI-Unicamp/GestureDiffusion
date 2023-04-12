while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container MDM_container_$number on gpu $gpu and port $port";

nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64G -v /work/rodolfo.tonoli/motion-diffusion-model:/workspace/motion-diffusion-model/ -p $port --name MDM_container$number multimodal-research-group-mdm:latest /bin/bash
