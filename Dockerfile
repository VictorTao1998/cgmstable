FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11 \
    apt install libopenexr-dev zlib1g-dev openexr \
    apt install xorg-dev  # display widows \
    apt install libglfw3-dev \
    pip install attrdict>=2.0.1 h5py>=2.10.0 imageio>=2.6.1 \
    pip install git+https://github.com/aleju/imgaug numpy>=1.17.3 \
    pip install opencv-python==4.1.2.30 \
    pip install git+https://github.com/jamesbowman/openexrpython.git \
    pip install oyaml>=0.9 pathlib>=1.0.1 Pillow>=6.2.0 scikit-image>=0.15.0 scipy>=1.3.1 \
    pip install Shapely>=1.6.4.post2 tensorboardX>=1.9 tensorflow>=1.14.0 termcolor>=1.1.0 torch>=1.3.0 torchvision>=0.4.1 \
    pip install tqdm>=4.36.1 open3d>=0.8.0.0    