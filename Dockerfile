# base.docker

FROM ubuntu:vivid

MAINTAINER Oscar Esteban <code@oscaresteban.es>

# Update packages
RUN apt-get update

# Install wget and git
RUN apt-get install -y wget curl git

# Enable neurodebian
RUN wget -O- http://neuro.debian.net/lists/vivid.de-m.full | tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN wget -O- http://neuro.debian.net/lists/vivid.us-tn.full >> /etc/apt/sources.list.d/neurodebian.sources.list
RUN apt-key adv --recv-keys --keyserver hkp://pgp.mit.edu:80 0xA5D32F012649A5A9
RUN apt-get update

# install supervisor
RUN apt-get -y install supervisor
ADD files/supervisord.conf /etc/supervisor/supervisord.conf
RUN mkdir -p /var/log/supervisor

# Install python-dev
RUN apt-get install -y python-dev

# Install pip
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

# Install afni and fsl
RUN apt-get install -y fsl afni ants

# Install dependencies
RUN apt-get install -y liblapack-dev libblas-dev libpng-dev libfreetype6 libfreetype6-dev libhdf5-dev libxml2-dev libxslt-dev

# pyzmq
RUN apt-get install -y libzmq-dev

# Virtual framebuffer for headless operation
RUN apt-get install -y xvfb
RUN pip install xvfbwrapper

RUN pip install lockfile
RUN pip install --upgrade numpy

# replace .bashrc
RUN mkdir -p /root/.local/bin
RUN mkdir -p /root/.local/lib
ADD files/bashrc /root/.bashrc

WORKDIR /scratch/src/
ADD files/run_mriqcp.sh run_mriqcp.sh
RUN chmod +x run_mriqcp.sh
RUN git clone https://github.com/oesteban/nipype.git && cd /scratch/src/nipype && git checkout exp/mriqc && git pull && pip install -e .
RUN pip install -e git+https://github.com/oesteban/quality-assessment-protocol.git@enh/SmartQCWorkflow#egg=qap
RUN git clone https://github.com/poldracklab/mriqc.git && cd /scratch/src/mriqc && git pull && pip install -e .


# run container with supervisor (from scivm/scientific-python-2.7)
CMD ["/usr/bin/supervisord"]