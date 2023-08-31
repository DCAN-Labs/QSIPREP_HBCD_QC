#The base image is FreeSurfer's synthstrip package
FROM python:3.9.16-slim-bullseye

#Install relevant python packages
RUN python3 -m pip install nibabel==3.2.2
RUN python3 -m pip install dipy==1.6.0
RUN python3 -m pip install matplotlib==3.3.4
RUN python3 -m pip install scipy==1.8.0
RUN python3 -m pip install pybids=0.16.3
RUN python3 -m pip install imageio=2.31.2


#Make code and data directory
RUN mkdir /hbcd_code

#Copy code, assign permissions
ADD run.py /hbcd_code/run.py
RUN chmod 555 -R /hbcd_code
ENV PATH="${PATH}:/hbcd_code"
RUN pipeline_name=qsiprep_qc && cp /hbcd_code/run.py /hbcd_code/$pipeline_name

#Define entrypoint
ENTRYPOINT ["qsiprep_qc"]