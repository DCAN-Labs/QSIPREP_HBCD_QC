#The base image is QSIPrep
FROM pennbbl/qsiprep:0.20.0

#Make code and data directory
RUN mkdir /hbcd_code

#Copy code, assign permissions
ADD run.py /hbcd_code/run.py
RUN chmod 555 -R /hbcd_code
ENV PATH="${PATH}:/hbcd_code"
RUN pipeline_name=qsiprep_qc && cp /hbcd_code/run.py /hbcd_code/$pipeline_name

#Define entrypoint
ENTRYPOINT ["qsiprep_qc"]