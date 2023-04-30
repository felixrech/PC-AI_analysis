FROM python:3.10

ADD 04_dataset_access 04_dataset_access
ADD 05_topic_modeling 05_topic_modeling
ADD 06_explorer 06_explorer

WORKDIR 06_explorer/explorer
RUN git clone https://github.com/facebookresearch/fastText.git && cd fastText && pip install .
RUN pip install -r requirements.txt
CMD ["gunicorn", "--workers=4", "-b 0.0.0.0:8055", "-t 600", "--worker-class=sync", "--preload", "app:server"]