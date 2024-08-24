FROM python:3.10-slim

WORKDIR /opt/app-root/src/

COPY src/tat_hack_back/requirements.txt /opt/app-root/src/tat_hack_back/

RUN pip install --no-cache-dir -r /opt/app-root/src/tat_hack_back/requirements.txt

ENV PYTHONPATH /opt/app-root/src
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--config", "/opt/app-root/src/tat_hack_back/api/config/gunicorn.conf.py"]